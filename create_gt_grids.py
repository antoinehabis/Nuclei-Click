import numpy as np
import tifffile
from scipy import ndimage
from scipy.ndimage.morphology import binary_opening
from skimage import measure
from skimage.morphology import disk
import warnings
from config import *
import warnings
import matplotlib.pyplot as plt
import itertools



class Gtgrid:
    def __init__(self, img_gt, img_baseline, area=200, bool_remove_borders=False):
        self.img_gt = img_gt
        self.img_baseline = img_baseline
        self.bool_remove_borders = bool_remove_borders

        if self.bool_remove_borders:
            self.img_gt = self.remove_nuclei_border(self.img_gt)
            self.img_baseline = self.remove_nuclei_border(self.img_baseline)

    def remove_nuclei_border(self, img, margin=0):
        uniques, counts = np.unique(img, return_counts=True)
        for rep in uniques:
            tmp = np.where(img == rep)
            x_min, x_max = np.min(tmp[0]), np.max(tmp[0])
            y_min, y_max = np.min(tmp[1]), np.max(tmp[1])
            bool_erase = (
                (x_min <= margin)
                or (x_max >= img.shape[0] - margin)
                or (y_min <= margin)
                or (y_max >= img.shape[0] - margin)
            )
            bool_erase = bool_erase
            if bool_erase:
                img[img == rep] = 0
        return img

    def create_dictionnary(self):

        dic_pure_associations = {}
        """
        find good associations first

        """
        uniques_gt = np.unique(self.img_gt)[1:]
        uniques_baseline = np.unique(self.img_baseline)[1:]

        for unique_gt in uniques_gt:
            for unique_baseline in uniques_baseline:
                nuclei_baseline = self.img_baseline == unique_baseline
                nuclei_gt = self.img_gt == unique_gt
                iou = np.sum(nuclei_baseline * nuclei_gt) / np.sum(
                    np.maximum(nuclei_baseline, nuclei_gt)
                )
                if iou > 0.5:
                    dic_pure_associations[unique_gt] = unique_baseline

        """
        dictionnary contains all the good associations
        keys are the ground truth 
        values are the baseline

        """
        dic_merges = {}

        for nb_nuclei_baseline in uniques_baseline:
            nuclei_baseline = self.img_baseline == nb_nuclei_baseline
            l = []
            for nb_nuclei_gt in uniques_gt:
                nuclei_gt = self.img_gt == nb_nuclei_gt

                iou = np.sum(nuclei_baseline * nuclei_gt) / np.sum(nuclei_gt)

                if iou >= 0.5:
                    l.append(nb_nuclei_gt)

            if len(l) >= 2:
                for element in l:
                    dic_merges[element] = nb_nuclei_baseline

        """cherchons si stardist n'a pas splité des noyaux"""
        dic_splits = {}

        for nb_nuclei_gt in uniques_gt:
            nuclei_gt = self.img_gt == nb_nuclei_gt
            l = []
            for nb_nuclei_baseline in uniques_baseline:
                nuclei_baseline = self.img_baseline == nb_nuclei_baseline
                iou = np.sum(nuclei_gt * nuclei_baseline) / np.sum(nuclei_baseline)
                if iou >= 0.5:
                    l.append(nb_nuclei_baseline)

            if len(l) >= 2:
                dic_splits[nb_nuclei_gt] = l

        splits = list(dic_splits.keys())
        baseline_splits = list(itertools.chain(*list(dic_splits.values())))
        merges = list(dic_merges.values())
        gt_merges = list(dic_merges.keys())

        fn = list(set(uniques_gt) - set(list(dic_pure_associations.keys())) - set(gt_merges) - set(splits))
        fp = list(set(uniques_baseline) - set(list(dic_pure_associations.values())) - set(baseline_splits) - set(merges))

        return (merges,
                splits,
                fn,
                fp,
        )

    def get_size_of_error(self, binary, margin=15):
        indexes = np.argwhere(binary)
        bottom_left = np.min(indexes, 0)
        upper_right = np.max(indexes, 0)
        return np.max(np.abs(bottom_left - upper_right)) + margin

    def flatten(self, list_):
        new_list = []
        for v in list_:
            if type(v) != list:
                new_list.append(v)
            else:
                new_list = new_list + v
        return new_list

    def create_dic_errors(self):
        all_errors = self.create_dictionnary()
        centers = []
        sizes = []
        receptive_field_sizes = np.array([5, 13, 29, 61, 125])

        keys = ["merge", "split", "fn", "fp"]
        dic = {}
        for key, errors, img in zip(
            keys,
            all_errors,
            (self.img_baseline, self.img_gt, self.img_gt, self.img_baseline),
        ):
            scale1 = np.zeros((256, 256))
            scale2 = np.zeros((128, 128))
            scale3 = np.zeros((64, 64))
            scale4 = np.zeros((32, 32))
            scale5 = np.zeros((16, 16))

            scales = [scale1, scale2, scale3, scale4, scale5]

            for nb in errors:
                error = img == nb
                rows, columns = np.where(error)[0:2]
                min_rows, max_rows = np.min(rows), np.max(rows)
                min_columns, max_columns = np.min(columns), np.max(columns)
                mean_rows = (max_rows + min_rows) // 2
                mean_columns = (max_columns + min_columns) // 2
                size = self.get_size_of_error(error)
                scale = np.minimum(np.searchsorted(receptive_field_sizes, size), 4)
                factor_resize = 1 / (2 ** (scale))
                row, col = np.round(factor_resize * mean_rows).astype(int), np.round(
                    factor_resize * mean_columns
                ).astype(int)

                row, col = np.clip(row, 0, 256 // (2**scale) - 1), np.clip(
                    col, 0, 256 // (2**scale) - 1
                )

                scales[scale][row, col] = 1
            scales = [u for u in scales if np.sum(u) > 0]
            dic[key] = scales

        all_scales_used = []
        for key, value in dic.items():
            for l in value:
                all_scales_used.append(l.shape[0])
        return dic, np.unique(all_scales_used)

    def create_grid(self):
        dic = self.create_dic_errors()[0]
        u = np.zeros((256, 256, 4))
        for i, (error, array) in enumerate(dic.items()):
            u_error = np.zeros((256, 256))
            if len(array) != 0:
                for arr in array:
                    factor_resize = 256 / arr.shape[0]
                    coord_image = (
                        (np.stack(np.where(arr)).T + 1 / 2) * factor_resize
                    ).astype(int)
                    for coord in coord_image:
                        u_error[coord[0], coord[1]] = 1
            u[:, :, i] = u_error
        return u
