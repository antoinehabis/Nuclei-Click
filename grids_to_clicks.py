import os
import sys
import numpy as np
import scipy
import tifffile
from config import *
from tqdm import tqdm
from scipy.stats import multivariate_normal
from skimage.segmentation import relabel_sequential
from numba import jit


@jit(nopython=True)
def get_positions(labels):
    n = labels.max() + 1
    positions = np.zeros((n, 2), dtype=np.float32)

    m_00 = np.zeros(n, dtype=np.uint)
    m_01 = np.zeros(n, dtype=np.uint)
    m_10 = np.zeros(n, dtype=np.uint)

    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            m_00[labels[i, j]] += 1
            m_01[labels[i, j]] += i
            m_10[labels[i, j]] += j

    positions[:, 0] = m_01 / m_00
    positions[:, 1] = m_10 / m_00

    return positions


class Grid_to_click:
    def __init__(self, grid, filename, path_baseline):
        self.filename = filename
        self.img_baseline = tifffile.imread(
            os.path.join(path_baseline,'baseline',self.filename)
        ).astype(np.uint8)
        self.img_baseline = relabel_sequential(self.img_baseline)[0]
        self.nucleis, self.counts = np.unique(self.img_baseline, return_counts=True)
        self.positions = get_positions(self.img_baseline).T[:, 1:]

        self.grid_merge = grid[:,:,0]
        self.grid_split = grid[:,:,1]
        self.grid_fn = grid[:,:,2]
        self.grid_fp = grid[:,:,3]

    def grid_to_click(self, which="fp"):
        if which == "fp":
            grid = self.grid_fp
        else:
            grid = self.grid_merge
        size = len(np.where(grid)[0])

        print(which, size)

        detections = np.zeros((2, size))

        ligns, columns = np.where(grid)
        detections = (np.stack([ligns, columns], axis=0)).astype(int)
        im = np.zeros((parameters["dim"], parameters["dim"]))

        if detections.size == 0:
            return im
        else:
            if len(self.nucleis) == 1:
                return im
            else:
                nucleis_target = []

                for i in range(size):
                    nucleis_target.append(
                        np.argmin(
                            np.sqrt(
                                np.sum(
                                    (
                                        self.positions
                                        - np.expand_dims(detections[:, i], axis=-1)
                                    )
                                    ** 2,
                                    axis=0,
                                )
                            )
                        )
                    )
                im = np.zeros((parameters["dim"], parameters["dim"]))

                for nuclei in nucleis_target:
                    obj = self.img_baseline == self.nucleis[nuclei + 1]
                    ligns, columns = np.where(obj)[0:2]

                    mean_ligns = np.mean(ligns)
                    mean_columns = np.mean(columns)
                    mean = np.array([mean_ligns, mean_columns])
                    cov = np.cov(ligns, columns) 
                    try:
                        if (
                            not (np.isnan(cov).any())
                            and len(np.unique(ligns)) > 1
                            and len(np.unique(columns)) > 1
                        ):
                            function = multivariate_normal(mean, cov)
                            img = np.swapaxes(
                                function.pdf(
                                    np.indices((parameters["dim"], parameters["dim"])).T
                                ),
                                0,
                                1,
                            )
                            img = img / np.max(img)
                            im = np.maximum(im, img)
                    except: 
                        pass
        return im

    def split_to_click(self):
        size = len(np.where(self.grid_split)[0])
        print("split", size)
        detections = np.zeros((2, size))

        ligns, columns = np.where(self.grid_split)
        detections = (np.stack([ligns, columns], axis=0)).astype(int)
        im = np.zeros((parameters["dim"], parameters["dim"]))
        if detections.size == 0:
            return im
        else:
            if len(self.nucleis) <= 2:
                return im

            else:
                nucleis_target = []

                for i in range(size):
                    nucleis_target.append(
                        np.argsort(
                            np.sqrt(
                                np.sum(
                                    (
                                        self.positions
                                        - np.expand_dims(detections[:, i], axis=-1)
                                    )
                                    ** 2,
                                    axis=0,
                                )
                            )
                        )[:2]
                    )

                for nuclei in nucleis_target:
                    nuclei1 = nuclei[0]
                    nuclei2 = nuclei[1]

                    obj1 = self.img_baseline == self.nucleis[nuclei1 + 1]
                    obj2 = self.img_baseline == self.nucleis[nuclei2 + 1]
                    ligns, columns = np.where(np.logical_or(obj1,obj2))[0:2]
                    mean_ligns = np.mean(ligns)
                    mean_columns = np.mean(columns)
                    mean = np.array([mean_ligns, mean_columns])
                    cov = np.cov(ligns,columns) 

                    if (
                        not (np.isnan(cov).any())
                        and len(np.unique(ligns)) > 1
                        and len(np.unique(columns)) > 1
                    ):
                        function = multivariate_normal(mean, cov)
                        img = np.swapaxes(
                            function.pdf(
                                np.indices((parameters["dim"], parameters["dim"])).T
                            ),
                            0,
                            1,
                        )
                        img = img / np.max(img)
                        im = np.maximum(im, img)

        return im

    def fn_to_click(self):
        size = len(np.where(self.grid_fn)[0])
        print("fn", size)
        im = np.zeros((parameters["dim"], parameters["dim"]))
        ligns, columns = np.where(self.grid_fn)
        detections = (np.stack([ligns, columns], axis=0)).astype(int)

        if detections.size == 0:
            return im
        else:
            if detections.shape[1] > 0:
                if len(self.counts) == 1:
                    mean_radius = 15
                else:
                    mean_radius = np.sqrt(np.median(self.counts[1:]) / np.pi)

                for i in range(detections.shape[1]):
                    mean = detections[:, i]
                    function = scipy.stats.multivariate_normal(
                        mean, np.eye(2) * mean_radius * 3
                    )
                    img = np.swapaxes(
                        function.pdf(
                            np.indices((parameters["dim"], parameters["dim"])).T
                        ),
                        0,
                        1,
                    )
                    img = img / np.max(img)
                    im = np.maximum(im, img)

        return im

    def final_click(self):
        click_fp = self.grid_to_click(which="fp")
        click_merge = self.grid_to_click(which="merge")
        click_split = self.split_to_click()
        click_fn = self.fn_to_click()

        click = np.stack([click_merge, click_split, click_fn, click_fp], axis=-1)

        return click
