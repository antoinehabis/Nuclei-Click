import torch
from autoencoder.dataloader import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from create_gt_grids import Gtgrid
from sklearn.metrics import pairwise_distances
from utils import *
import numpy as np
from config import *



class FindSame2:
    def __init__(self, df, model, threshold_images, threshold_nucleis, baseline):
        dataset = CustomImageDataset(
            path_images=path_images, dataframe=df, augmentation=False
        )
        loader_test = DataLoader(
            batch_size=32, dataset=dataset, num_workers=16, shuffle=False
        )

        self.model = model
        self.df = df
        self.dic_grids_errors = None
        self.scales_to_use = None
        self.receptive_field_sizes = np.array([5, 13, 29, 61, 125])
        self.threshold_images = threshold_images
        self.threshold_nucleis = threshold_nucleis

        f0, f1, f2, f3, f4 = self.predict(model, loader_test)
        self.all_f0s = np.concatenate(f0)
        self.all_f1s = np.concatenate(f1)
        self.all_f2s = np.concatenate(f2)
        self.all_f3s = np.concatenate(f3)
        self.all_f4s = np.concatenate(f4)
        self.dic_errors_near = {}

    def predict(self, model, dl):
        features1, features2, features3, features4, features5 = (
            [],
            [],
            [],
            [],
            [],
        )
        model.eval()
        model.cuda()
        loss_tot = 0.0
        with torch.no_grad():
            for batch in tqdm(dl):
                inputs = batch[0].cuda()
                _, feature1, feature2, feature3, feature4, feature5 = model(inputs)

                features1.append(feature1.cpu().detach().numpy())
                features2.append(feature2.cpu().detach().numpy())
                features3.append(feature3.cpu().detach().numpy())
                features4.append(feature4.cpu().detach().numpy())
                features5.append(feature5.cpu().detach().numpy())

        return features1, features2, features3, features4, features5

    def extract_errors(self, index):
        self.index = index
        self.filename_support = self.df.iloc[self.index].filename
        self.image_support = (
            tifffile.imread(os.path.join(path_images, self.filename_support)) / 255
        )
        self.img_baseline = tifffile.imread(
            os.path.join(path_stardist, "baseline", self.filename_support)
        )
        self.img_gt = tifffile.imread(
            os.path.join(path_gt, "baseline", self.filename_support)
        )
        self.G = Gtgrid(self.img_gt, self.img_baseline, bool_remove_borders=True)
        filename_support = self.df.iloc[self.index].filename
        self.dic_grids_errors, self.scales_to_use = self.G.create_dic_errors()

    def show_errors_loc(self):
        count = 0
        image = self.image_support.copy()
        for key, value in self.dic_grids_errors.items():
            for array in value:
                grid_size = array.shape[0]
                for centers in np.argwhere(array):
                    scale = np.log2(256 // grid_size).astype(int)
                    width = self.receptive_field_sizes[scale]
                    centers = ((centers * 256 / grid_size)).astype(int)
                    image = draw_red_square(
                        self.image_support,
                        [centers[1], centers[0]],
                        width,
                        key=key,
                        count=count,
                    )
                    count += 1

        fig = plt.figure(figsize=(10, 7))
        # setting values to rows and column variables
        rows = 1
        columns = 3

        # Adds a subplot at the 1st position
        fig.add_subplot(rows, columns, 1)

        # showing image
        plt.imshow(image)
        plt.axis("off")
        plt.title("raw image with bounding box of errors")

        # Adds a subplot at the 2nd position
        fig.add_subplot(rows, columns, 2)

        # showing image
        plt.imshow(self.img_gt)
        plt.axis("off")
        plt.title("ground truth segmentation")

        # Adds a subplot at the 3rd position
        fig.add_subplot(rows, columns, 3)

        # showing image
        plt.imshow(self.img_baseline)
        plt.axis("off")
        plt.title("baseline segmentation")
        plt.show()

    def select_images_near(self):
        f5s = np.mean(self.all_f4s, axis=(-1, -2))
        pdist5 = pairwise_distances(f5s)
        inferior_to_threshold = np.argwhere(
            pdist5[self.index] < self.threshold_images
        ).flatten()
        argsort = np.argsort(pdist5[self.index])
        self.sub_dataset = [i for i in argsort if i in inferior_to_threshold]

        ## select_features of features of the corresponding images
        sub_f0s = self.all_f0s[self.sub_dataset]
        sub_f1s = self.all_f1s[self.sub_dataset]
        sub_f2s = self.all_f2s[self.sub_dataset]
        sub_f3s = self.all_f3s[self.sub_dataset]
        sub_f4s = self.all_f4s[self.sub_dataset]

        dic_scale_features = {}
        dic_scale_features[0] = sub_f0s
        dic_scale_features[1] = sub_f1s
        dic_scale_features[2] = sub_f2s
        dic_scale_features[3] = sub_f3s
        dic_scale_features[4] = sub_f4s
        self.dic_scale_features = dic_scale_features

        self.filenames_near = [
            self.df.iloc[index].filename for index in self.sub_dataset
        ]

        self.images_near = np.array(
            [
                tifffile.imread(os.path.join(path_images, u)) / 255
                for u in self.filenames_near
            ]
        )
        self.grids = np.zeros((len(self.images_near), 256, 256, 4))
        self.images_near_nuclei = []

    def find_nearest_nucleis(self, norm, threshold_nuclei):
        # print(norm.shape)
        args = np.argsort(norm, axis=None)
        q1, r1 = np.divmod(args, norm.shape[-1] * norm.shape[-2])
        q2, r2 = np.divmod(r1, norm.shape[-2])
        indexes = np.sum(norm.flatten() < threshold_nuclei)
        return np.stack([q1, q2, r2]).T[:indexes]

    def find_same_nuclei(self):
        self.filenames_nuclei_near = []
        count = 0
        images_with_same_errors_detected = []
        for i, key in enumerate(self.dic_grids_errors.keys()):
            """Go through the 4 different class of errors clicked by the user"""
            for grid in self.dic_grids_errors[key]:
                """For each type of error we go through the grids that indicate the position of the errors"""

                for errors in np.argwhere(grid):
                    """for each grid we go through the positions"""

                    scale = np.log2(256 // grid.shape[0])
                    x = self.dic_scale_features[scale.astype(int)]
                    sub_features = torch.tensor(x)

                    feature_support = sub_features[0, :, errors[0], errors[1]]
                    roll_features = torch.transpose(sub_features, 0, 1)

                    norm = torch.norm(
                        roll_features - feature_support[:, None, None, None], dim=0
                    ) / (torch.norm(feature_support))
                    feature_size = feature_support.shape[0] // 16
                    norm = norm.cpu().detach().numpy()
                    grid_size = norm.shape[-1]
                    args = self.find_nearest_nucleis(
                        norm, self.threshold_nucleis * feature_size
                    )
                    images_with_same_errors_detected.append(args[:, 0])

                    # factor_resize = (256 / (2 ** (scale)) - 1) / (256 - 1)

                    for nuclei in args:
                        nuclei[1:] = (2 ** (scale) * (nuclei[1:] + 0.5)).astype(int)
                        self.grids[nuclei[0], nuclei[1], nuclei[2], i] = 1
                        width = self.receptive_field_sizes[
                            np.log2(256 // grid_size).astype(int)
                        ]
                        self.images_near_nuclei.append(
                            draw_red_square(
                                self.images_near[nuclei[0]],
                                [nuclei[2], nuclei[1]],
                                width,
                                key=key,
                                count=count,
                            )
                        )
                    count += 1
        if len(images_with_same_errors_detected) > 0:
            images_with_same_errors_detected = np.unique(
                np.concatenate(images_with_same_errors_detected)
            )
        return (
            self.images_near,
            images_with_same_errors_detected,
            self.sub_dataset,
            self.grids,
            self.filenames_near,
        )
