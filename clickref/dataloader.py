import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from torch.utils.data import DataLoader, Dataset
import numpy as np
from config import *
import tifffile
import torch

path_baseline = path_stardist_modified


class CustomImageDataset(Dataset):
    def __init__(self, path_baseline, path_images, path_gt, dataframe, augmenter_bool):
        self.path_gt = path_gt
        self.path_baseline = path_baseline
        self.path_images = path_images
        self.dataframe = dataframe
        self.indices = self.dataframe.index.tolist()
        self.augmenter_bool = augmenter_bool

    def __len__(self):
        return len(self.dataframe)

    def augmenter(self, image, click, baseline, output):
        k = np.random.choice([1, 2, 3])

        image = np.rot90(image, k=k, axes=(0, 1))
        click = np.rot90(click, k=k, axes=(0, 1))
        baseline = np.rot90(baseline, k=k, axes=(0, 1))
        output = np.rot90(output, k=k, axes=(0, 1))
        alea_shift1 = np.random.random()
        alea_shift2 = np.random.random()

        if alea_shift1 > 0.5:
            image = np.flipud(image)
            click = np.flipud(click)
            baseline = np.flipud(baseline)
            output = np.flipud(output)

        if alea_shift2 > 0.5:
            image = np.fliplr(image)
            click = np.fliplr(click)
            baseline = np.fliplr(baseline)
            output = np.fliplr(output)

        return image, click, baseline, output

    def __getitem__(self, idx):
        filename = self.dataframe.iloc[idx]["filename"]
        image = tifffile.imread(os.path.join(self.path_images, filename)) / 255
        click = tifffile.imread(os.path.join(self.path_baseline, "click", filename))

        """ Baseline segmentation """

        contour_baseline = tifffile.imread(
            os.path.join(self.path_baseline, "contour", filename)
        ).astype(float)

        binary_baseline = tifffile.imread(
            os.path.join(self.path_baseline, "binary", filename)
        )
        contour_baseline = contour_baseline.astype(np.float32)
        binary_baseline = ((binary_baseline - contour_baseline) > 0).astype(np.float32)
        baseline = np.expand_dims(
            np.zeros((parameters["dim"], parameters["dim"]))
            + contour_baseline
            + 2 * binary_baseline,
            -1,
        )

        """ Ground truth """

        contour_gt = tifffile.imread(
            os.path.join(self.path_gt, "contour", filename)
        ).astype(float)
        binary_gt = tifffile.imread(
            os.path.join(self.path_gt, "binary", filename)
        ).astype(float)

        binary_gt = ((binary_gt - contour_gt) > 0).astype(float)
        gt = np.expand_dims(
            np.zeros((parameters["dim"], parameters["dim"]))
            + contour_gt
            + 2 * binary_gt,
            -1,
        )

        """ Augmenter """

        if self.augmenter_bool:
            image, click, baseline, gt = self.augmenter(image, click, baseline, gt)

        # image = np.concatenate((image, click), axis=-1)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        baseline = np.transpose(baseline, (2, 0, 1)).astype(np.float32)
        gt = np.transpose(gt, (2, 0, 1)).astype(np.int64)
        click = np.transpose(click, (2, 0, 1)).astype(np.float32)
        # print(image;shape)
        baseline_one_hot = torch.nn.functional.one_hot(torch.tensor(baseline,dtype = torch.int64).squeeze(),num_classes=3)
        # print(baseline_one_hot.shape)
        baseline_one_hot = torch.moveaxis(baseline_one_hot, -1, 0).to(torch.float32)
        return (
            torch.tensor(image),
            baseline_one_hot,
            torch.tensor(gt),
            torch.tensor(click.copy()),
        )


dataset_train = CustomImageDataset(
    path_baseline=path_baseline,
    path_images=path_images,
    path_gt=path_gt,
    dataframe=df_train,
    augmenter_bool=True,
)


# dataset_test = CustomImageDataset(
#     path_baseline=path_baseline,
#     path_images=path_images,
#     path_gt=path_gt,
#     dataframe=df_test,
#     augmenter_bool=False,
# )


dataset_val = CustomImageDataset(
    path_baseline=path_baseline,
    path_images=path_images,
    path_gt=path_gt,
    dataframe=df_val,
    augmenter_bool=False,
)

loader_train = DataLoader(
    batch_size=parameters["batch_size"],
    dataset=dataset_train,
    num_workers=16,
    shuffle=True,
)

loader_val = DataLoader(
    batch_size=parameters["batch_size"],
    dataset=dataset_val,
    num_workers=16,
    shuffle=False,
)

# loader_test = DataLoader(
#     batch_size=parameters["batch_size"],
#     dataset=dataset_test,
#     num_workers=4,
#     shuffle=False,
# )

dataloaders = {"train": loader_train, "val": loader_val}
