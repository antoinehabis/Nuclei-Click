import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile
import os
from config import *
from skimage.morphology import (
    area_opening,
    binary_opening,
    binary_closing,
    binary_dilation,
)
from skimage.morphology import disk
from skimage.measure import label
from matching import matching


def remove_wrong_annotation(img_gt):
    nuclei, count = np.unique(img_gt, return_counts=True)
    args = np.argwhere(count < 5)
    for arg in args:
        img_gt[img_gt == arg] = 0
    return img_gt


def AJI(crop_img_new_seg, crop_img_gt):
    Is = 0
    Us = 0
    intersection = 0
    union = 0

    uniques = np.unique(crop_img_new_seg)[1:]
    uniques_true = np.unique(crop_img_gt)[1:]

    if len(uniques) == 0:
        if len(uniques_true) == 0:
            return 1.0
        else:
            return 0.0

    used = np.zeros(uniques.shape)
    for nuc_truth in uniques_true:
        binary_truth = crop_img_gt == nuc_truth
        IOUs = []
        cards = []
        Is = []
        Us = []
        for nuc in uniques:
            binary_seg = crop_img_new_seg == nuc
            I = np.sum(np.logical_and(binary_seg, binary_truth))
            U = np.sum(np.logical_or(binary_seg, binary_truth))
            Us.append(U)
            Is.append(I)
            IOUs.append(I / U)
        ind = np.argmax(np.array(IOUs))
        used[ind] = 1

        intersection += np.array(Is)[ind]
        union += np.array(Us)[ind]

    for not_used in np.argwhere(np.logical_not(used)):
        union += np.sum(crop_img_new_seg == uniques[not_used])
    if union == 0.0:
        return 1
    else:
        return intersection / union


def DICE(crop_img_new_seg, crop_img_gt):
    Is = 0
    Us = 0
    intersection = 0
    union = 0

    uniques = np.unique(crop_img_new_seg)[1:]
    uniques_true = np.unique(crop_img_gt)[1:]
    if len(uniques) == 0:
        if len(uniques_true) == 0:
            return 1.0
        else:
            return 0.0
    used = np.zeros(uniques.shape)
    for nuc_truth in uniques_true:
        binary_truth = crop_img_gt == nuc_truth
        IOUs = []
        cards = []
        Is = []
        Us = []
        for nuc in uniques:
            binary_seg = crop_img_new_seg == nuc
            I = 2 * np.sum(np.logical_and(binary_seg, binary_truth))
            U = np.sum(binary_seg) + np.sum(binary_truth)
            Us.append(U)
            Is.append(I)
            IOUs.append(I / U)
        ind = np.argmax(np.array(IOUs))
        used[ind] = 1

        intersection += np.array(Is)[ind]
        union += np.array(Us)[ind]

    for not_used in np.argwhere(np.logical_not(used)):
        union += np.sum(crop_img_new_seg == uniques[not_used])
    if union == 0.0:
        return 1
    else:
        return intersection / union


def calculate_precision_recall(img_new, img_gt):
    unused = np.unique(img_new)[1:]
    matched_pairs = {}
    tp = 0
    fn = 0
    for nuclei_gt in np.unique(img_gt)[1:]:
        match = False

        for nuclei_seg in unused:
            bool_gt = img_gt == nuclei_gt
            bool_new = img_new == nuclei_seg
            bool_iou = (
                np.sum(np.logical_and(bool_gt, bool_new))
                / np.sum(np.logical_or(bool_gt, bool_new))
            ) > 0.5

            if bool_iou:
                match = True
                matched_pairs[nuclei_gt] = nuclei_seg
                tp += 1
                indice = np.where(unused == nuclei_seg)[0][0]
                unused = np.delete(unused, indice)
        if not (match):
            fn += 1

    fp = len(unused)

    bool_fp = tp + fp == 0
    bool_fn = tp + fn == 0

    precision = 1 if bool_fp else tp / (tp + fp)

    recall = tp / (tp + fn) if not (bool_fn) else 0

    F1 = 2 * (recall * precision) / (recall + precision + 1e-3)
    return precision, recall, F1, matched_pairs


def model_predict(
    filename,
    model,
    path_baseline,
    count_erase=15,
    radius=3,
):
    model.eval()
    model.cuda()

    click = tifffile.imread(os.path.join(path_baseline, "click", filename))
    image_init = tifffile.imread(os.path.join(path_images, filename)) / 255

    contour_baseline = tifffile.imread(
        os.path.join(path_baseline, "contour", filename)
    ).astype(float)

    binary_baseline = tifffile.imread(
        os.path.join(path_baseline, "binary", filename)
    ).astype(float)

    inside_baseline = ((binary_baseline - contour_baseline) > 0).astype(float)

    background_baseline = 1 - np.maximum(contour_baseline, inside_baseline)

    baseline = np.stack(
        (background_baseline, contour_baseline, inside_baseline), axis=-1
    )

    image = np.concatenate((image_init, click), axis=-1)
    image_tensor = torch.Tensor(
        np.transpose(np.expand_dims(image, axis=0), (0, -1, 1, 2))
    ).cuda()
    baseline_tensor = torch.Tensor(
        np.transpose(np.expand_dims(baseline, axis=0), (0, -1, 1, 2))
    ).cuda()

    img = model(image_tensor, baseline_tensor)[0]

    img = np.transpose(img.cpu().detach().numpy(), (1, -1, 0))
    exp = np.exp(img)
    arg = np.argmax(exp, -1)

    binary = (arg == 2.0).astype(int)
    labels = label(binary)

    _, count = np.unique(labels, return_counts=True)
    indx = np.argwhere(count < count_erase).flatten()
    for i in indx:
        labels[labels == i] = 0.0

    black = np.zeros(labels.shape)

    for i, nuclei in enumerate(np.unique(labels)[1:]):
        black = np.maximum(
            black, (i + 1) * binary_dilation(labels == nuclei, disk(2)).astype(int)
        )
    return black.astype(int), exp


def model_predict_with_click(
    filename,
    model,
    path_baseline,
    click,
    count_erase=15,
    radius=3,
):
    model.eval()
    model.cuda()

    image_init = tifffile.imread(os.path.join(path_images, filename)) / 255

    contour_baseline = tifffile.imread(
        os.path.join(path_baseline, "contour", filename)
    ).astype(float)
    binary_baseline = tifffile.imread(
        os.path.join(path_baseline, "binary", filename)
    ).astype(float)

    inside_baseline = ((binary_baseline - contour_baseline) > 0).astype(float)
    background_baseline = 1 - np.maximum(contour_baseline, inside_baseline)
    baseline = np.stack(
        (background_baseline, contour_baseline, inside_baseline), axis=-1
    )

    image = np.concatenate((image_init, click), axis=-1)
    image_tensor = torch.Tensor(
        np.transpose(np.expand_dims(image, axis=0), (0, -1, 1, 2))
    ).cuda()
    baseline_tensor = torch.Tensor(
        np.transpose(np.expand_dims(baseline, axis=0), (0, -1, 1, 2))
    ).cuda()

    img = model(image_tensor, baseline_tensor)[0]

    img = np.transpose(img.cpu().detach().numpy(), (1, -1, 0))
    exp = np.exp(img)

    arg = np.argmax(exp, -1)

    binary = (arg == 2.0).astype(int)
    labels = label(binary)

    _, count = np.unique(labels, return_counts=True)
    indx = np.argwhere(count < count_erase).flatten()
    for i in indx:
        labels[labels == i] = 0.0

    black = np.zeros(labels.shape)

    for i, nuclei in enumerate(np.unique(labels)[1:]):
        black = np.maximum(
            black, (i + 1) * binary_dilation(labels == nuclei, disk(2)).astype(int)
        )

    return (
        black.astype(int),
        exp,
        binary_baseline,
    )


def predict_click(filename, model, path_baseline, count_erase=15, radius=3):
    f = plt.figure(figsize=2 * np.array([10, 20]))
    img = tifffile.imread(os.path.join(path_images, filename)) / 255
    fontsize = 15
    click = tifffile.imread(os.path.join(path_baseline, "click", filename))
    click = n_channel_to_rgb(click)
    img_gt = tifffile.imread(os.path.join(path_gt, "baseline", filename))
    img_gt = remove_wrong_annotation(img_gt)

    contour_baseline = tifffile.imread(
        os.path.join(path_baseline, "contour", filename)
    ).astype(float)
    patch_baseline = tifffile.imread(
        os.path.join(path_baseline, "binary", filename)
    ).astype(float)

    patch_baseline = ((patch_baseline - contour_baseline) > 0).astype(float)
    background_baseline = 1 - np.maximum(contour_baseline, patch_baseline)
    baseline = np.stack(
        (background_baseline, contour_baseline, patch_baseline), axis=-1
    )
    img_baseline = tifffile.imread(os.path.join(path_baseline, "baseline", filename))

    f.add_subplot(1, 7, 5)

    img_new, new_softmax = model_predict(
        filename=filename,
        model=model,
        path_baseline=path_baseline,
        count_erase=count_erase,
        radius=radius,
    )

    plt.imshow(img_new.astype(int), cmap="nipy_spectral")
    plt.title("new segmentation", fontdict={"fontsize": fontsize})
    plt.axis("off")

    f.add_subplot(1, 7, 4)
    plt.imshow(click)
    plt.title("Click map", fontdict={"fontsize": fontsize})
    plt.axis("off")

    f.add_subplot(1, 7, 1)
    plt.imshow(img)
    plt.title("patch", fontdict={"fontsize": fontsize})
    plt.axis("off")

    f.add_subplot(1, 7, 2)
    plt.imshow(baseline)
    plt.title("initial segmentation", fontdict={"fontsize": fontsize})
    plt.axis("off")

    f.add_subplot(1, 7, 3)
    plt.imshow(img_gt)
    plt.title("ground truth", fontdict={"fontsize": fontsize})
    plt.axis("off")

    f.add_subplot(1, 7, 6)
    plt.imshow(new_softmax)
    plt.title("new softmax", fontdict={"fontsize": fontsize})
    plt.axis("off")
    plt.show()

    print("new segmentation")
    print(calculate_precision_recall(img_new, img_gt))
    print("stardist")
    print(calculate_precision_recall(img_baseline, img_gt))
    return img_new, img_gt


def model_predict_batch(pred, count_erase=15, radius=3):
    arg = np.argmax(pred, -1)
    binary = (arg == 2).astype(int)
    labels = label(binary)
    black = np.zeros(labels.shape)

    _, count = np.unique(labels, return_counts=True)

    indx = np.argwhere(count < count_erase).flatten()

    for i in indx:
        labels[labels == i] = 0.0

    for i, nuclei in enumerate(np.unique(labels)[1:]):
        black = np.maximum(
            black, (i + 1) * binary_dilation(labels == nuclei, disk(radius)).astype(int)
        )
    return black



def n_channel_to_rgb(click, colormap = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]):
    # Ensure the colormap has n entries
    if len(colormap) != click.shape[2]:
        raise ValueError("Colormap should have the same number of entries as channels.")

    # Create an empty RGB image
    height, width, _ = click.shape
    rgb_image = np.zeros((height, width, 3))

    # Map each channel to its respective color
    for channel_index, color in enumerate(colormap):
        rgb_image += click[:, :, channel_index, np.newaxis] * color

    # Clip the values to be within the 0-255 range
    rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

    return rgb_image
