from config import *
from skimage.segmentation import relabel_sequential
from skimage.measure import label
import numpy as np
import tifffile
from tqdm import tqdm

path_gt = os.path.join(path_pannuke, "gt")


img_fold1 = np.load(path_image_fold1)
img_fold2 = np.load(path_image_fold2)
img_fold3 = np.load(path_image_fold3)

dic = {}
dic["1"] = img_fold1
dic["2"] = img_fold2
dic["3"] = img_fold3

### Saving images
print("Saving images...")
for fold in list(dic.keys()):
    for i in range(dic[fold].shape[0]):
        img = dic[fold][i]
        tifffile.imsave(
            os.path.join(path_images, "nuclei_" + fold + "_" + str(i) + ".tif"), img
    )

mask_fold1 = np.load(path_mask_fold1)
mask_fold1 = np.amax(mask_fold1[:, :, :, :-1], -1)

mask_fold2 = np.load(path_mask_fold2)
mask_fold2 = np.amax(mask_fold2[:, :, :, :-1], -1)

mask_fold3 = np.load(path_mask_fold3)
mask_fold3 = np.amax(mask_fold3[:, :, :, :-1], -1)

dic = {}
dic["1"] = mask_fold1
dic["2"] = mask_fold2
dic["3"] = mask_fold3


def correct_annotation(img, j):
    img = relabel_sequential(img)[0]
    max_ = np.max(img)
    for i in range(1, max_):
        out = label(img == i)
        cc = np.unique(out)[1:]

        if len(cc) > 1:
            """
            Annotation is wrong!
            Some unconnected components have same label.
            We need to correct this.
            """

            for l, u in enumerate(cc):
                img[out == u] = max_ + l
    return img


### Saving masks
print("Saving masks...")

#### Choose Pannuke folder
for fold in list(dic.keys()):
    for i in tqdm(range((dic[fold].shape[0]))):
        mask = dic[fold][i].astype(int)
        mask = correct_annotation(mask, i)

        tifffile.imsave(
            os.path.join(path_gt, "baseline", "nuclei_" + fold + "_" + str(i) + ".tif"),
            mask,
        )
