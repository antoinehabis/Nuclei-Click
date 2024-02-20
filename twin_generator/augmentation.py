import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))


from config import *
from numba import jit
from skimage.filters import gaussian
import albumentations as A
import largestinteriorrectangle as lir
import cv2
import numpy as np
import elasticdeform
from config import *
from glob import glob
import tifffile
dim = 256


def get_elastic_deform_and_crop(img):

    ### img must have values between 0 and 1

    img_deformed = elasticdeform.deform_random_grid(img, sigma=5, axis=(0, 1))
    q = np.mean(img_deformed, -1)
    ret = (1 - (q == 0)).astype("bool")
    rect = lir.lir(ret)
    cropped_img_deformed = img_deformed[rect[1] : rect[3], rect[0] : rect[2]]
    cropped_img_deformed = (cropped_img_deformed*255).astype(np.uint8)
    resize_cropped_img_deformed = cv2.resize(cropped_img_deformed, (256,256),interpolation = cv2.INTER_AREA)
    return resize_cropped_img_deformed.astype(np.uint8)



def augmentation(img,displacement= np.random.randn(3, 3, 2) * 7):

    ps = np.random.random(10)

    if ps[0]>1/4 and ps[0]<1/2:
        img = np.rot90(img,axes = [0,1], k=1)
        
    if ps[0]>1/2 and ps[0]<3/4:
        img = np.rot90(img,axes = [0,1], k=2)
        
    if ps[0]>3/4 and ps[0]<1:
        img = np.rot90(img,axes = [0,1], k=3)
        
    if ps[1]>0.5:
        img = np.flipud(img)
        
    if ps[2]>0.5:
        img = np.fliplr(img)
        
    if ps[3]>0.5:

        ker = np.random.random(1)*1.0
        img = gaussian(img, sigma=(ker, ker), truncate=3.5, channel_axis=2)
    img = np.clip(img,0,1)
    img  = get_elastic_deform_and_crop(img)
    ### image in type np.uint8

    img = A.RandomBrightnessContrast(brightness_limit=[-0.05,0.15], contrast_limit=[-0.1,0.1], p=1.0)(image = img)['image']
    img = A.augmentations.transforms.RandomGamma (gamma_limit=(80, 120), eps=None, always_apply=False, p=1)(image = img)['image']
    img = img / 255
    return img


