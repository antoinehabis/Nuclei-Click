from config import *
from tqdm import tqdm 
import tifffile
import numpy as np
from glob import glob
import cv2
path_baseline = path_stardist

for path in tqdm(glob(os.path.join(path_baseline,'baseline/*'))):
    img = tifffile.imread(path)
    binary = (img > 0).astype(float)
    cts = np.zeros(binary.shape)
    filename = path.split("/")[-1]
    for u in np.unique(img)[1:]:
        nuclei = ((img == u)).astype(np.uint8)
        contours, _ = cv2.findContours(nuclei, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cts = cv2.drawContours(cts, contours, -1, 1, 2)
    cts = (cts > 0).astype(float)
    tifffile.imsave(os.path.join(path_baseline, 'contour', filename), cts)
    tifffile.imsave(os.path.join(path_baseline, 'binary', filename), binary)