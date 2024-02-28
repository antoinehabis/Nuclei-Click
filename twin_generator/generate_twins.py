from augmentation import augmentation
from config import *
import tifffile
import numpy as np
import sys
from tqdm import tqdm

nb_twins = 20
sys.path.append(os.getcwd())


def generate_twins(filename):
    
    img = tifffile.imread(os.path.join(path_images, filename))
    for i in range(nb_twins+1):
        if i==0:
            tifffile.imsave(os.path.join(path_twins,filename),img)
        else:
            twin_filename = filename.split('.')[0]+'_twin'+str(i-1)+'.tif'
            img_augmented = (augmentation(img/255)*255).astype(np.uint8)
            tifffile.imsave(os.path.join(path_twins,twin_filename), img_augmented)

for filename in tqdm(list(df_val['filename'])):
    generate_twins(filename)