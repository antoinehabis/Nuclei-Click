import multiprocessing
from augmentation import augmentation
from config import *
import tifffile
import numpy as np
nb_twins = 20

def generate_twins(filename):
    
    img = tifffile.imread(os.path.join(path_images, filename))
    for i in range(nb_twins+1):
        if i==0:
            tifffile.imsave(os.path.join(path_twins,filename),img)
        else:
            twin_filename = filename.split('.')[0]+'_twin'+str(i)+'.tif'
            img_augmented = (augmentation(img/255)*255).astype(np.uint8)
            tifffile.imsave(os.path.join(path_twins,twin_filename), img_augmented)

if __name__ == '__main__':dd 
    
    multiprocessing.set_start_method('forkserver', force=True)
    pool = multiprocessing.Pool(processes=16)                         
    pool.map(generate_twins,list(df_val['filename']))
    pool.close()