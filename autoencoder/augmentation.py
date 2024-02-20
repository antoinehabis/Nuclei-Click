import numpy as np

def augmentation(img):

    ps = np.random.random(10)
 
    if ps[0]>1/4 and ps[0]<1/2:
        img= np.rot90(img,axes = [0,1], k=1)
        
    if ps[0]>1/2 and ps[0]<3/4:
        img = np.rot90(img,axes = [0,1], k=2)
         
    if ps[0]>3/4 and ps[0]<1:
        img= np.rot90(img,axes = [0,1], k=3)
        
    if ps[1]>0.5:
        img = np.flipud(img)
        
    if ps[2]>0.5:
        img = np.fliplr(img)
    return img