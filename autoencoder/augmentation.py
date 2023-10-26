
# from config import *
# from numba import jit
# from skimage.filters import gaussian
# import albumentations as A
# from tps import ThinPlateSpline
# import largestinteriorrectangle as lir
import numpy as np
# dim = 512

# @jit
# def mapping_coordinates(coordinates, img):
    
#     n,m = img.shape[:2]
#     new_img = np.zeros(img.shape)
    
#     for i in range(n):
#         for j in range(m): 
            
#             coord = coordinates[i,j]
#             x = coord[0]
#             y = coord[1]
#             if x >= 0 and x <m:
#                 if y >= 0 and y <n:
#                     new_img[i, j] = img[y, x]
#     return new_img
   



# def elastic_transform(img,
#                       mask1,
#                       mask2,
#                       std = 7):
    
#     tps = ThinPlateSpline()

    
#     l,c = img.shape[:2]

#     n = 8
    
#     mesh = np.stack(np.meshgrid(np.arange(c), np.arange(l)),-1)
#     X_c = np.concatenate((np.random.randint(0, c, (n, 1)), np.random.randint(0, l, (n, 1))), axis = -1)

#     mean = 0

#     X_t = X_c + np.round(np.random.randn(n*2)*std).reshape(n,2)
#     tps.fit(X_c, X_t)

#     # # Transform new points
#     Y = tps.transform(mesh.reshape(-1,2))
#     coordinates = np.round(Y.reshape(l,c, 2)).astype(int)
    
#     return (mapping_coordinates(coordinates, img),
#             mapping_coordinates(coordinates, mask1),
#             mapping_coordinates(coordinates, mask2))



# def augmentation(img,
#                  mask1,
#                  mask2):
    
#     if np.max(img)<=1:
#         img = (img*255).astype(np.uint8)
#     mask_shape = mask1.shape
    
#     if len(mask_shape)==2:
#         mask1 = np.expand_dims(mask1, -1)
        
#     ps = np.random.random(10)
#     if ps[0]>1/4 and ps[0]<1/2:
#         img, mask, mask2 = np.rot90(img,axes = [0,1], k=1), np.rot90(mask1,axes = [0,1], k=1), np.rot90(mask2,axes = [0,1], k=1)
        
#     if ps[0]>1/2 and ps[0]<3/4:
#         img, mask, mask2 = np.rot90(img,axes = [0,1], k=2), np.rot90(mask1,axes = [0,1], k=2), np.rot90(mask2,axes = [0,1], k=2)
        
#     if ps[0]>3/4 and ps[0]<1:
#         img, mask, mask2 = np.rot90(img,axes = [0,1], k=3), np.rot90(mask1,axes = [0,1], k=3), np.rot90(mask2,axes = [0,1], k=3)
        
#     if ps[1]>0.5:
#         img, mask, mask2 = np.flipud(img), np.flipud(mask1), np.flipud(mask2)
        
#     if ps[2]>0.5:
#         img, mask, mask2 = np.fliplr(img), np.fliplr(mask1), np.fliplr(mask2)
        
#     if ps[3]>0.5:
        
#         ker = np.random.random(1)*1.0
#         img = gaussian(img, sigma=(ker, ker), truncate=3.5, channel_axis=2)
#     if np.max(img)<=(1+1e-1):
        
#         img = np.clip((img*255),0,255)
#     img = img.astype(np.uint8)
#     img = A.RandomBrightnessContrast(brightness_limit=[-0.05,0.15], contrast_limit=[-0.1,0.1], p=1.0)(image = img)['image']
#     img = A.augmentations.transforms.RandomGamma (gamma_limit=(80, 120), eps=None, always_apply=False, p=1)(image = img)['image']
#     img = img / 255
    
#     mask1 = mask1[:,:]
#     mask2 = mask2[:,:]

#     img, mask1, mask2 = elastic_transform(img,
#                                           mask1,
#                                           mask2,
#                                           std=4)
    
    
    
#     q = np.mean(img,-1)
#     ret = (1 - (q == 0)).astype('bool')
#     rect = lir.lir(ret)
#     crop_img = img[rect[1]:rect[3],rect[0]:rect[2]]
#     crop_mask1 = mask1[rect[1]:rect[3],rect[0]:rect[2]]
#     crop_mask2 = mask2[rect[1]:rect[3],rect[0]:rect[2]]

#     resize_img = cv2.resize(crop_img, (256,256),interpolation = cv2.INTER_AREA)
#     resize_mask1 = cv2.resize(crop_mask1, (256,256),interpolation = cv2.INTER_NEAREST)
#     resize_mask2 = cv2.resize(crop_mask2, (256,256),interpolation = cv2.INTER_NEAREST)
    
#     return np.clip(resize_img, 0, 1), resize_mask1, resize_mask2


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