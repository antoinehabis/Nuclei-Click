
from config import *
from sklearn.mixture import GaussianMixture as GMM
from skimage.morphology import binary_opening, disk, convex_hull_image
from skimage.measure import label
import numpy as np
import cv2


class ModifyImg:        

    def merge(self, input_):

        connected_comp = label(input_>0)
        unique_baseline = np.unique(input_)
        unique = np.unique(connected_comp)

        max_ = np.max(unique_baseline)
        new = input_.copy()
        tmp = True

        if len(unique[1:]) >0 :
            """ if there is a least one nuclei"""

            for comp in unique[1:] : 
                mask = np.where(connected_comp == comp)

                merge = np.unique(input_[mask])
                if len(merge)>1 and tmp:

                    new[mask] = 0
                    new += convex_hull_image(np.logical_or(input_ == merge[0], input_ == merge[1])) * (max_+1)
                    tmp = False
        return new

    def split(self, input_):

        new = input_.copy()
        uniques = np.unique(input_)
        
        if len(uniques)>3:
            try: 

                r = np.random.choice(uniques[1:])
                comp = r
                zeros = np.zeros((parameters['dim'], parameters['dim']))
                indexes = np.where(new==comp)
                new[indexes] = 0
                n = len(indexes[0])
                X = np.array(indexes).T/256
                gmm = GMM(n_components=2).fit(X)
                labels = gmm.predict(X)+np.max(uniques)+1

                for i in range(n):
                    new[indexes[0][i],indexes[1][i]] = labels[i]

                tmp = new.copy()
                for split in np.unique(labels):
                    new[np.where(new==split)] = 0
                    new += binary_opening(tmp ==split,disk(4))*split
            except: 
                pass

        return new

    def draw_new_contours(self, input_):
        black = np.zeros(input_.shape)
        for u in np.unique(input_)[1:]:
            nuclei = ((input_==u)).astype(np.uint8)
            contours,_  = cv2.findContours(nuclei, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            black = cv2.drawContours(black,contours, -1, 3,2)
        black = black >0
        return black
    
    def modify(self, input_):

        input_1 = self.merge(input_)
        input_2 = self.split(input_1)
        contour = self.draw_new_contours(input_2)

        return input_2, contour

