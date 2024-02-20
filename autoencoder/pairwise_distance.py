
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from numba import jit
from config import *
import time

u = np.load(os.path.join(path_encodings, "encodings256.npy"))

def find_cosine_similarities(u):

    cos = np.tensordot(u[...,0],np.moveaxis(u.T,(0,1,2),(-1,0,1)), axes = ([1],[0])).T
    # x1 = np.linalg.norm(u,axis=1).T[...,None]
    # x2 = np.linalg.norm(u.T[0],axis=0)[None]
    norms = np.dot(np.linalg.norm(u,axis=1).T[...,None],np.linalg.norm(u.T[0],axis=0)[None])
    pairwise_cosine = np.amax(cos/norms, axis = 0)
    return pairwise_cosine

v = find_cosine_similarities(u)
np.save('pairwise_distance'+ str(parameters['n_embedding'])+'.npy',v)