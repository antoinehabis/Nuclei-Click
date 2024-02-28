
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from numba import jit
from config import *
import time


u = np.load(os.path.join(path_encodings, "encodings"+str(parameters['n_embedding'])+"_"+str(parameters['contractive'])+".npy"))

def find_cosine_similarities(u):

    """
    Calculates the cosine similarities between all vectors in the matrix u.

    Parameters
    ----------
    u : np.ndarray
        The matrix of encodings, shape (n_encodings, n_embedding, 4)

    Returns
    -------
    np.ndarray
        The pairwise cosine similarities, shape (n_encodings, n_encodings)
    """
    u_moved = np.moveaxis(u,(0,1,2),(1,0,2))
    cos = np.tensordot(u[...,0],u_moved, axes = ([1],[0])).T
    v4 = np.linalg.norm(u,axis = 1)[...,None]
    v1 = np.linalg.norm(u[...,0],axis = 1)[None]
    norms = np.tensordot(v4,v1, axes = ([-1],[0]))
    norms = np.moveaxis(norms,(0,1,-1),(1,0,-1))
    pairwise_cosine = np.amax(cos/norms, axis = 0)
    return pairwise_cosine

v = find_cosine_similarities(u)
np.save("pairwise_distances"+str(parameters['n_embedding'])+"_"+str(parameters['contractive'])+".npy",v)