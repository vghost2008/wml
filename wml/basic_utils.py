import numpy as np

def np_unstack(array,axis=0):
    unstacked = np.split(array, array.shape[axis], axis=axis)
    unstacked = [arr.squeeze(axis=axis) for arr in unstacked]
    return unstacked