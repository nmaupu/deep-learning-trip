import h5py
import numpy as np


def load_dataset(filename, xkey, ykey):
    dataset = h5py.File(filename, "r")
    X = np.array(dataset[xkey][:])
    y = np.array(dataset[ykey][:])
    return X, y
