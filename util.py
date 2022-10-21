import numpy as np
from torch.nn import CrossEntropyLoss
import torch


# Author: Guillaume Gagné-Labelle
# Date: Oct 2022
# Project: Kaggle Competition - IFT3395
# Description: Classification of the sum of 2 MNIST images

def set_seed(n):
    np.random.seed(n)


# input: [N, m] ; output: [N, softmax(m)]
def softmax(A):
    Z = np.exp(A).sum(1)[:, np.newaxis]  # shape: N
    return np.exp(A) / Z


# cross-entropy loss function given a label in (0, 18)
# distr.shape = [N, m]
# labels.shape = [N]
def xent(distr, labels, reduction='mean', epsilon=1e-12):

    # clipping elements where distr < eps or distr > 1-eps
    distr = np.clip(distr, epsilon, 1-epsilon)
    labels = labels[:, np.newaxis]

    distr = np.take_along_axis(distr, labels, axis=1)

    return - np.sum(np.log(distr)) / labels.shape[0] if reduction == 'mean' else - np.sum(np.log(distr))
