import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from matplotlib.pyplot import imshow

# Authors: Guillaume Gagné-Labelle, Yann Saah, Giovanni Belval
# Date: Oct 2022
# Project: Kaggle Competition - IFT3395
# Description: Classification of the sum of 2 MNIST images

class Data:
    # load the data, split it and shuffle it
    @staticmethod
    def get_mnist(args):
        mnist_data = pd.DataFrame(pd.read_csv(args.path_data +'train.csv')).to_numpy()[:, :-1].reshape(-1, 1, 28, 56) #.reshape((-1, 28, 2, 28)).transpose(0, 2, 1, 3)  # [50000, 28, 56]
        mnist_labels = pd.DataFrame(pd.read_csv(args.path_data +'train_result.csv')).to_numpy()[:, :]  # [50000, [orig_index, label]]
        test_data = pd.DataFrame(pd.read_csv(args.path_data +'test.csv')).to_numpy()[:, :-1].reshape(-1, 1, 28, 56)

        train_indices = np.arange(mnist_data.shape[0])
        np.random.shuffle(train_indices)
        mnist_data = mnist_data[train_indices]
        mnist_labels = mnist_labels[train_indices]

        train_data, val_train_data, val_test_data = np.split(mnist_data, [int(args.train_pct * mnist_data.shape[0]), int((args.train_pct + args.val_train_pct) * mnist_data.shape[0])])
        train_labels, val_train_labels, val_test_labels = np.split(mnist_labels, [int(args.train_pct * mnist_labels.shape[0]), int((args.train_pct + args.val_train_pct) * mnist_labels.shape[0])])

        return train_data, train_labels, val_train_data, val_train_labels, val_test_data, val_test_labels, test_data

    def __init__(self, data, labels, batch_size):
        self.data = data
        self.labels = labels
        self.size = labels.shape[0]
        self.batch_size = batch_size

    # Iterator over a batch of data
    def __iter__(self):
        self.indices = np.arange(self.size)
        if self.batch_size > 1: np.random.shuffle(self.indices)
        self.i = 0

        return self

    def __next__(self):
        if self.i + self.batch_size < self.size:
            indices = self.indices[self.i : self.i+self.batch_size]
            self.i += self.batch_size
            return self.data[indices], self.labels[indices]
        elif self.i < self.size:
            indices = self.indices[self.i:]
            self.i += self.batch_size
            return self.data[indices], self.labels[indices]
        else:
            raise StopIteration

