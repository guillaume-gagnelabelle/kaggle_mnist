import numpy as np
import pandas as pd
from scipy import sparse

# Author: Guillaume Gagné-Labelle
# Date: Oct 2022
# Project: Kaggle Competition - IFT3395
# Description: Classification of the sum of 2 MNIST images

class Data:

    @staticmethod
    def get_mnist(path):
        mnist_data = pd.DataFrame(pd.read_csv(path+'train.csv')).to_numpy()[:, :-1] #.reshape((-1, 28, 2, 28)).transpose(0, 2, 1, 3)  # [50000, 28, 56]
        mnist_labels = pd.DataFrame(pd.read_csv(path+'train_result.csv')).to_numpy()[:, :]  # [50000, [orig_index, label]]
        test_data = pd.DataFrame(pd.read_csv(path+'test.csv')).to_numpy()[:, :-1]

        # mnist_data = mnist_data.reshape(-1,28,28)

        train_indices = np.arange(mnist_data.shape[0])
        np.random.shuffle(train_indices)
        mnist_data = mnist_data[train_indices]
        mnist_labels = mnist_labels[train_indices]

        train_data, val_train_data, val_test_data = np.split(mnist_data, [int(0.70 * mnist_data.shape[0]), int(0.85 * mnist_data.shape[0])])
        train_labels, val_train_labels, val_test_labels = np.split(mnist_labels, [int(0.70 * mnist_labels.shape[0]), int(0.85 * mnist_labels.shape[0])])

        # # Normalize data?
        # train_data = (train_data - train_data.mean(0)) / (train_data.std(0)+1e-10)

        return train_data, train_labels, val_train_data, val_train_labels, val_test_data, val_test_labels, test_data

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.size = labels.shape[0]

    def __iter__(self):
        self.indices = np.arange(self.size)
        np.random.shuffle(self.indices)
        self.i = 0
        self.batch_size = 16

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

