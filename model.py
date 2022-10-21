import numpy
import numpy as np
from util import softmax
import matplotlib.pyplot as plt

# Author: Guillaume Gagné-Labelle
# Date: Oct 2022
# Project: Kaggle Competition - IFT3395
# Description: Classification of the sum of 2 MNIST images

class logistic:
    def __init__(self, args):
        self.W = np.random.normal(size=(28*56, 19))  # num_classes = 19, dimension = 28x28
        self.b = np.zeros(19)
        self.lr = args.lr
        self.decay = args.decay


    # predict a label out of a probability distribution computed by a logistic regression from an image
    # input: batch of pairs of images: [batch_size, 2, 28, 28]
    # output: batch of labels: [batch_size]
    def prediction(self, images):
        images = images.reshape(images.shape[0], -1)

        prob = np.matmul(self.W.transpose(), images[:,:,np.newaxis]).squeeze() + self.b  # W_t X + b
        prob = softmax(prob)
        pred = np.argmax(prob, axis=1)

        return pred, prob


    def update(self, images, prob, labels):
        # Compute the gradient out of a probability distribution and a loss
        # input: 2 scalars
        # output:[d, m] == W.shape
        def grad_b(prob, labels):
            N = prob.shape[0]  # batch_size
            m = prob.shape[1]  # num_classes

            grad = np.zeros(m)
            prob = np.take_along_axis(prob, labels[:, np.newaxis], axis=1).squeeze()  # probability associate with the good label for each image. i.e. prob[i] = P(labels[i])

            for i in range(N):
                grad += prob[i] * np.ones(m)
                grad[labels[i]] -= 1

            return grad

        def grad_w(images, prob, labels):
            N = prob.shape[0]  # batch_size
            m = prob.shape[1]  # num_classes
            d = self.W.shape[0]

            grad = np.empty_like(self.W)
            prob = np.take_along_axis(prob, labels[:, np.newaxis], axis=1).squeeze()  # probability associate with the good label for each image. i.e. prob[i] = P(labels[i])

            for p in range(d):
                E = np.zeros(m)
                for n in range(N):
                    E += prob[n] * np.ones(m)
                    E[labels[n]] -= 1

                    E *= images[n][p]  # [m]
                grad[p] = E

            return grad

        self.W = self.W - self.lr * (grad_w(images, prob, labels) + self.decay * self.W)
        self.b = self.b - self.lr * grad_b(prob, labels)
