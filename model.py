import numpy
import numpy as np
from util import softmax
import torch
from torch import nn
import torch.nn.functional as F
import time


# Author: Guillaume Gagné-Labelle
# Date: Oct 2022
# Project: Kaggle Competition - IFT3395
# Description: Classification of the sum of 2 MNIST images

class logistic:
    def __init__(self, args):
        self.W = np.random.normal(size=(28 * 28, 10))  # num_classes = 10, dimension = 28x28
        self.b = np.zeros(10)
        self.lr = args.lr
        self.decay = args.decay

    # predict a label out of a probability distribution computed by a logistic regression from an image
    # input: batch of pairs of images: [batch_size, 2, 28, 28]
    # output: batch of probabilities for both images: [batch_size, 10], [batch_size, 10]
    #         batch of labels: [batch_size]
    def prediction(self, images):
        B = images.shape[0]  # batch size
        image1 = images[:, 0, :, :].reshape(B, -1)
        image2 = images[:, 1, :, :].reshape(B, -1)

        prob1 = softmax(np.matmul(self.W.transpose(), image1[:, :, np.newaxis]).squeeze() + self.b)  # W_t X1 + b
        prob2 = softmax(np.matmul(self.W.transpose(), image2[:, :, np.newaxis]).squeeze() + self.b)  # W_t X2 + b

        prob = np.zeros((B, 19))
        for b in range(B):
            for k in range(19):
                for i in range(10):
                    if 0 <= k - i < 10: prob[b, k] += prob1[b, i] * prob2[b, k - i]

        pred = np.argmax(prob, axis=1)

        return pred, prob, prob1, prob2


    def update(self, images, prob, prob1, prob2, labels):
        B = images.shape[0]
        image1 = images[:, 0, :, :].reshape(B, -1)
        image2 = images[:, 1, :, :].reshape(B, -1)

        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        for N in range(B):
            for i in range(10):
                if i != labels[N]: continue
                A = - 1 / prob[N, i]
                for k in range(i):
                    B1 = prob2[N, i - k]
                    B2 = prob1[N, i - k]
                    for m in range(10):
                        if k == m:
                            C1 = prob1[N, k] * (1 - prob1[N, k])
                            C2 = prob2[N, k] * (1 - prob2[N, k])
                        else:
                            C1 = - prob1[N, k] * prob2[N, m]
                            C2 = - prob2[N, k] * prob2[N, m]

                        grad_b[m] += A * (B1*C1 + B2*C2)
                        for n in range(28 * 28):
                            D1 = image1[N, n]
                            D2 = image2[N, n]
                            grad_W[n][m] += A * (B1*C1*D1 + B2*C2*D2)

        self.W -= (self.lr * grad_W / B + 2e-4 * self.W)  # lambda = 1e-4 (Ridge)
        self.b -= self.lr * grad_b / B


class InferenceModel(nn.Module):
    def __init__(self, in_dim):
        super().__init__()

        self.pred = get_enc(in_dim)
        self.cls = nn.Sequential(nn.Linear(128, 19))  # not softmaxed

    def forward(self, d_x):
        pred = self.pred(d_x)
        pred = self.cls(pred)
        return pred


def get_enc(in_dim):
    enc_sz = 256
    return nn.Sequential(
        nn.Conv2d(in_dim[0], 32, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
        nn.BatchNorm2d(64),
        nn.ReLU(),

        nn.MaxPool2d(2, 2),  # 14, 28
        nn.Flatten(start_dim=1),
        nn.Linear(64 * 14 * 28, enc_sz),
        nn.BatchNorm1d(enc_sz),

        nn.ReLU(),
        nn.Linear(enc_sz, 128),
        nn.BatchNorm1d(128),
    )


# Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 14,28
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))  # 7, 14
        self.fc = nn.Linear(704, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # define a conv layer with output channels as 16, kernel size of 3 and stride of 1
        self.conv11 = nn.Conv2d(1, 16, (3, 5), 1)  # Input = 1x28x56  Output = 16x26x52
        self.conv12 = nn.Conv2d(1, 16, (5, 9), 1)  # Input = 1x28x56  Output = 16x24x48
        self.conv13 = nn.Conv2d(1, 16, (7, 13), 1)  # Input = 1x28x56  Output = 16x22x44
        self.conv14 = nn.Conv2d(1, 16, (9, 17), 1)  # Input = 1x28x56  Output = 16x20x40

        # define a conv layer with output channels as 32, kernel size of 3 and stride of 1
        self.conv21 = nn.Conv2d(16, 32, (3, 5), 1)  # Input = 16x26x52 Output = 32x24x48
        self.conv22 = nn.Conv2d(16, 32, (5, 9), 1)  # Input = 16x24x48 Output = 32x20x40
        self.conv23 = nn.Conv2d(16, 32, (7, 13), 1)  # Input = 16x22x44 Output = 32x16x32
        self.conv24 = nn.Conv2d(16, 32, (9, 17), 1)  # Input = 16x20x40  Output = 32x12x24

        # define a conv layer with output channels as 64, kernel size of 3 and stride of 1
        self.conv31 = nn.Conv2d(32, 64, (3, 5), 1)  # Input = 32x24x48 Output = 64x22x44
        self.conv32 = nn.Conv2d(32, 64, (5, 9), 1)  # Input = 32x20x40 Output = 64x16x32
        self.conv33 = nn.Conv2d(32, 64, (7, 13), 1)  # Input = 32x16x32 Output = 64x10x20
        self.conv34 = nn.Conv2d(32, 64, (9, 17), 1)  # Input = 32x12x24 Output = 64x4x8

        # define a max pooling layer with kernel size 2
        self.maxpool = nn.MaxPool2d(2)  # Output = 64x11x22
        # self.maxpool1 = nn.MaxPool2d(1)
        # define dropout layer with a probability of 0.25
        self.dropout1 = nn.Dropout(0.25)
        # define dropout layer with a probability of 0.5
        self.dropout2 = nn.Dropout(0.5)

        # define a linear(dense) layer with 128 output features
        self.fc11 = nn.Linear(64 * 11 * 22, 256)
        self.fc12 = nn.Linear(64 * 8 * 16, 256)  # after maxpooling 2x2
        self.fc13 = nn.Linear(64 * 5 * 10, 256)
        self.fc14 = nn.Linear(64 * 2 * 4, 256)

        # define a linear(dense) layer with output features corresponding to the number of classes in the dataset
        self.fc21 = nn.Linear(256, 128)
        self.fc22 = nn.Linear(256, 128)
        self.fc23 = nn.Linear(256, 128)
        self.fc24 = nn.Linear(256, 128)

        self.fc33 = nn.Linear(128 * 4, 19)

    def forward(self, inp):
        # Use the layers defined above in a sequential way (follow the same as the layer definitions above) and
        # write the forward pass, after each of conv1, conv2, conv3 and fc1 use a relu activation.

        x = F.relu(self.conv11(inp))
        x = F.relu(self.conv21(x))
        x = F.relu(self.maxpool(self.conv31(x)))
        # print(x.shape)
        # x = torch.flatten(x, 1)
        x = x.view(-1, 64 * 11 * 22)
        x = self.dropout1(x)
        x = F.relu(self.fc11(x))
        x = self.dropout2(x)
        x = self.fc21(x)

        y = F.relu(self.conv12(inp))
        y = F.relu(self.conv22(y))
        y = F.relu(self.maxpool(self.conv32(y)))
        # x = torch.flatten(x, 1)
        y = y.view(-1, 64 * 8 * 16)
        y = self.dropout1(y)
        y = F.relu(self.fc12(y))
        y = self.dropout2(y)
        y = self.fc22(y)

        z = F.relu(self.conv13(inp))
        z = F.relu(self.conv23(z))
        z = F.relu(self.maxpool(self.conv33(z)))
        # x = torch.flatten(x, 1)
        z = z.view(-1, 64 * 5 * 10)
        z = self.dropout1(z)
        z = F.relu(self.fc13(z))
        z = self.dropout2(z)
        z = self.fc23(z)

        ze = F.relu(self.conv14(inp))
        ze = F.relu(self.conv24(ze))
        ze = F.relu(self.maxpool(self.conv34(ze)))
        # x = torch.flatten(x, 1)
        ze = ze.view(-1, 64 * 2 * 4)
        ze = self.dropout1(ze)
        ze = F.relu(self.fc14(ze))
        ze = self.dropout2(ze)
        ze = self.fc24(ze)

        out_f = torch.cat((x, y, z, ze), dim=1)
        # out_f1 = torch.cat((out_f, ze), dim=1)
        out = self.fc33(out_f)

        output = F.log_softmax(out, dim=1)
        return out
