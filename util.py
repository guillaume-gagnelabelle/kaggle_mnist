import numpy as np
from torch.nn import CrossEntropyLoss
import torch
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"

# Authors: Guillaume Gagné-Labelle, Yann Saah, Giovanni Belval
# Date: Oct 2022
# Project: Kaggle Competition - IFT3395
# Description: Classification of the sum of 2 MNIST images

def set_seed(n):
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)

def write_results(model1, model2, model3, dataset):
    out = "Index,Class\n"
    model1.eval()
    model2.eval()
    model3.eval()
    index = 0
    for images, _ in iter(dataset):
        with torch.no_grad():
            images = torch.tensor(images, device=device).float()
            _, pred1 = model1(images).max(1)
            _, pred2 = model2(images).max(1)
            _, pred3 = model3(images).max(1)

            if pred1[0].item() == pred2[0].item(): pred = pred1
            elif pred1[0].item() == pred3[0].item(): pred = pred1
            elif pred2[0].item() == pred3[0].item(): pred = pred2
            else: pred = pred1

            out = out + str(index) + "," + str(pred[0].item()) + "\n"
            index += 1

    model1.train()
    model2.train()
    model3.train()
    return out

def write_single_results(model, dataset):
    out = "Index,Class\n"
    # model.eval()
    index = 0
    for images, _ in iter(dataset):
        pred, prob, prob1, prob2 = model.prediction(images)

        out = out + str(index) + "," + str(pred[0]) + "\n"
        index += 1

    # model.train()
    return out


# input: [N, m] ; output: [N, softmax(m)]
def softmax(A):
    if A.ndim == 1: A = A[np.newaxis, :]
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

def assess_acc(model1, model2, model3, dataset):
    model1.eval()
    model2.eval()
    model3.eval()

    loss_total, correct, total = 0, torch.tensor(0, device=device), torch.tensor(0, device=device)
    correct1, correct2, correct3 = torch.tensor(0, device=device), torch.tensor(0, device=device), torch.tensor(0, device=device)
    for images, labels_with_indices in iter(dataset):
        with torch.no_grad():
            images = torch.tensor(images, device=device).float()
            _, pred1 = model1(images).max(1)
            _, pred2 = model2(images).max(1)
            _, pred3 = model3(images).max(1)

            if pred1[0].item() == pred2[0].item(): pred = pred1
            elif pred1[0].item() == pred3[0].item(): pred = pred1
            elif pred2[0].item() == pred3[0].item(): pred = pred2
            else: pred = pred1
        total += images.shape[0]
        indices = labels_with_indices[:, 0]
        labels = labels_with_indices[:, 1]

        correct += (pred.cpu().numpy() == labels).sum()
        correct1 += (pred1.cpu().numpy() == labels).sum()
        correct2 += (pred2.cpu().numpy() == labels).sum()
        correct3 += (pred3.cpu().numpy() == labels).sum()

    acc = 100 * correct.item() / dataset.size
    acc1 = 100 * correct1.item() / dataset.size
    acc2 = 100 * correct2.item() / dataset.size
    acc3 = 100 * correct3.item() / dataset.size
    model1.train()
    model2.train()
    model3.train()
    return acc, acc1, acc2, acc3

