import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict, defaultdict
from util import *
from data import *
from model import logistic

# Author: Guillaume Gagné-Labelle
# Date: Oct 2022
# Project: Kaggle Competition - IFT3395
# Description: Classification of the sum of 2 MNIST images

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument("--max_epoch", type=int, default=16)  # number of iterations on the train set
args_form.add_argument("--batch_size", type=int, default=16)
args_form.add_argument("--decay", type=float, default=1e-4)
args_form.add_argument("--lr", type=float, default=1e-3)
args_form.add_argument("--loss_function", type=str, default="xent")
args_form.add_argument("--path_data", type=str, default="mnist/")
args_form.add_argument("--path_model", type=str, default="mnist/")
args_form.add_argument("--save_models", action='store_true')


def main():
    args = args_form.parse_args()
    set_seed(0)
    args.performance = defaultdict(OrderedDict)

    train_data, train_labels, val_train_data, val_train_labels, val_test_data, val_test_labels, test_data = Data.get_mnist(args.path_data)
    train_set = Data(data=train_data, labels=train_labels)
    val_train_set = Data(data=val_train_data, labels=val_train_labels)
    val_test_set = Data(data=val_test_data, labels=val_test_labels)
    test_set = Data(data=test_data, labels=np.zeros_like((test_data.shape[0], 2)))

    model = logistic(args)

    # Did not implement other loss function yet, but can do so by adding elif:
    if args.loss_function == "xent": criterion = xent
    else: criterion = xent


    for epoch in range(args.max_epoch):
        for set in ("train", "val_train", "val_test"):
            if set == "train": dataset = train_set
            elif set == "val_train": dataset = val_train_set
            else: dataset = val_test_set

            ## Core loop
            loss, acc = assess(model, dataset, criterion, update_model=(set=="train"))

            print(set)
            print("loss : ", loss)
            print("acc  : ", acc)
            print("avg W: ", model.W.mean())
            print("avg b: ", model.b.mean())
            print()
            args.performance[set][epoch] = loss
            args.performance[set][epoch] = acc

    # Testing here, must return as csv file



def assess(model, dataset, criterion, update_model=False, testing=False):
    loss, correct, total = 0, 0, 0

    i = 0
    for images, labels_with_indices in iter(dataset):
        total += images.shape[0]
        indices = labels_with_indices[:, 0]
        labels = labels_with_indices[:, 1]

        pred, prob = model.prediction(images)

        loss += criterion(prob, labels)
        correct += (pred == labels).sum()

        if(i % 1000 == 0):
            print(loss / total)
            print(correct * 100 / total)
            print()
        i+=1

        if update_model: model.update(images, prob, labels)

    return loss / dataset.size, 100 * correct / dataset.size

if __name__ == main():
    main()


