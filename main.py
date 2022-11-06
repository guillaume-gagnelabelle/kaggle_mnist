import numpy as np
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import argparse
from collections import OrderedDict, defaultdict
from util import *
from data import *
from model import logistic

# Authors: Guillaume Gagné-Labelle, Yann Saah, Giovanni Belval
# Date: Oct 2022
# Project: Kaggle Competition - IFT3395
# Description: Classification of the sum of 2 MNIST images

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument("--max_epoch", type=int, default=64)  # number of iterations on the train set
args_form.add_argument("--batch_size", type=int, default=16)
args_form.add_argument("--decay", type=float, default=1e-5)
args_form.add_argument("--lr", type=float, default=1e-3)
args_form.add_argument("--train_pct", type=float, default=0.7)
args_form.add_argument("--val_train_pct", type=float, default=0.15)
args_form.add_argument("--path_data", type=str, default="mnist/")


def main():
    args = args_form.parse_args()
    set_seed(0)
    args.loss = defaultdict(OrderedDict)
    args.acc = defaultdict(OrderedDict)

    train_data, train_labels, val_train_data, val_train_labels, val_test_data, val_test_labels, test_data = Data.get_mnist(args)
    train_set = Data(data=train_data, labels=train_labels, batch_size=args.batch_size)
    val_train_set = Data(data=val_train_data, labels=val_train_labels, batch_size=args.batch_size)
    val_test_set = Data(data=val_test_data, labels=val_test_labels, batch_size=args.batch_size)
    test_set = Data(data=test_data, labels=np.empty((test_data.shape[0], 2)), batch_size=1)

    model = logistic(args)
    criterion = xent

    best_loss, best_acc = np.inf, 0
    loss, acc = best_loss, best_acc
    for epoch in range(args.max_epoch):
        for set in ("train", "val_train", "val_test"):
            if set == "train": dataset = train_set
            elif set == "val_train": dataset = val_train_set
            else: dataset = val_test_set

            if dataset.size > 0:
                loss, acc = assess(model, dataset, criterion, update_model=(set=="train"))

            if set == "val_train" and loss < best_loss:
                res = write_single_results(model, test_set)
                best_loss = loss
                best_acc = acc

            args.loss[set][epoch] = loss
            args.acc[set][epoch] = acc

    plt.title("loss")
    plt.plot(args.loss["train"].keys(), args.loss["train"].values(), label="train")
    plt.plot(args.loss["val_train"].keys(), args.loss["val_train"].values(), label="val_train")
    plt.plot(args.loss["val_test"].keys(), args.loss["val_test"].values(), label="val_test")
    plt.legend()
    plt.grid()
    plt.savefig("loss_lr%.5f_wd%.5f.png" % (args.lr, args.decay))

    plt.figure()
    plt.title('acc')
    plt.plot(args.acc["train"].keys(), args.acc["train"].values(), label="train")
    plt.plot(args.acc["val_train"].keys(), args.acc["val_train"].values(), label="val_train")
    plt.plot(args.acc["val_test"].keys(), args.acc["val_test"].values(), label="val_test")
    plt.legend()
    plt.grid()
    plt.savefig("acc_lr%.5f_wd%.5f.png" % (args.lr, args.decay))

    print("----------- End of Training -----------")
    print("--- lr: %.5f --- wd: %.5f ---" % (args.lr, args.decay))
    print("Best loss: %.9f" % best_loss)
    print("Best  acc: %.9f" % best_acc)
    print("---------------------------------------")
    print()

    # open text file
    if args.use_scheduler:
        title = "results.csv"
    else:
        title = "results_no_sched.csv"
    text_file = open(title, "w")
    text_file.write(res)
    text_file.close()



def assess(model, dataset, criterion, update_model=False):
    loss, correct, total = 0, 0, 0

    for images, labels_with_indices in iter(dataset):

        total += images.shape[0]
        indices = labels_with_indices[:, 0]
        labels = labels_with_indices[:, 1]

        pred, prob, prob1, prob2 = model.prediction(images)

        loss += criterion(prob, labels)
        correct += (pred == labels).sum()

        if update_model: model.update(images, prob, prob1, prob2, labels)

    return loss / dataset.size, 100 * correct / dataset.size

if __name__ == main():
    main()


