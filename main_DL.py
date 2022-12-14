import argparse
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
from util import *
from data_DL import *
from model import InferenceModel
from model import LeNet5
from model import Net
import torch.optim as optim
import torch.nn as nn

# Authors: Guillaume Gagné-Labelle, Yann Saah, Giovanni Belval
# Date: Oct 2022
# Project: Kaggle Competition - IFT3395
# Description: Classification of the sum of 2 MNIST images

args_form = argparse.ArgumentParser(allow_abbrev=False)
args_form.add_argument("--max_epoch", type=int, default=64)  # number of iterations on the train set
args_form.add_argument("--batch_size", type=int, default=128)
args_form.add_argument("--decay", type=float, default=1e-5)
args_form.add_argument("--lr", type=float, default=1e-3)
args_form.add_argument("--train_pct", type=float, default=0.7)
args_form.add_argument("--val_train_pct", type=float, default=0.15)
args_form.add_argument("--path_data", type=str, default="mnist/")
args_form.add_argument("--path_model", type=str, default="mnist/")
args_form.add_argument("--use_scheduler", action='store_true')


def main():
    args = args_form.parse_args()
    set_seed(0)
    args.loss1 = defaultdict(OrderedDict)
    args.loss2 = defaultdict(OrderedDict)
    args.loss3 = defaultdict(OrderedDict)
    args.acc = defaultdict(OrderedDict)  # combined precision
    args.acc1 = defaultdict(OrderedDict) # individual precision
    args.acc2 = defaultdict(OrderedDict) #
    args.acc3 = defaultdict(OrderedDict) #

    train_data, train_labels, val_train_data, val_train_labels, val_test_data, val_test_labels, test_data = Data.get_mnist(args)
    train_set = Data(data=train_data, labels=train_labels, batch_size=args.batch_size)
    val_train_set = Data(data=val_train_data, labels=val_train_labels, batch_size=args.batch_size)
    val_test_set = Data(data=val_test_data, labels=val_test_labels, batch_size=args.batch_size)
    test_set = Data(data=test_data, labels=np.empty((test_data.shape[0], 2)), batch_size=1)

    model1 = Net().to(device)
    model2 = LeNet5(19).to(device)
    model3 = InferenceModel((1, 28, 28)).to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer1 = optim.Adam(model1.parameters(), lr=args.lr, weight_decay=0)
    optimizer2 = optim.Adam(model2.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer3 = optim.Adam(model3.parameters(), lr=args.lr, weight_decay=0)
    if args.use_scheduler:
        scheduler1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.max_epoch)
        scheduler2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.max_epoch)
        scheduler3 = optim.lr_scheduler.CosineAnnealingLR(optimizer3, T_max=args.max_epoch)

    best_loss, best_acc = np.inf, 0
    mean_loss, mean_acc = best_loss, best_acc
    for epoch in range(args.max_epoch):
        for set in ("train", "val_train", "val_test"):
            if set == "train": dataset = train_set
            elif set == "val_train": dataset = val_train_set
            else: dataset = val_test_set

            ## Core loop
            if set == "train":
                assess(model1, dataset, criterion, optimizer1, update_model=True)
                assess(model2, dataset, criterion, optimizer2, update_model=True)
                assess(model3, dataset, criterion, optimizer3, update_model=True)
                if args.use_scheduler:
                    scheduler1.step()
                    scheduler2.step()
                    scheduler3.step()
            if dataset.size > 0:
                loss1, acc1 = assess(model1, dataset, criterion, optimizer1)
                loss2, acc2 = assess(model2, dataset, criterion, optimizer2)
                loss3, acc3 = assess(model3, dataset, criterion, optimizer3)
                mean_loss = (loss1 + loss2 + loss3) / 3
                mean_acc = (acc1 + acc2 + acc3) / 3
            if set == "val_train" and mean_loss < best_loss:
                res = write_results(model1, model2, model3, test_set)
                # res = write_single_results(model3, test_set)  # write results for a singel model instead
                best_loss = mean_loss
                best_acc = mean_acc

            args.loss1[set][epoch] = loss1
            args.loss2[set][epoch] = loss2
            args.loss3[set][epoch] = loss3

            # assess the good accuracy, not only the mean
            accuracy, accuracy1, accuracy2, accuracy3 = assess_acc(model1, model2, model3, dataset)
            args.acc[set][epoch] = accuracy
            args.acc1[set][epoch] = accuracy1
            args.acc2[set][epoch] = accuracy2
            args.acc3[set][epoch] = accuracy3

    # Plot of the average loss among the 3 models over different datasets
    plt.title("loss")
    plt.plot(args.loss["train"].keys(), args.loss["train"].values(), label="train")
    plt.plot(args.loss["val_train"].keys(), args.loss["val_train"].values(), label="val_train")
    plt.plot(args.loss["val_test"].keys(), args.loss["val_test"].values(), label="val_test")
    plt.legend()
    plt.grid()
    plt.savefig("loss_lr%.5f_wd%.5f.png" % (args.lr, args.decay))

    # Plot of the combined acc among the 3 models over different datasets
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
    if args.use_scheduler: title = "results.csv"
    else: title = "results_no_sched.csv"
    text_file = open(title, "w")
    text_file.write(res)
    text_file.close()

def assess(model, dataset, criterion, optimizer, update_model=False):
    if update_model: model.train()
    else: model.eval()
    loss_total, correct, total = 0, torch.tensor(0, device=device), torch.tensor(0, device=device)

    for images, labels_with_indices in iter(dataset):
        if update_model:
            optimizer.zero_grad()
            images = torch.tensor(images, device=device).float()
            labels_with_indices = torch.tensor(labels_with_indices, device=device)
            prob = model(images)
        else:
            with torch.no_grad():
                images = torch.tensor(images, device=device).float()
                labels_with_indices = torch.tensor(labels_with_indices, device=device)
                prob = model(images)

        total += images.shape[0]
        indices = labels_with_indices[:, 0]
        labels = labels_with_indices[:, 1]

        _, pred = prob.max(1)

        loss = criterion(prob, labels)
        loss_total += loss.item()
        correct += (pred == labels).sum()

        if update_model:
            loss.backward()
            optimizer.step()

    model.train()
    return loss_total / dataset.size, 100 * correct.item() / dataset.size


if __name__ == main():
    main()
