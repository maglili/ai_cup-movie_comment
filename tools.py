import os
import re
import csv
import torch
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

# import EarlyStopping
from pytorchtools import EarlyStopping


def get_device():
    """
    Check if GPU avaliable.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("There are %d GPU(s) available." % torch.cuda.device_count())
        print("We will use the GPU:", torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using the CPU instead.")
        device = torch.device("cpu")
    print("=" * 10)
    print()
    return device


def setting_path(model_name, batch_size, learning_rate, epochs, mode="train"):
    # setting path
    cwd = os.getcwd()
    if "/" in model_name:
        model_name = re.sub("/", "_", model_name)

    if mode != "retrin":
        folder_name = (
            model_name
            + "_bs"
            + str(batch_size)
            + "_lr"
            + str(learning_rate)
            + "_epo"
            + str(epochs)
        )
    else:
        folder_name = (
            model_name
            + "_bs"
            + str(batch_size)
            + "_lr"
            + str(learning_rate)
            + "_epo"
            + str(epochs)
            + mode,
        )

    if (mode == "train") or (mode == "retrain"):
        model_path = os.path.abspath(
            os.path.join(cwd, "result", folder_name, "train", "model")
        )
        metric_path = os.path.abspath(
            os.path.join(cwd, "result", folder_name, mode, "metrics")
        )
        history_path = os.path.abspath(
            os.path.join(cwd, "result", folder_name, mode, "history")
        )
        fig_path = os.path.abspath(
            os.path.join(cwd, "result", folder_name, mode, "figures")
        )
        # Crate path if folder not exists.
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        if not os.path.isdir(metric_path):
            os.makedirs(metric_path)
        if not os.path.isdir(history_path):
            os.makedirs(history_path)

    elif mode == "test":
        model_path = os.path.abspath(
            os.path.join(cwd, "result", folder_name, "train", "model")
        )
        history_path = os.path.abspath(
            os.path.join(cwd, "result", folder_name, mode, "history")
        )
        metric_path = os.path.abspath(
            os.path.join(cwd, "result", folder_name, mode, "metrics")
        )
        fig_path = None
        # Crate path if folder not exists.
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        if not os.path.isdir(metric_path):
            os.makedirs(metric_path)
        if not os.path.isdir(history_path):
            os.makedirs(history_path)

    else:
        data_path = os.path.abspath(os.path.join(cwd, "result", folder_name, mode))
        model_path = os.path.abspath(
            os.path.join(cwd, "result", folder_name, "train", "model")
        )
        metric_path = os.path.abspath(
            os.path.join(cwd, "result", folder_name, mode, "metrics")
        )
        history_path = None
        fig_path = None
        if not os.path.isdir(data_path):
            os.makedirs(data_path)

    print("metric_path:", metric_path)
    print("model_path:", model_path)
    print("history_path:", history_path)
    print("fig_path:", fig_path)

    return metric_path, model_path, history_path, fig_path


def batch_iter(model, dataloader, optimizer, scheduler, device, training=True):

    correct = 0
    training_loss = 0
    TP = FP = TN = FN = 0
    y_list = []
    yhat_list = []

    if training:
        model.train()
        for batch in dataloader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()
            output = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = output[0]
            logits = output[1]
            loss.backward()
            _, yhat = torch.max(logits.data, 1)
            correct += (yhat == b_labels).sum().item()
            training_loss += loss.item()

            yhat_prob = F.softmax(logits, dim=1)[:, 1]

            # calc metric
            TP += ((yhat == 1) & (b_labels == 1)).sum().item()
            FP += ((yhat == 1) & (b_labels == 0)).sum().item()
            TN += ((yhat == 0) & (b_labels == 0)).sum().item()
            FN += ((yhat == 0) & (b_labels == 1)).sum().item()

            # calc auc

            y_list.extend(b_labels.cpu().detach().numpy())
            yhat_list.extend(yhat_prob.cpu().detach().numpy())

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

    else:
        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                output = model(
                    b_input_ids, attention_mask=b_input_mask, labels=b_labels
                )
                loss = output[0]
                logits = output[1]

                yhat_prob = F.softmax(logits, dim=1)[:, 1]

                # Accumulate the validation loss.
                training_loss += loss.item()
                _, yhat = torch.max(logits.data, 1)
                correct += (yhat == b_labels).sum().item()

                # calc metric
                TP += ((yhat == 1) & (b_labels == 1)).sum().item()
                FP += ((yhat == 1) & (b_labels == 0)).sum().item()
                TN += ((yhat == 0) & (b_labels == 0)).sum().item()
                FN += ((yhat == 0) & (b_labels == 1)).sum().item()

                # calc auc
                y_list.extend(b_labels.cpu().detach().numpy())
                yhat_list.extend(yhat_prob.cpu().detach().numpy())

    return model, correct, training_loss, (TP, FP, TN, FN), y_list, yhat_list


def calc_metrics(
    N_data,
    dataloader,
    correct,
    training_loss,
    confusion_matrix,
    y_list,
    yhat_list,
    useful_stuff,
    mtype,
):
    (TP, FP, TN, FN) = confusion_matrix
    fpr, tpr, _ = metrics.roc_curve(y_list, yhat_list)
    auc = metrics.auc(fpr, tpr)
    acc = correct / N_data
    loss = training_loss / len(dataloader)
    useful_stuff = save_result(
        useful_stuff, acc, loss, (TP, FP, TN, FN), auc, fpr, tpr, mtype
    )

    return useful_stuff


def save_result(useful_stuff, acc, loss, confusion_matrix, auc, fpr, tpr, type):
    (TP, FP, TN, FN) = confusion_matrix
    useful_stuff[type + "_loss"].append(loss)
    useful_stuff[type + "_acc"].append(acc)
    useful_stuff[type + "_metric"].append((TP, FP, TN, FN))
    useful_stuff[type + "_auc"].append(auc)
    useful_stuff[type + "_fpr"].append(fpr)
    useful_stuff[type + "_tpr"].append(tpr)

    print(type + " acc: {0:.2f}".format(acc))
    print(type + " loss: {0:.2f}".format(loss))
    print(type + " auc: {0:.2f}".format(auc))
    print("-" * 25)

    return useful_stuff


def train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    N_train,
    N_test,
    device,
    scheduler,
    path,
    epochs=4,
    patience=3,
):
    """
    train the BERT model.
    Data definition:
        Training: metric that model in training model on train set.
        train: metric that model in eval model on train set.
        valid: metric that model in eval model on valid set.
    """

    useful_stuff = {
        "training_loss": [],
        "training_acc": [],
        "training_auc": [],
        "training_metric": [],
        "training_fpr": [],
        "training_tpr": [],
        "train_loss": [],
        "train_acc": [],
        "train_auc": [],
        "train_metric": [],
        "train_fpr": [],
        "train_tpr": [],
        "valid_loss": [],
        "valid_acc": [],
        "valid_auc": [],
        "valid_metric": [],
        "valid_fpr": [],
        "valid_tpr": [],
    }

    # initialize the early_stopping object
    model_path = os.path.join(path, "model.pkl")
    early_stopping = EarlyStopping(patience=patience, path=model_path, verbose=True)

    for epoch in range(epochs):
        # training
        (
            model,
            correct,
            training_loss,
            (TP, FP, TN, FN),
            y_list,
            yhat_list,
        ) = batch_iter(model, train_loader, optimizer, scheduler, device, training=True)

        useful_stuff = calc_metrics(
            N_train,
            train_loader,
            correct,
            training_loss,
            (TP, FP, TN, FN),
            y_list,
            yhat_list,
            useful_stuff,
            mtype="training",
        )

        # evaluate training
        (
            model,
            correct,
            training_loss,
            (TP, FP, TN, FN),
            y_list,
            yhat_list,
        ) = batch_iter(
            model, train_loader, optimizer, scheduler, device, training=False
        )

        useful_stuff = calc_metrics(
            N_train,
            train_loader,
            correct,
            training_loss,
            (TP, FP, TN, FN),
            y_list,
            yhat_list,
            useful_stuff,
            mtype="train",
        )

        # Validation
        (
            model,
            correct,
            training_loss,
            (TP, FP, TN, FN),
            y_list,
            yhat_list,
        ) = batch_iter(
            model, valid_loader, optimizer, scheduler, device, training=False
        )

        useful_stuff = calc_metrics(
            N_test,
            valid_loader,
            correct,
            training_loss,
            (TP, FP, TN, FN),
            y_list,
            yhat_list,
            useful_stuff,
            mtype="valid",
        )

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(training_loss / len(valid_loader), model, epoch)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        print("=" * 25)

    return useful_stuff, early_stopping.best_epoch


def retrain_model(
    model,
    train_loader,
    optimizer,
    N_train,
    device,
    scheduler,
    path,
    epochs=4,
):
    """
    Retrain the BERT model.
    You should use train_model first to get best hyperparameters.

    Data definition:
        Training: metric that model in training model on train set.
        train: metric that model in eval model on train set.
    """

    useful_stuff = {
        "training_loss": [],
        "training_acc": [],
        "training_auc": [],
        "training_metric": [],
        "training_fpr": [],
        "training_tpr": [],
        "train_loss": [],
        "train_acc": [],
        "train_auc": [],
        "train_metric": [],
        "train_fpr": [],
        "train_tpr": [],
    }

    # initialize the early_stopping object
    model_path = os.path.join(path, "model.pkl")

    for epoch in range(epochs):
        # training
        (
            model,
            correct,
            training_loss,
            (TP, FP, TN, FN),
            y_list,
            yhat_list,
        ) = batch_iter(model, train_loader, optimizer, scheduler, device, training=True)

        useful_stuff = calc_metrics(
            N_train,
            train_loader,
            correct,
            training_loss,
            (TP, FP, TN, FN),
            y_list,
            yhat_list,
            useful_stuff,
            mtype="training",
        )

        # evaluate training
        (
            model,
            correct,
            training_loss,
            (TP, FP, TN, FN),
            y_list,
            yhat_list,
        ) = batch_iter(
            model, train_loader, optimizer, scheduler, device, training=False
        )

        useful_stuff = calc_metrics(
            N_train,
            train_loader,
            correct,
            training_loss,
            (TP, FP, TN, FN),
            y_list,
            yhat_list,
            useful_stuff,
            mtype="train",
        )

        print("=" * 25)

    return useful_stuff, None


def save_metrics(
    path,
    mtype,
    best_epoch=None,
    **metrics,
):
    """
    save metrics as csv files
    """
    if best_epoch != None:
        filename = mtype + "_metrics_best.txt"
    else:
        filename = mtype + "_metrics.txt"
    file_path = os.path.join(path, filename)
    with open(file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["[" + mtype + " average]"])
        writer.writerow(["ACC: {:.2}".format(metrics["acc"])])
        writer.writerow(["LOSS: {:.2}".format(metrics["loss"])])
        writer.writerow(["Recall: {:.2}".format(metrics["recall"])])
        writer.writerow(["Specificity: {:.2}".format(metrics["specificity"])])
        writer.writerow(["Precision: {:.2}".format(metrics["precision"])])
        writer.writerow(["NPV: {:.2}".format(metrics["npv"])])
        writer.writerow(["F1: {:.2}".format(metrics["f1"])])
        writer.writerow(["MCC: {:.2}".format(metrics["mcc"])])
        writer.writerow(["AUC: {:.2}".format(metrics["auc"])])


def final_metric(history, mtype="train", best_epoch=None):
    """
    Calculate metric.
    """

    if best_epoch != None:
        idx = best_epoch
    else:
        idx = -1

    # get metrics value
    (TP, FP, TN, FN) = history[mtype + "_metric"][idx]
    auc = history[mtype + "_auc"][idx]
    loss = history[mtype + "_loss"][idx]

    # calc metric
    acc = (TP + TN) / (TP + FP + TN + FN)
    recall = TP / (TP + FN) if TP != 0 else 0.0  # 召回率是在所有正樣本當中，能夠預測多少正樣本的比例
    specificity = TN / (TN + FP) if TN != 0 else 0.0  # 特異度是在所有負樣本當中，能夠預測多少負樣本的比例
    precision = TP / (TP + FP) if TP != 0 else 0.0  # 準確率為在所有預測為正樣本中，有多少為正樣本
    npv = TN / (TN + FN) if TN != 0 else 0.0  # npv為在所有預測為負樣本中，有多少為負樣本
    f1 = (
        (2 * recall * precision) / (recall + precision)
        if (recall + precision) != 0.0
        else 0.0
    )  # F1-score則是兩者的調和平均數
    mcc = (
        (TP * TN - FP * FN) / np.sqrt(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
        if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) != 0
        else 0.0
    )

    print("\n[" + mtype + " average]\n")
    print("ACC: {:.2}".format(acc))
    print("LOSS: {:.2}".format(loss))
    print()
    print("Recall: {:.2}".format(recall))
    print("Specificity: {:.2}".format(specificity))
    print("Precision: {:.2}".format(precision))
    print("NPV: {:.2}".format(npv))
    print()
    print("F1: {:.2}".format(f1))
    print("MCC: {:.2}".format(mcc))
    print("AUC: {:.2}".format(auc))
    print()

    return {
        "acc": acc,
        "loss": loss,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
        "npv": npv,
        "f1": f1,
        "mcc": mcc,
        "auc": auc,
    }


def plot_roc(history, fig_path="./", best_epoch=None, show=False):
    """
    plot roc curve and save as png
    """
    if best_epoch != None:
        idx = best_epoch
        filename = "roc_best.png"
    else:
        idx = -1
        filename = "roc.png"

    # train
    tr_auc = history["train_auc"][idx]
    tr_fpr = history["train_fpr"][idx]
    tr_tpr = history["train_tpr"][idx]
    plt.plot(tr_fpr, tr_tpr, label="Train AUC = %0.2f" % tr_auc)

    # valid
    if "valid_auc" in history:
        va_auc = history["valid_auc"][idx]
        va_fpr = history["valid_fpr"][idx]
        va_tpr = history["valid_tpr"][idx]
        plt.plot(va_fpr, va_tpr, label="valid AUC = %0.2f" % va_auc)

    if best_epoch:
        plt.title("Best epoch's ROC")
    else:
        plt.title("ROC")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.legend()
    plt.savefig(os.path.join(fig_path, filename), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_lr(metric, history, fig_path="./", best_epoch=None, show=False):
    metric = metric.lower()
    if best_epoch != None:
        plt.axvline(x=best_epoch, color="r", linestyle="--", label="early stopping")
    # train
    plt.plot(
        history["train_" + metric],
        label="train",
    )

    # valid
    if ("valid_" + metric) in history:
        plt.plot(
            history["valid_" + metric],
            label="valid",
        )

    plt.ylabel(metric)
    plt.xlabel("epochs")
    axes = plt.gca()
    if metric == "acc":
        axes.set_ylim([0.5, 1])
    else:
        axes.set_ylim([0, 1])
    plt.legend()
    plt.title("training / valid " + metric + " vs iterations")
    plt.grid()
    if best_epoch:
        filename = metric + "_best.png"
    else:
        filename = metric + ".png"
    plt.savefig(os.path.join(fig_path, filename), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def tokenizing(sent_list, sent_label, tokenizer):
    """
    Main tokenizing process.
    """

    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids_list = []
    attention_masks_list = []
    # For every sentence...
    for sent in sent_list:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=400,  # 512,  # Pad & truncate all sentences.
            padding="max_length",
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors="pt",  # Return pytorch tensors.
            truncation=True,
        )

        # Add the encoded sentence to the list.
        input_ids_list.append(encoded_dict["input_ids"])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks_list.append(encoded_dict["attention_mask"])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_masks = torch.cat(attention_masks_list, dim=0)
    if sent_label is not None:
        labels = torch.tensor(sent_label)
    else:
        labels = None

    return input_ids, attention_masks, labels


def eval_model(
    model,
    test_loader,
    N_test,
    device,
):
    """
    eval the data.
    """

    useful_stuff = {
        "test_loss": [],
        "test_acc": [],
        "test_auc": [],
        "test_metric": [],
        "test_fpr": [],
        "test_tpr": [],
    }

    # evaluate test metrics
    (
        model,
        correct,
        training_loss,
        (TP, FP, TN, FN),
        y_list,
        yhat_list,
    ) = batch_iter(model, test_loader, None, None, device, training=False)

    useful_stuff = calc_metrics(
        N_test,
        test_loader,
        correct,
        training_loss,
        (TP, FP, TN, FN),
        y_list,
        yhat_list,
        useful_stuff,
        mtype="test",
    )

    return useful_stuff
