import os
import csv
import torch
import torch.nn.functional as F
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


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


def tokenizing_for_bert(train_list_x, train_list_y, tokenizer, train=True):
    """
    Tokenize abstracts and return data as Bert model input tensor.
    """

    print("tokenizing for bert input")

    # save k-fold data by dict
    # key: train + idx, value: tensor
    input_ids_dict = {}
    attention_masks_dict = {}
    labels_dict = {}

    # main process
    # For every fold...
    for idx in range(len(train_list_x)):

        sent_list = train_list_x[idx]
        sent_label = train_list_y[idx]

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
        labels = torch.tensor(sent_label)

        if train:
            input_ids_dict["tr_" + str(idx)] = input_ids
            attention_masks_dict["tr_" + str(idx)] = attention_masks
            labels_dict["tr_" + str(idx)] = labels
        else:
            input_ids_dict["va_" + str(idx)] = input_ids
            attention_masks_dict["va_" + str(idx)] = attention_masks
            labels_dict["va_" + str(idx)] = labels

    return input_ids_dict, attention_masks_dict, labels_dict


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
    type="training",
):
    (TP, FP, TN, FN) = confusion_matrix
    fpr, tpr, _ = metrics.roc_curve(y_list, yhat_list)
    auc = metrics.auc(fpr, tpr)
    acc = correct / N_data
    loss = training_loss / len(dataloader)
    useful_stuff = save_result(
        useful_stuff, acc, loss, (TP, FP, TN, FN), auc, fpr, tpr, type
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

    print(type + " loss: {0:.2f}".format(loss))
    print(type + " acc: {0:.2f}".format(acc))
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
    epochs=4,
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

    for _ in range(epochs):
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
            type="training",
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
            type="train",
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
            type="valid",
        )

        print("=" * 25)

    return useful_stuff


def final_metric(history, path, mtype="train"):
    """
    Calculate metric.
    """
    # init
    ACC = []
    LOSS = []
    RECALL = []
    SPECIFICITY = []
    PRECISION = []
    NPV = []
    F1 = []
    MCC = []
    AUC = []
    FPR = []
    TPR = []

    for i in range(len(history)):

        (TP, FP, TN, FN) = history[i]["_metric"][-1]
        auc = history[i][mtype + "_auc"][-1]
        fpr = history[i][mtype + "_fpr"][-1]
        tpr = history[i][mtype + "_tpr"][-1]
        loss = history[i][mtype + "_loss"][-1]

        acc = (TP + TN) / (TP + FP + TN + FN)

        recall = TP / (TP + FN) if TP != 0 else 0  # 召回率是在所有正樣本當中，能夠預測多少正樣本的比例
        specificity = TN / (TN + FP) if TN != 0 else 0  # 特異度是在所有負樣本當中，能夠預測多少負樣本的比例
        precision = TP / (TP + FP) if TP != 0 else 0  # 準確率為在所有預測為正樣本中，有多少為正樣本
        npv = TN / (TN + FN) if TN != 0 else 0  # npv為在所有預測為負樣本中，有多少為負樣本
        f1 = (
            (2 * recall * precision) / (recall + precision)
            if (recall + precision) != 0
            else 0
        )  # F1-score則是兩者的調和平均數

        mcc = (
            (TP * TN - FP * FN)
            / np.sqrt(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
            if ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) != 0
            else 0
        )

        ACC.append(acc)
        LOSS.append(loss)
        RECALL.append(recall)
        SPECIFICITY.append(specificity)
        PRECISION.append(precision)
        NPV.append(npv)
        F1.append(f1)
        MCC.append(mcc)
        AUC.append(auc)
        FPR.append(fpr)
        TPR.append(tpr)

    print("\n[" + mtype + " average]\n")
    print("ACC: {:.2}".format((np.mean(ACC))))
    print("LOSS: {:.2}".format(np.mean(LOSS)))
    print()
    print("Recall: {:.2}".format(np.mean(RECALL)))
    print("Specificity: {:.2}".format(np.mean(SPECIFICITY)))
    print("Precision: {:.2}".format(np.mean(PRECISION)))
    print("NPV: {:.2}".format(np.mean(NPV)))
    print()
    print("F1: {:.2}".format(np.mean(F1)))
    print("MCC: {:.2}".format(np.mean(MCC)))
    print("AUC: {:.2}".format(np.mean(AUC)))
    print()

    # save result
    save_metrics(
        path,
        mtype,
        train_metric,
        ACC,
        LOSS,
        RECALL,
        SPECIFICITY,
        PRECISION,
        NPV,
        F1,
        MCC,
        AUC,
    )


def save_metrics(
    path,
    mtype,
    ACC,
    LOSS,
    RECALL,
    SPECIFICITY,
    PRECISION,
    NPV,
    F1,
    MCC,
    AUC,
):
    """
    save metrics as csv files
    """
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["\n[" + mtype + " average]\n"])
        writer.writerow(["ACC: {:.2}".format((np.mean(ACC)))])
        writer.writerow(["LOSS: {:.2}".format(np.mean(LOSS))])
        writer.writerow(["Recall: {:.2}".format(np.mean(RECALL))])
        writer.writerow(["Specificity: {:.2}".format(np.mean(SPECIFICITY))])
        writer.writerow(["Precision: {:.2}".format(np.mean(PRECISION))])
        writer.writerow(["NPV: {:.2}".format(np.mean(NPV))])
        writer.writerow(["F1: {:.2}".format(np.mean(F1))])
        writer.writerow(["MCC: {:.2}".format(np.mean(MCC))])
        writer.writerow(["AUC: {:.2}".format(np.mean(AUC))])


def setting_path(model_name, batch_size, epochs):
    # setting path
    cwd = os.getcwd()
    print("cwd", cwd)

    folder_name = model_name + "_bs_" + str(batch_size) + "_epo" + str(epochs)

    metric_path = os.path.abspath(os.path.join(cwd, "result", folder_name, "metrics"))
    model_path = os.path.abspath(os.path.join(cwd, "result", folder_name, "model"))
    history_path = os.path.abspath(os.path.join(cwd, "result", folder_name, "history"))
    fig_path = os.path.abspath(os.path.join(cwd, "result", folder_name, "figures"))

    print("metric_path:", metric_path)
    print("model_path:", model_path)
    print("history_path:", history_path)
    print("fig_path:", fig_path)

    if not os.path.isdir(metric_path):
        os.makedirs(metric_path)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    if not os.path.isdir(history_path):
        os.makedirs(history_path)
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    return metric_path, model_path, history_path, fig_path


def calc_avg(training_hist):
    """
    Plot learning curve
    """

    a1 = a2 = a3 = a4 = []  # init

    for i in range(len(training_hist)):
        if i == 0:
            a1 = np.array(training_hist[0]["train_loss"].copy())
            a2 = np.array(training_hist[0]["valid_loss"].copy())
            a3 = np.array(training_hist[0]["train_acc"].copy())
            a4 = np.array(training_hist[0]["valid_acc"].copy())
            continue
        a1 = a1 + np.array(training_hist[i]["train_loss"])
        a2 = a2 + np.array(training_hist[i]["valid_loss"])
        a3 = a3 + np.array(training_hist[i]["train_acc"])
        a4 = a4 + np.array(training_hist[i]["valid_acc"])

    a1 /= len(training_hist)
    a2 /= len(training_hist)
    a3 /= len(training_hist)
    a4 /= len(training_hist)

    a1 = a1.tolist()
    a2 = a2.tolist()
    a3 = a3.tolist()
    a4 = a4.tolist()

    return a1, a2, a3, a4


def plot_roc(training_hist, fig_path, mtype="train"):
    """
    plot roc curve and save as png
    """
    for i in range(len(training_hist)):
        auc = training_hist[i][mtype + "_auc"][-1]
        fpr = training_hist[i][mtype + "_fpr"][-1]
        tpr = training_hist[i][mtype + "_tpr"][-1]

        plt.plot(fpr, tpr, label="Fold-" + str(i) + " AUC = %0.2f" % auc)

    plt.title(mtype + " roc curve")
    plt.legend(loc="lower right")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    filename = mtype + "-roc.png"
    plt.savefig(os.path.join(fig_path, filename))  # , bbox_inches="tight")
    plt.close()


def plot_figure(training_hist, fig_path):

    a1, a2, a3, a4 = calc_avg(training_hist)

    # color
    tr_color = ["#2ff5f2", "#2ff5e8", "#2ff5c0", "#2fbdf5", "#2f99f5"]
    val_color = ["#f5952f", "#f5ac2f", "#f5c02f", "#f5d72f", "#f5ee2f"]

    # train loss
    for idx in range(len(training_hist)):
        plt.plot(
            training_hist[idx]["train_loss"],
            "--",
            alpha=0.6,
            label="train" + str(idx),
            # color=color,
        )
    plt.plot(a1, label="average training")

    # valid loss
    for idx in range(len(training_hist)):
        plt.plot(
            training_hist[idx]["valid_loss"],
            "--",
            alpha=0.6,
            label="valid" + str(idx),
            # color=color,
        )

    plt.plot(a2, label="average valid")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.legend()
    plt.title("training / valid loss vs iterations")
    plt.grid()
    plt.savefig(os.path.join(fig_path, "loss.png"))  # , bbox_inches="tight")
    plt.close()

    # train acc
    for idx in range(len(training_hist)):
        plt.plot(
            training_hist[idx]["train_acc"],
            "--",
            alpha=0.6,
            label="train" + str(idx),
            # color=color,
        )
    plt.plot(a3, label="average training")

    # valid acc
    for idx in range(len(training_hist)):
        plt.plot(
            training_hist[idx]["valid_acc"],
            "--",
            alpha=0.6,
            label="valid" + str(idx),
            # color=color,
        )
    plt.plot(a4, label="average valid")
    plt.ylabel("acc")
    plt.xlabel("epochs")
    axes = plt.gca()
    axes.set_ylim([0.5, 1])
    plt.legend()
    plt.title("training / valid acc vs iterations")
    plt.grid()
    plt.savefig(os.path.join(fig_path, "acc.png"))  # , bbox_inches="tight")
    plt.close()

    # roc
    plot_roc(training_hist, fig_path, mtype="train")
    plot_roc(training_hist, fig_path, mtype="valid")
