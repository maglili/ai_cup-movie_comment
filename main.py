import argparse
import random
import pickle
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)
from tools import *


# =================================argparser================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--mode",
    nargs="?",
    type=str,
    choices=["train", "test", "predict"],
    default="train",
    help="train model or evaluate data.",
)
parser.add_argument(
    "-bs",
    "--batch_size",
    nargs="?",
    type=int,
    default=4,
    help="Number of training epochs.",
)
parser.add_argument(
    "-epo",
    "--epochs",
    nargs="?",
    type=int,
    default=4,
    help="Number of training epochs.",
)
parser.add_argument(
    "--model_name",
    nargs="?",
    type=str,
    default="bert-base-cased",
    help="Huggingface model name",
)
args = parser.parse_args()
# =================================argparser================================

# keep reandom seed
seed_val = 0
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)

if args.mode == "train":

    # check gpu
    device = get_device()

    # setting path
    metric_path, model_path, history_path, fig_path = setting_path(
        args.model_name,
        args.batch_size,
        args.epochs,
        args.mode,
    )

    path = "./data/train_valid_split/level_1/"
    with open(os.path.join(path, "X_tr.pkl"), "rb") as f:
        X_tr = pickle.load(f)
    with open(os.path.join(path, "X_va.pkl"), "rb") as f:
        X_va = pickle.load(f)
    with open(os.path.join(path, "y_tr.pkl"), "rb") as f:
        y_tr = pickle.load(f)
    with open(os.path.join(path, "y_va.pkl"), "rb") as f:
        y_va = pickle.load(f)

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=False)

    input_ids_tr, attention_masks_tr, labels_tr = tokenizing(
        X_tr.values, y_tr.values, tokenizer
    )
    input_ids_va, attention_masks_va, labels_va = tokenizing(
        X_va.values, y_va.values, tokenizer
    )

    # Holdout dataset
    Trainset = TensorDataset(input_ids_tr, attention_masks_tr, labels_tr)
    Validset = TensorDataset(input_ids_va, attention_masks_va, labels_va)

    # Training
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
        hidden_dropout_prob=0.35,
        attention_probs_dropout_prob=0.25,
    )
    model.to(device)

    # This code is taken from:
    # https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L102

    # Don't apply weight decay to any parameters whose names include these tokens.
    # (Here, the BERT doesn't have `gamma` or `beta` parameters, only `bias` terms)
    no_decay = ["bias", "LayerNorm.weight"]

    # Separate the `weight` parameters from the `bias` parameters.
    # - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01.
    # - For the `bias` parameters, the 'weight_decay_rate' is 0.0.
    optimizer_grouped_parameters = [
        # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        # Filter for parameters which *do* include those.
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]

    # Note - `optimizer_grouped_parameters` only includes the parameter values, not
    # the names.

    N_train = len(Trainset)
    N_test = len(Validset)
    print("Num of train samples:", N_train)
    print("Num of valid samples:", N_test)
    print()

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=4e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
    )

    train_loader = DataLoader(Trainset, shuffle=True, batch_size=args.batch_size)
    valid_loader = DataLoader(Validset, shuffle=False, batch_size=args.batch_size)

    # Total number of training steps is [number of batches] x [number of epochs].
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_loader) * args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps * 0.1, num_training_steps=total_steps
    )

    history, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        N_train=N_train,
        N_test=N_test,
        device=device,
        scheduler=scheduler,
        path=model_path,
        epochs=args.epochs,
        patience=3,
    )

    # save trainin_history
    with open(os.path.join(history_path, "hist.pkl"), "wb") as f:
        pickle.dump(history, f)

    # metric
    tr_metric = final_metric(history, mtype="train", best_epoch=best_epoch)
    va_metric = final_metric(history, mtype="valid", best_epoch=best_epoch)
    save_metrics(metric_path, "train", best_epoch=None, **tr_metric)
    save_metrics(metric_path, "valid", best_epoch=None, **va_metric)

    # plot learning curve
    plot_lr("acc", history, fig_path=fig_path, best_epoch=best_epoch, show=False)
    plot_lr("loss", history, fig_path=fig_path, best_epoch=best_epoch, show=False)
    plot_roc(history, fig_path=fig_path, best_epoch=best_epoch, show=False)


elif args.mode == "test":
    # load data
    with open("./data/train_test_split/X_te.pkl", "rb") as f:
        X_test = pickle.load(f)
    with open("./data/train_test_split/y_te.pkl", "rb") as f:
        y_test = pickle.load(f)

    # check gpu
    device = get_device()

    # setting path
    metric_path, model_path, history_path, _ = setting_path(
        args.model_name, args.batch_size, args.epochs, args.mode
    )

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=False)
    input_ids_te, attention_masks_te, labels_te = tokenizing(
        X_test.values, y_test.values, tokenizer
    )

    # dataset
    testdataset = TensorDataset(input_ids_te, attention_masks_te, labels_te)
    print("test dataset:", len(testdataset))

    # load in model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)
    PATH = os.path.join(model_path, "model.pkl")
    model.load_state_dict(torch.load(PATH))

    # evaluate
    N_test = len(testdataset)
    test_loader = DataLoader(testdataset, shuffle=False, batch_size=16)

    history = eval_model(
        model=model,
        test_loader=test_loader,
        N_test=N_test,
        device=device,
    )

    # save trainin_history
    with open(os.path.join(history_path, "hist.pkl"), "wb") as f:
        pickle.dump(history, f)

    # save metric
    te_metric = final_metric(history, mtype="test")
    save_metrics(metric_path, "test", best_epoch=None, **te_metric)

else:
    with open("./data/test_rm_br.pkl", "rb") as f:
        test = pickle.load(f)
    test_data = test["review"].values

    # check gpu
    device = get_device()

    # setting path
    _, model_path, _, _ = setting_path(
        args.model_name, args.batch_size, args.epochs, args.mode
    )

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, do_lower_case=False)
    input_ids, attention_masks, _ = tokenizing(test_data, None, tokenizer)

    # dataset
    testdataset = TensorDataset(input_ids, attention_masks)
    print("Prediction dataset:", len(testdataset))

    # load in model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)
    PATH = os.path.join(model_path, "model.pkl")
    model.load_state_dict(torch.load(PATH))

    # loader
    test_loader = DataLoader(testdataset, shuffle=False, batch_size=16)

    # prediction
    pred = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)

            output = model(b_input_ids, attention_mask=b_input_mask)
            logits = output[0]

            _, yhat = torch.max(logits.data, 1)

            pred.extend(yhat.cpu().detach().numpy())

    print("Answer and prediction have same length:", len(pred) == len(test))

    # save result
    path = os.path.abspath(os.path.join(metric_path, ".."))
    print("output:", path)
    submission = pd.DataFrame({"ID": test["ID"].values, "sentiment": pred})
    submission.to_csv(
        os.path.join(path, "submission.csv"), encoding="utf-8", index=False
    )
