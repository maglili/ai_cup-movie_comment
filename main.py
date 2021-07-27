import argparse
import random
import pickle
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
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
    choices=["train", "eval"],
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
    "--model",
    nargs="?",
    type=str,
    default="bert-base-cased",
    help="Huggingface model name",
)
args = parser.parse_args()
# =================================argparser================================

if args.mode == "train":
    print("Mode:", args.mode)

    # keep reandom seed
    seed_val = 0
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)

    # check gpu
    device = get_device()

    # setting path
    metric_path, model_path, history_path, fig_path = setting_path(
        args.model, args.batch_size, args.epochs, train="train"
    )

    # load data
    with open("./data/pkl/X_train_list.pkl", "rb") as f:
        X_train_list = pickle.load(f)
    with open("./data/pkl/X_valid_list.pkl", "rb") as f:
        X_valid_list = pickle.load(f)
    with open("./data/pkl/y_train_list.pkl", "rb") as f:
        y_train_list = pickle.load(f)
    with open("./data/pkl/y_valid_list.pkl", "rb") as f:
        y_valid_list = pickle.load(f)

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

    (
        input_ids_train_dict,
        attention_masks_train_dict,
        labels_train_dict,
    ) = tokenizing_for_bert(X_train_list, y_train_list, tokenizer)

    input_ids_cv_dict, attention_masks_cv_dict, labels_cv_dict = tokenizing_for_bert(
        X_valid_list, y_valid_list, tokenizer, train=False
    )

    # Prepare torch dataset
    tr_set = []
    va_set = []
    for idx in range(len(input_ids_train_dict)):
        tr_set.append(
            TensorDataset(
                input_ids_train_dict["tr_" + str(idx)],
                attention_masks_train_dict["tr_" + str(idx)],
                labels_train_dict["tr_" + str(idx)],
            )
        )
        va_set.append(
            TensorDataset(
                input_ids_cv_dict["va_" + str(idx)],
                attention_masks_cv_dict["va_" + str(idx)],
                labels_cv_dict["va_" + str(idx)],
            )
        )

    # hypterparameter
    print("epochs:", args.epochs)
    print("batch_size:", args.batch_size)
    print()

    # training
    training_hist = []

    for fold in tqdm(range(len(tr_set))):

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
            hidden_dropout_prob=0.4,
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
                "weight_decay_rate": 0.1,
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

        N_train = len(tr_set[fold])
        N_test = len(va_set[fold])
        print("\n[Fold]:", fold)
        print("Num of train samples:", N_train)
        print("Num of valid samples:", N_test)
        print()

        optimizer = AdamW(
            model.parameters(),
            lr=4e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        )

        train_dataloader = DataLoader(
            tr_set[fold], shuffle=True, batch_size=args.batch_size
        )

        validation_dataloader = DataLoader(
            va_set[fold], shuffle=False, batch_size=args.batch_size
        )

        # Total number of training steps is [number of batches] x [number of epochs].
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_dataloader) * args.epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps * 0.1,
            num_training_steps=total_steps,
        )

        history = train_model(
            model=model,
            train_loader=train_dataloader,
            valid_loader=validation_dataloader,
            optimizer=optimizer,
            N_train=N_train,
            N_test=N_test,
            device=device,
            scheduler=scheduler,
            path=model_path,
            epochs=args.epochs,
            patience=3,
        )

        training_hist.append(history)
        print("*" * 25)
        print("*" * 25)
        print("*" * 25)

    # save trainin_history
    with open(os.path.join(history_path, "/hist.pkl"), "wb") as f:
        pickle.dump(training_hist, f)

    # save model
    torch.save(model.state_dict(), os.path.join(model_path, "model.pkl"))

    # metric
    final_metric(training_hist, metric_path=metric_path, mtype="train")
    final_metric(training_hist, metric_path=metric_path, mtype="valid")

    # plot learning curve
    plot_figure(training_hist, fig_path)


else:
    print("Mode:", args.mode)
