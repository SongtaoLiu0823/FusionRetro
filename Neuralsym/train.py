import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import DataLoader
from tqdm import trange
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from model import TemplateNN_Highway, TemplateNN_FC
from dataset import FingerprintDataset

parser = argparse.ArgumentParser("train.py")
# mode & metadata
parser.add_argument("--expt_name", help="experiment name", type=str, default="")
parser.add_argument("--do_train", help="whether to train", action="store_true")
parser.add_argument("--do_test", help="whether to test", action="store_true")
parser.add_argument("--model", help="['Highway', 'FC']", type=str, default='Highway')
# file names
parser.add_argument("--templates_file", help="templates_file", type=str, default="training_templates")
parser.add_argument("--prodfps_prefix", help="npz file of product fingerprints", type=str)
parser.add_argument("--labels_prefix", help="npy file of labels", type=str)
parser.add_argument("--csv_prefix", help="csv file of various metadata about the rxn", type=str)
parser.add_argument("--radius", help="Fingerprint radius", type=int, default=2)
parser.add_argument("--min_freq", help="Min freq of template", type=int, default=1)
parser.add_argument("--fp_size", help="Fingerprint size", type=int, default=32681)
# parser.add_argument("--fp_type", help='Fingerprint type ["count", "bit"]', type=str, default="count")
# training params
parser.add_argument('--device', type=int, default=0)
parser.add_argument("--checkpoint", help="whether to save model checkpoints", action="store_true")
parser.add_argument("--random_seed", help="random seed", type=int, default=42)
parser.add_argument("--bs", help="batch size", type=int, default=128)
parser.add_argument("--bs_eval", help="batch size (valid/test)", type=int, default=256)
parser.add_argument("--learning_rate", help="learning rate", type=float, default=1e-3)
parser.add_argument("--epochs", help="num. of epochs", type=int, default=30)
parser.add_argument("--early_stop", help="whether to use early stopping", action="store_true") # type=bool, default=True) 
parser.add_argument("--early_stop_patience", help="num. of epochs tolerated without improvement in criteria before early stop", type=int, default=2)
parser.add_argument("--early_stop_min_delta", help="min. improvement in criteria needed to not early stop", type=float, default=1e-4)
parser.add_argument("--lr_scheduler_factor", help="factor by which to reduce LR (ReduceLROnPlateau)", type=float, default=0.3)
parser.add_argument("--lr_scheduler_patience", help="num. of epochs with no improvement after which to reduce LR (ReduceLROnPlateau)", type=int, default=1)
parser.add_argument("--lr_cooldown", help="epochs to wait before resuming normal operation (ReduceLROnPlateau)", type=int, default=0)
# model params
parser.add_argument("--hidden_size", help="hidden size", type=int, default=512)
parser.add_argument("--depth", help="depth", type=int, default=5)

args = parser.parse_args()  

DATA_FOLDER = Path(__file__).resolve().parent / 'data'
CHECKPOINT_FOLDER = Path(__file__).resolve().parent / 'checkpoint'

torch.manual_seed(args.random_seed)
random.seed(args.random_seed)
os.environ["PYTHONHASHSEED"] = str(args.random_seed)
np.random.seed(args.random_seed)


'''Loading template file'''

with open(f"{DATA_FOLDER}/{args.templates_file}", 'r') as f:
    templates = f.readlines()

templates_filtered = []
for p in templates:
    pa, cnt = p.strip().split(': ')
    if int(cnt) >= args.min_freq:
        templates_filtered.append(pa)


''' Model hyperparameter'''
if args.model == 'Highway':
    model = TemplateNN_Highway(
        output_size=len(templates_filtered),
        size=args.hidden_size,
        num_layers_body=args.depth,
        input_size=args.fp_size
    )
elif args.model == 'FC':
    model = TemplateNN_FC(
        output_size=len(templates_filtered),
        size=args.hidden_size,
        input_size=args.fp_size
    )
else:
    raise ValueError('Unrecognized model name')

device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

train_dataset = FingerprintDataset(args.prodfps_prefix+'_train.npz', args.labels_prefix+'_train.npy')
train_size = len(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

valid_dataset = FingerprintDataset(args.prodfps_prefix+'_valid.npz', args.labels_prefix+'_valid.npy')
valid_size = len(valid_dataset)
valid_loader = DataLoader(valid_dataset, batch_size=args.bs_eval, shuffle=False)
del train_dataset, valid_dataset

proposals_data_valid = pd.read_csv(f"{DATA_FOLDER}/{args.csv_prefix}_valid.csv", index_col=None, dtype='str')

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode='max', # monitor top-1 val accuracy
                factor=args.lr_scheduler_factor,
                patience=args.lr_scheduler_patience,
                cooldown=args.lr_cooldown,
                verbose=True
            )

train_losses, valid_losses = [], []
k_to_calc = [1, 2, 3, 5, 10, 20, 50, 100]
train_accs, val_accs = defaultdict(list), defaultdict(list)
max_valid_acc = float('-inf')
wait = 0
for epoch in trange(args.epochs):
    train_loss, train_correct, train_seen = 0, defaultdict(int), 0
    train_loader = tqdm(train_loader, desc='training')
    model.train()
    for data in train_loader:
        inputs, labels, idxs = data
        inputs, labels = inputs.to(device), labels.to(device)

        model.zero_grad()
        optimizer.zero_grad()
        outputs = model(inputs)
        # mask out rxn_smi w/ no valid template, giving loss = 0
        # logging.info(f'{outputs.shape}, {idxs.shape}, {(idxs < len(templates_filtered)).shape}')
        # [300, 33045], [300], [300]
        outputs = torch.where(
            (labels.view(-1, 1).expand_as(outputs) < len(templates_filtered)), outputs, torch.tensor([0.], device=outputs.device)
        )
        labels = torch.where(
            (labels < len(templates_filtered)), labels, torch.tensor([0.], device=labels.device).long()
        )
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_seen += labels.shape[0]
        outputs = nn.Softmax(dim=1)(outputs)

        for k in k_to_calc:
            batch_preds = torch.topk(outputs, k=k, dim=1)[1]
            train_correct[k] += torch.where(batch_preds == labels.view(-1,1).expand_as(batch_preds))[0].shape[0]

        train_loader.set_description(f"training: loss={train_loss/train_seen:.4f}, top-1 acc={train_correct[1]/train_seen:.4f}")
        train_loader.refresh()
    train_losses.append(train_loss/train_seen)
    for k in k_to_calc:
        train_accs[k].append(train_correct[k]/train_seen)

    model.eval()
    with torch.no_grad():
        valid_loss, valid_correct, valid_seen = 0, defaultdict(int), 0
        valid_loader = tqdm(valid_loader, desc='validating')
        for i, data in enumerate(valid_loader):
            inputs, labels, idxs = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            valid_seen += labels.shape[0]
            outputs = nn.Softmax(dim=-1)(outputs)

            for k in k_to_calc:
                batch_preds = torch.topk(outputs, k=k, dim=1)[1]
                valid_correct[k] += torch.where(batch_preds == labels.view(-1,1).expand_as(batch_preds))[0].shape[0]

            valid_loader.set_description(f"validating: top-1 acc={valid_correct[1]/valid_seen:.4f}") # loss={valid_loss/valid_seen:.4f}, 
            valid_loader.refresh()

    # valid_losses.append(valid_loss/valid_seen)
    for k in k_to_calc:
        val_accs[k].append(valid_correct[k]/valid_seen)

    lr_scheduler.step(val_accs[1][-1])

    if args.checkpoint and val_accs[1][-1] > max_valid_acc:
        # checkpoint model
        model_state_dict = model.state_dict()
        checkpoint_dict = {
            "epoch": epoch,
            "state_dict": model_state_dict, "optimizer": optimizer.state_dict(),
            "train_accs": train_accs, "train_losses": train_losses,
            "valid_accs": val_accs, "valid_losses": valid_losses,
            "max_valid_acc": max_valid_acc
        }
        checkpoint_filename = (
            CHECKPOINT_FOLDER
            / f"{args.expt_name}.pth.tar" # _{epoch:04d}
        )
        torch.save(checkpoint_dict, checkpoint_filename)

    if args.early_stop and max_valid_acc - val_accs[1][-1] > args.early_stop_min_delta:
        if args.early_stop_patience <= wait:
            break
        else:
            wait += 1
    else:
        wait = 0
        max_valid_acc = max(max_valid_acc, val_accs[1][-1])
