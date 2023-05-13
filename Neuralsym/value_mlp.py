import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import logging
import argparse
import random
import numpy as np
from tqdm import tqdm, trange
from torch.utils.data import TensorDataset, DataLoader, RandomSampler


class ValueMLP(nn.Module):
    def __init__(self, n_layers, fp_dim, latent_dim, dropout_rate):
        super(ValueMLP, self).__init__()
        self.n_layers = n_layers
        self.fp_dim = fp_dim
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        logging.info('Initializing value model: latent_dim=%d' % self.latent_dim)

        layers = []
        layers.append(nn.Linear(fp_dim, latent_dim))
        # layers.append(nn.BatchNorm1d(latent_dim,
        #                              track_running_stats=False))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))
        for _ in range(self.n_layers - 1):
            layers.append(nn.Linear(latent_dim, latent_dim))
            # layers.append(nn.BatchNorm1d(latent_dim,
            #                              track_running_stats=False))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
        layers.append(nn.Linear(latent_dim, 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, fps):
        x = fps
        x = self.layers(x)
        x = torch.log(1 + torch.exp(x))

        return x

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# ===================== model ====================== #
parser.add_argument('--fp_dim', type=int, default=2048)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--latent_dim', type=int, default=128)

# ==================== training ==================== #
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ValueMLP(
        n_layers=args.n_layers,
        fp_dim=args.fp_dim,
        latent_dim=args.latent_dim,
        dropout_rate=0.1
    )

all_fps = torch.load("fps.pt").to(device)
all_costs = torch.load("costs.pt").to(device)
index = list(range(all_fps.shape[0]))
random.shuffle(index)

train_fps = all_fps[index[0:int(0.9*len(index))]]
val_fps = all_fps[index[int(0.9*len(index)):]]

train_costs = all_costs[index[0:int(0.9*len(index))]]
val_costs = all_costs[index[int(0.9*len(index)):]]

train_data = TensorDataset(train_fps, train_costs)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)


val_data = TensorDataset(val_fps, val_costs)
val_sampler = RandomSampler(val_data)
val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)

model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
best_val_loss = np.inf
for epoch in trange(args.n_epochs):
    model.train()
    train_loss = 0
    train_seen = 0
    pbar = tqdm(train_dataloader)
    for data in pbar:
        optimizer.zero_grad()
        fps, values = data
        v_pred = model(fps).squeeze()
        loss = F.mse_loss(v_pred, values)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_seen += fps.shape[1]
        pbar.set_description('[loss: %f]' % (loss))
    

    model.eval()
    with torch.no_grad():
        val_loss = 0
        val_seen = 0
        pbar = tqdm(val_dataloader)
        for data in pbar:
            fps, values = data
            v_pred = model(fps).squeeze()
            loss = F.mse_loss(v_pred, values)
            val_loss += loss.item()
            val_seen += fps.shape[1]
            pbar.set_description('[loss: %f]' % (loss))

    logging.info(
                '[Epoch %d/%d] [training loss: %f] [validation loss: %f]' %
                (epoch, args.n_epochs, train_loss/train_seen, val_loss/val_seen)
            )
    if val_loss / val_seen < best_val_loss:
        best_val_loss = val_loss / val_seen
        torch.save(model.state_dict(), 'value_mlp.pkl')
