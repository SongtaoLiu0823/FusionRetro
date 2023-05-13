import argparse
import random
import numpy as np
from tqdm import tqdm, trange
from preprocess import get_dataset, convert_symbols_to_inputs, get_vocab_size
from modeling import TransformerConfig, Transformer, get_products_mask, get_reactants_mask, get_mutual_mask

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from rdkit.rdBase import DisableLog

DisableLog('rdApp.warning')

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--max_length', type=int, default=200, help='The max length of a molecule.')
parser.add_argument('--embedding_size', type=int, default=64, help='The size of embeddings')
parser.add_argument('--hidden_size', type=int, default=640, help='The size of hidden units')
parser.add_argument('--num_hidden_layers', type=int, default=3, help='Number of layers in encoder\'s module. Default 3.')
parser.add_argument('--num_attention_heads', type=int, default=10, help='Number of attention heads. Default 10.')
parser.add_argument('--intermediate_size', type=int, default=512, help='The size of hidden units of position-wise layer.')
parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument("--batch_size", default=64, type=int, help="Total batch size for training.")
parser.add_argument("--warmup", default=16000.0, type=float)
parser.add_argument("--l_factor", default=20.0, type=float)

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_gpu = torch.cuda.device_count()

config = TransformerConfig(vocab_size=get_vocab_size(),
                           embedding_size=args.embedding_size,
                           hidden_size=args.hidden_size,
                           num_hidden_layers=args.num_hidden_layers,
                           num_attention_heads=args.num_attention_heads,
                           intermediate_size=args.intermediate_size,
                           hidden_dropout_prob=args.hidden_dropout_prob)


# Get train data
train_products_list, train_reactants_list = get_dataset('train')
(train_products_input_ids, 
train_products_input_mask, 
train_reactants_input_ids, 
train_reactants_input_mask, 
train_label_ids) = convert_symbols_to_inputs(train_products_list, train_reactants_list, args.max_length)


train_products_input_ids = torch.LongTensor(train_products_input_ids).to(device)
train_reactants_input_ids = torch.LongTensor(train_reactants_input_ids).to(device)
train_label_ids = torch.LongTensor(train_label_ids).to(device)
train_products_input_mask = torch.FloatTensor(train_products_input_mask).to(device)
train_reactants_input_mask = torch.FloatTensor(train_reactants_input_mask).to(device)

# Construct dataset
train_data = TensorDataset(train_products_input_ids, 
                           train_reactants_input_ids, 
                           train_label_ids,
                           train_products_input_mask, 
                           train_reactants_input_mask)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

model = Transformer(config)
if num_gpu > 1:
    model = torch.nn.DataParallel(model)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.01)

global_step = 0
for epoch in trange(1, int(args.epochs)+1, desc="Epoch"):
    total_t = 0
    total_sum_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        global_step += 0.1
        lr = args.l_factor * min(1.0, global_step/args.warmup) / max(global_step, args.warmup)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        model.train()
        optimizer.zero_grad()
        products_ids, reactants_ids, label_ids, products_mask, reactants_mask = batch
        mutual_mask = get_mutual_mask([reactants_mask, products_mask])
        products_mask = get_products_mask(products_mask)
        reactants_mask = get_reactants_mask(reactants_mask)
        logits = model(products_ids, reactants_ids, products_mask, reactants_mask, mutual_mask)
        loss = F.cross_entropy(torch.reshape(logits, (-1, logits.shape[-1])), torch.flatten(label_ids))
        loss.backward()
        optimizer.step()
        total_t += torch.flatten(label_ids).size()[0]
        total_sum_loss += loss.item() * torch.flatten(label_ids).size()[0]
    print("Loss: %f.\n" %(total_sum_loss/total_t))
    if epoch % 100 == 0:
        torch.save(model, "models/epoch_%s_transformer.pkl" %(epoch))

