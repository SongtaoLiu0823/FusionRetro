import os
import numpy as np
import torch
import torch.nn as nn
import logging
import pandas as pd
import math
import json
import argparse

from copy import deepcopy
from rdkit.Chem import AllChem
from rdkit import Chem
from tqdm import tqdm
from gln.common.cmd_args import cmd_args
from gln.common.consts import DEVICE
from gln.test.model_inference import RetroGLN


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


def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool)
    arr[onbits] = 1

    if pack:
        arr = np.packbits(arr)
    fp = 1 * np.array(arr)

    return fp


def value_fn(smi):
    fp = smiles_to_fp(smi, fp_dim=local_args.fp_dim).reshape(1,-1)
    fp = torch.FloatTensor(fp).to(DEVICE)
    v = value_model(fp).item()
    return v


def cano_smiles(smiles):
    try:
        tmp = Chem.MolFromSmiles(smiles)
        if tmp is None:
            return None, smiles        
        tmp = Chem.RemoveHs(tmp)
        if tmp is None:
            return None, smiles
        [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
        return tmp, Chem.MolToSmiles(tmp)            
    except:
        return None, smiles


def get_inference_answer(smiles, beam_size):
    pred_struct = model.run(smiles, 5*beam_size, 5*beam_size, rxn_type='UNK')
    if pred_struct is None:
        return []
    reactants_list = pred_struct['reactants']
    scores_list = pred_struct['scores']
    answer = []
    aim_size = beam_size
    for i in range(len(reactants_list)):
        if aim_size == 0:
            break
        reactants = reactants_list[i].split('.')
        score = scores_list[i]
        num_valid_reactant = 0
        sms = set()
        for r in reactants:
            m = Chem.MolFromSmiles(r)
            if m is not None:
                num_valid_reactant += 1
                sms.add(Chem.MolToSmiles(m))
        if num_valid_reactant != len(reactants):
            continue
        if len(sms):
            try:
                answer.append([sorted(list(sms)), -math.log10(score)])
            except:
                answer.append([sorted(list(sms)), -math.log10(score+1e-10)]) 
            aim_size -= 1

    return answer


def load_dataset(split):
    file_name = "%s_dataset.json" % split
    file_name = os.path.expanduser(file_name)
    dataset = [] # (product_smiles, materials_smiles, depth)
    with open(file_name, 'r') as f:
        _dataset = json.load(f)
        for _, reaction_trees in _dataset.items():
            product = reaction_trees['1']['retro_routes'][0][0].split('>')[0]
            product = Chem.MolToSmiles(Chem.MolFromSmiles(product))
            _, product = cano_smiles(product)
            materials_list = []
            for i in range(1, int(reaction_trees['num_reaction_trees'])+1):
                materials_list.append(reaction_trees[str(i)]['materials'])
            dataset.append({
                "product": product,
                "targets": materials_list, 
                "depth": reaction_trees['depth']
            })

    return dataset


def check_reactant_is_material(reactant):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] in stock_inchikeys


def check_reactants_are_material(reactants):
    for reactant in reactants:
        if Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] not in stock_inchikeys:
            return False
    return True


def get_route_result(task):
    max_depth = task["depth"]
    # Initialization
    answer_set = []
    queue = []
    queue.append({
        "score": value_fn(task["product"]),
        "routes_info": [{"route": [task["product"]], "depth": 0}],  # List of routes information
        "starting_materials": [],
    })
    while True:
        if len(queue) == 0:
            break
        nxt_queue = []
        for item in queue:
            score = item["score"]
            routes_info = item["routes_info"]
            starting_materials = item["starting_materials"]
            first_route_info = routes_info[0]
            first_route, depth = first_route_info["route"], first_route_info["depth"]
            if depth > max_depth:
                continue
            expansion_mol = first_route[-1]
            for expansion_solution in get_inference_answer(first_route[-1], local_args.beam_size):
                iter_routes = deepcopy(routes_info)
                iter_routes.pop(0)
                iter_starting_materials = deepcopy(starting_materials)
                expansion_reactants, reaction_cost = expansion_solution[0], expansion_solution[1]
                expansion_reactants = sorted(expansion_reactants)
                if check_reactants_are_material(expansion_reactants) and len(iter_routes) == 0:
                    answer_set.append({
                        "score": score+reaction_cost-value_fn(expansion_mol),
                        "starting_materials": iter_starting_materials+expansion_reactants,
                        })
                else:
                    estimation_cost = 0
                    for reactant in expansion_reactants:
                        if check_reactant_is_material(reactant):
                            iter_starting_materials.append(reactant)
                        else:
                            estimation_cost += value_fn(reactant)
                            iter_routes = [{"route": first_route+[reactant], "depth": depth+1}] + iter_routes
                    nxt_queue.append({
                        "score": score+reaction_cost+estimation_cost-value_fn(expansion_mol),
                        "routes_info": iter_routes,
                        "starting_materials": iter_starting_materials
                    })
        queue = sorted(nxt_queue, key=lambda x: x["score"])[:local_args.beam_size]
            
    answer_set = sorted(answer_set, key=lambda x: x["score"])
    record_answers = set()
    final_answer_set = []
    for item in answer_set:
        score = item["score"]
        starting_materials = item["starting_materials"]
        answer_keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(m))[:14] for m in starting_materials]
        if '.'.join(sorted(answer_keys)) not in record_answers:
            record_answers.add('.'.join(sorted(answer_keys)))
            final_answer_set.append({
                "score": score,
                "answer_keys": answer_keys
            })
    final_answer_set = sorted(final_answer_set, key=lambda x: x["score"])[:local_args.beam_size]

    # Calculate answers
    ground_truth_keys_list = [
        set([
            Chem.MolToInchiKey(Chem.MolFromSmiles(target))[:14] for target in targets
        ]) for targets in task["targets"]
    ]
    for rank, answer in enumerate(final_answer_set):
        answer_keys = set(answer["answer_keys"])
        for ground_truth_keys in ground_truth_keys_list:
            if ground_truth_keys == answer_keys:
                return max_depth, rank
    
    return max_depth, None


if __name__ == "__main__":
    cmd_opt = argparse.ArgumentParser(description='Argparser for retro star search')
    cmd_opt.add_argument('--fp_dim', type=int, default=2048)
    cmd_opt.add_argument('--n_layers', type=int, default=1)
    cmd_opt.add_argument('--latent_dim', type=int, default=128)
    cmd_opt.add_argument('-epoch_for_search', default=10, type=int, help='model for search')
    cmd_opt.add_argument("-beam_size", help="beam size", type=int, default=5)
    local_args, _ = cmd_opt.parse_known_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    value_model = ValueMLP(
            n_layers=local_args.n_layers,
            fp_dim=local_args.fp_dim,
            latent_dim=local_args.latent_dim,
            dropout_rate=0.1
        ).to(DEVICE)
    value_model.load_state_dict(torch.load('value_mlp.pkl',  map_location=DEVICE))
    value_model.eval()

    model_dump = os.path.join(cmd_args.save_dir, 'model-%d.dump' % local_args.epoch_for_search)
    model = RetroGLN(model_dump)
    model.gln.to(DEVICE)

    stock = pd.read_hdf('zinc_stock_17_04_20.hdf5', key="table")  
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    overall_result = np.zeros((local_args.beam_size, 2))
    depth_hit = np.zeros((2, 15, local_args.beam_size))
    tasks = load_dataset("test")
    for task in tqdm(tasks):
        max_depth, rank = get_route_result(task)
        overall_result[:, 1] += 1
        depth_hit[1, max_depth, :] += 1
        if rank is not None:
            overall_result[rank:, 0] += 1
            depth_hit[0, max_depth, rank:] += 1

    print("overall_result: ", overall_result, 100 * overall_result[:, 0] / overall_result[:, 1])
    print("depth_hit: ", depth_hit, 100 * depth_hit[0, :, :] / depth_hit[1, :, :])
