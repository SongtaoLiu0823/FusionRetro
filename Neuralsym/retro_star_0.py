import numpy as np
import torch
import torch.nn as nn
import json
import os
import pandas as pd
import math
import scipy
import argparse
import multiprocessing

from copy import deepcopy
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun
from pathlib import Path
from typing import Dict, List
from scipy import sparse
from tqdm import tqdm
from rdkit import Chem, DataStructs
from model import TemplateNN_Highway
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


def mol_smi_to_count_fp(mol_smi: str, radius: int = 2, fp_size: int = 32681, dtype: str = "int32") -> scipy.sparse.csr_matrix:
    fp_gen = GetMorganGenerator(radius=radius, useCountSimulation=True, includeChirality=True, fpSize=fp_size)
    mol = Chem.MolFromSmiles(mol_smi)
    uint_count_fp = fp_gen.GetCountFingerprint(mol)
    count_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)
    return sparse.csr_matrix(count_fp, dtype=dtype)


class Proposer:
    def __init__(self, infer_config: Dict) -> None:
        super().__init__()
        self.device = device

        print(f"Loading templates from file: {infer_config['templates_file']}")
        with open(f"{DATA_FOLDER}/{infer_config['templates_file']}", 'r') as f:
            templates = f.readlines()
        self.templates_filtered = []
        for p in templates:
            pa, cnt = p.strip().split(': ')
            if int(cnt) >= infer_config['min_freq']:
                self.templates_filtered.append(pa)
        print(f'Total number of template patterns: {len(self.templates_filtered)}')

        self.model, self.indices = self.build_model(infer_config)
        self.model.eval()
        print('Done initializing proposer\n')

    def build_model(self, infer_config: Dict):
         # load model from checkpoint
        checkpoint = torch.load(
            f"{CHECKPOINT_FOLDER}/{infer_config['expt_name']}.pth.tar",
            map_location=self.device,
        )
        model = TemplateNN_Highway(
            output_size=len(self.templates_filtered),
            size=infer_config['hidden_size'],
            num_layers_body=infer_config['depth'],
            input_size=infer_config['final_fp_size']
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.to(self.device)

        indices = np.loadtxt(f"{DATA_FOLDER}/variance_indices.txt").astype('int')
        return model, indices

    def propose(self, 
                smi: str,
                topk: int = 5,
                **kwargs) -> List[Dict[str, List]]:

        answer = []
        with torch.no_grad():
            prod_fp = mol_smi_to_count_fp(smi, infer_config['radius'], infer_config['orig_fp_size'])
            logged = sparse.csr_matrix(np.log(prod_fp.toarray() + 1))
            final_fp = logged[:, self.indices]
            final_fp = torch.as_tensor(final_fp.toarray()).float().to(self.device)

            outputs = self.model(final_fp)
            outputs = nn.Softmax(dim=1)(outputs)
            preds = torch.topk(outputs, k=100, dim=1)[1].squeeze(dim=0).cpu().numpy()

            aim_size = topk
            for idx in preds:
                score = outputs[0, idx.item()].item()
                template = self.templates_filtered[idx.item()]
                try:
                    rxn = rdchiralReaction(template)
                    prod = rdchiralReactants(smi)
                    precs = rdchiralRun(rxn, prod)
                except:
                    precs = 'N/A'
                if precs != 'N/A' and precs != []:
                    reactants = set(precs[0].split("."))
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
                        answer.append([sorted(list(sms)), -math.log10(score+1e-10)]) # Tuple[precs, score] where precs is a List[str]
                        aim_size -= 1
                if aim_size == 0:
                    break
        return answer[:topk]


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
        "score": 0.0,
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
            for expansion_solution in proposer.propose(first_route[-1], topk=args.beam_size):
                iter_routes = deepcopy(routes_info)
                iter_routes.pop(0)
                iter_starting_materials = deepcopy(starting_materials)
                expansion_reactants, expansion_score = expansion_solution[0], expansion_solution[1]
                expansion_reactants = sorted(expansion_reactants)
                if check_reactants_are_material(expansion_reactants) and len(iter_routes) == 0:
                    answer_set.append({
                        "score": score+expansion_score,
                        "starting_materials": iter_starting_materials+expansion_reactants,
                        })
                else:
                    for reactant in expansion_reactants:
                        if check_reactant_is_material(reactant):
                            iter_starting_materials.append(reactant)
                        else:
                            iter_routes = [{"route": first_route+[reactant], "depth": depth+1}] + iter_routes
                    nxt_queue.append({
                        "score": score+expansion_score,
                        "routes_info": iter_routes,
                        "starting_materials": iter_starting_materials
                    })
        queue = sorted(nxt_queue, key=lambda x: x["score"])[:args.beam_size]
            
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
    final_answer_set = sorted(final_answer_set, key=lambda x: x["score"])[:args.beam_size]

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
    parser = argparse.ArgumentParser("dfs_bfs_search.py")
    parser.add_argument("--beam_size", help="beam size", type=int, default=5)
    parser.add_argument("--num_cores", help="number of cores", type=int, default=64)
    args = parser.parse_args()
    device = torch.device("cpu")

    DATA_FOLDER = Path(__file__).resolve().parent / 'data'
    CHECKPOINT_FOLDER = Path(__file__).resolve().parent / 'checkpoint'

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    infer_config = {
        'templates_file': 'training_templates',
        'min_freq': 1,
        'expt_name': 'Highway_42_depth0_dim300_lr1e3_stop2_fac30_pat1',
        'hidden_size': 300,
        'depth': 0,
        'orig_fp_size': 1000000,
        'final_fp_size': 32681,
        'radius': 2,
    }


    proposer = Proposer(infer_config)

    stock = pd.read_hdf('zinc_stock_17_04_20.hdf5', key="table")  
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    overall_result = np.zeros((args.beam_size, 2))
    depth_hit = np.zeros((2, 15, args.beam_size))
    tasks = load_dataset("test")
    pool = multiprocessing.Pool(args.num_cores)
    for result in tqdm(pool.imap_unordered(get_route_result, tasks), total=len(tasks)):
        max_depth, rank = result
        overall_result[:, 1] += 1
        depth_hit[1, max_depth, :] += 1
        if rank is not None:
            overall_result[rank:, 0] += 1
            depth_hit[0, max_depth, rank:] += 1

    print("overall_result: ", overall_result, 100 * overall_result[:, 0] / overall_result[:, 1])
    print("depth_hit: ", depth_hit, 100 * depth_hit[0, :, :] / depth_hit[1, :, :])
