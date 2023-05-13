import numpy as np
import torch
import torch.nn as nn
import scipy
import multiprocessing
import json
import math
import argparse

from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun
from pathlib import Path
from typing import Dict, List
from scipy import sparse
from tqdm import trange, tqdm
from rdkit import Chem
from model import TemplateNN_Highway
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


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


def mol_smi_to_count_fp(mol_smi: str, radius: int = 2, fp_size: int = 32681, dtype: str = "int32") -> scipy.sparse.csr_matrix:
    fp_gen = GetMorganGenerator(radius=radius, useCountSimulation=True, includeChirality=True, fpSize=fp_size)
    mol = Chem.MolFromSmiles(mol_smi)
    uint_count_fp = fp_gen.GetCountFingerprint(mol)
    count_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)
    return sparse.csr_matrix(count_fp, dtype=dtype)


def get_dataset(phase):
    file_name = "%s_dataset.json" %phase
    products_list = []
    reactants_list = []
    retro_reaction_set = set()
    with open(file_name, 'r') as f:
        dataset = json.load(f)
        for _, reaction_trees in dataset.items():
            max_num_materials = 0
            final_retro_routes_list = None
            for i in range(1, int(reaction_trees['num_reaction_trees'])+1):
                if len(reaction_trees[str(i)]['materials']) > max_num_materials:
                    max_num_materials = len(reaction_trees[str(i)]['materials'])
                    final_retro_routes_list = reaction_trees[str(i)]['retro_routes']

            for retro_route in final_retro_routes_list:
                for retro_reaction in retro_route:
                    if retro_reaction not in retro_reaction_set:
                        retro_reaction_set.add(retro_reaction)
                        products_list.append(retro_reaction.split('>>')[0])
                        reactants_list.append(retro_reaction.split('>>')[1])

    return products_list, reactants_list



class Proposer:
    def __init__(self, infer_config: Dict) -> None:
        super().__init__()
        self.device = torch.device("cpu")

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
                        answer.append([sorted(list(sms)), -math.log10(score)]) # Tuple[precs, score] where precs is a List[str]
                        aim_size -= 1
                if aim_size == 0:
                    break
        return answer[:topk]


def get_reaction_cost(task):
    product, ground_truth_reactants = task
    reaction = product + '>>' + ground_truth_reactants
    product = Chem.MolToSmiles(Chem.MolFromSmiles(product))
    _, product = cano_smiles(product)
    ground_truth_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in ground_truth_reactants.split('.')]) 
    for rank, solution in enumerate(proposer.propose(product, topk=args.beam_size)):
        flag = False
        predict_reactants, cost = solution[0], solution[1]
        answer_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in predict_reactants])
        if answer_keys == ground_truth_keys:
            return reaction, cost
        if flag: break
    return reaction, np.inf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--beam_size", help="beam size", type=int, default=10)
    parser.add_argument("--num_cores", help="The number of cores", type=int, default=8)

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

    train_products_list, train_reactants_list = get_dataset('train')
    tasks = []
    for epoch in trange(0, len(train_products_list)):
        product = train_products_list[epoch]
        ground_truth_reactants = train_reactants_list[epoch]
        tasks.append((product, ground_truth_reactants))

    pool = multiprocessing.Pool(args.num_cores)
    reaction_cost = {}
    for result in tqdm(pool.imap_unordered(get_reaction_cost, tasks), total=len(tasks)):
        reaction, cost = result
        if cost != np.inf:
            reaction_cost[reaction] = cost

    with open('reaction_cost.json', 'w') as f:
        f.write(json.dumps(reaction_cost, indent=4))

