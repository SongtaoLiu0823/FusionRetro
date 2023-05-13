#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate MEGAN model.

"""

import os

from rdkit.Chem import Mol

from src import config
from src.config import get_featurizer
from src.feat.megan_graph import MeganTrainingSamplesFeaturizer

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import logging
import math
import argh
import gin
import numpy as np
import torch
import pandas as pd
from tqdm import trange
from copy import deepcopy
from src.utils import mol_to_unmapped, mol_to_unmapped_smiles, mark_reactants
from rdkit import Chem
from tqdm import tqdm
from src.feat.utils import fix_explicit_hs

# noinspection PyUnresolvedReferences
from bin.train import train_megan
from src.model.megan import Megan
from src.model.beam_search import beam_search
from src.model.megan_utils import get_base_action_masks, RdkitCache
from src.utils import load_state_dict
from src.utils.dispatch_utils import run_with_redirection

logger = logging.getLogger(__name__)


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


def remap_product_to_canonical(input_mol: Mol):
    """
    Re-maps reaction according to order of atoms in RdKit - this makes sure that stereochemical SMILES are canonical.
    Note: this method does not transfer any information from target molecule to the input molecule
    (the input molecule is mapped according to its order of atoms in its canonical SMILES)
    """

    # converting Mol to smiles and again to Mol makes atom order canonical
    input_mol = Chem.MolFromSmiles(Chem.MolToSmiles(input_mol))

    for i, a in enumerate(input_mol.GetAtoms()):
        a.SetAtomMapNum(i + 1)

    return input_mol


def load_dataset(split):
    file_name = "data/%s_dataset.json" % split
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
    try:
        reactant_inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))
    except:
        return False
    return reactant_inchikey in stock_inchikeys


def check_reactants_are_material(reactants):
    for reactant in reactants:
        try:
            reactant_inchikey = Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))
        except:
            return False
        if reactant_inchikey not in stock_inchikeys:
            return False
    return True


def evaluate_megan(save_path: str, beam_size: int = 5, max_gen_steps: int = 16, beam_batch_size: int = 1, n_max_atoms: int = 200):
    """
    Evaluate MEGAN model
    """
    config_path = os.path.join(save_path, 'config.gin')
    gin.parse_config_file(config_path)

    featurizer_key = gin.query_parameter('train_megan.featurizer_key')
    featurizer = get_featurizer(featurizer_key)
    assert isinstance(featurizer, MeganTrainingSamplesFeaturizer)
    action_vocab = featurizer.get_actions_vocabulary(save_path)

    base_action_masks = get_base_action_masks(n_max_atoms + 1, action_vocab=action_vocab)

    logger.info("Creating model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    model_path = os.path.join(save_path, 'model_best.pt')
    checkpoint = load_state_dict(model_path)
    model = Megan(n_atom_actions=action_vocab['n_atom_actions'], n_bond_actions=action_vocab['n_bond_actions'],
                  prop2oh=action_vocab['prop2oh']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    reaction_types = None
    rdkit_cache = RdkitCache(props=action_vocab['props'])

    test_set = load_dataset("test")
    overall_result = np.zeros((beam_size, 2))
    depth_hit = np.zeros((2, 15, beam_size))
    for epoch in trange(0, len(test_set)):
        # Initialization
        answer_set = []
        queue = []
        queue.append({
            "score": 0.0,
            "routes_info": [{"route": [test_set[epoch]["product"]], "depth": 0}],  # List of routes information
            "starting_materials": [],
        })
        max_depth = test_set[epoch]["depth"]
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
                
                ## Get expansion result
                input_mols = []
                input_mol = Chem.MolFromSmiles(first_route[-1])

                # remap product molecule according to canonical SMILES atom order
                try:
                    input_mol = remap_product_to_canonical(input_mol)
                except:
                    continue

                # fix a bug in marking explicit Hydrogen atoms by RdKit
                try:
                    input_mol = fix_explicit_hs(input_mol)
                except:
                    print("Error")

                input_mols.append(input_mol)
                with torch.no_grad():
                    beam_search_results = beam_search([model], input_mols, rdkit_cache=rdkit_cache, max_steps=max_gen_steps,
                                                        beam_size=2*beam_size, batch_size=beam_batch_size,
                                                        base_action_masks=base_action_masks, max_atoms=n_max_atoms,
                                                        reaction_types=reaction_types,
                                                        action_vocab=action_vocab)[0]
                expan_solution_list = []
                aim_size = beam_size
                for i in range(len(beam_search_results)):
                    if aim_size == 0:
                        break
                    reactants = beam_search_results[i]['final_smi_unmapped'].split('.')
                    score = beam_search_results[i]['prob']
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
                        expan_solution_list.append([sorted(list(sms)), -math.log10(score)]) 
                        aim_size -= 1


                for expansion_solution in expan_solution_list:
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
            queue = sorted(nxt_queue, key=lambda x: x["score"])[:beam_size]
                
        answer_set = sorted(answer_set, key=lambda x: x["score"])
        record_answers = set()
        final_answer_set = []
        for item in answer_set:
            score = item["score"]
            starting_materials = item["starting_materials"]
            answer_keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(m)) for m in starting_materials]
            if '.'.join(sorted(answer_keys)) not in record_answers:
                record_answers.add('.'.join(sorted(answer_keys)))
                final_answer_set.append({
                    "score": score,
                    "answer_keys": answer_keys
                })
        final_answer_set = sorted(final_answer_set, key=lambda x: x["score"])[:beam_size]

        # Calculate answers
        ground_truth_keys_list = [
            set([
                Chem.MolToInchiKey(Chem.MolFromSmiles(target)) for target in targets
            ]) for targets in test_set[epoch]["targets"]
        ]
        overall_result[:, 1] += 1
        depth_hit[1, test_set[epoch]["depth"], :] += 1
        for rank, answer in enumerate(final_answer_set):
            answer_keys = set(answer["answer_keys"])
            flag = False
            for ground_truth_keys in ground_truth_keys_list:
                if ground_truth_keys == answer_keys:
                    overall_result[rank:, 0] += 1
                    depth_hit[0, test_set[epoch]["depth"], rank:] += 1
                    flag = True
                    break
            if flag: break

    print("overall_result: ", overall_result, 100 * overall_result[:, 0] / overall_result[:, 1])
    print("depth_hit: ", depth_hit, 100 * depth_hit[0, :, :] / depth_hit[1, :, :])


if __name__ == '__main__':
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    stock = pd.read_hdf('data/zinc_stock_17_04_20.hdf5', key="table")  
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x for x in stockinchikey_list])

    run_with_redirection(argh.dispatch_command)(evaluate_megan)
