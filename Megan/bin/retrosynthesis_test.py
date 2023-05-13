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


def get_dataset(phase):
    file_name = "data/%s_dataset.json" %phase
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


def evaluate_megan(save_path: str, beam_size: int = 10, max_gen_steps: int = 16, beam_batch_size: int = 1, n_max_atoms: int = 200):
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

    overall_result = np.zeros((beam_size, 2))
    test_products_list, test_reactants_list = get_dataset('test')
    for epoch in trange(0, len(test_products_list)):
        ground_truth_reactants = test_reactants_list[epoch]
        product = test_products_list[epoch]
        product = Chem.MolToSmiles(Chem.MolFromSmiles(product))
        _, product = cano_smiles(product)
        ground_truth_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in ground_truth_reactants.split('.')]) 
        overall_result[:, 1] += 1


        ## Get expansion result
        input_mols = []
        input_mol = Chem.MolFromSmiles(product)

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
        solution_list = []
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
                solution_list.append([sorted(list(sms)), -math.log10(score)]) 
                aim_size -= 1

        
        for rank, solution in enumerate(solution_list):
            flag = False
            predict_reactants, _ = solution[0], solution[1]
            try:
                answer_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in predict_reactants])
            except:
                continue
            if answer_keys == ground_truth_keys:
                overall_result[rank:, 0] += 1
                flag = True
            if flag: break

    print("overall_result: ", overall_result, 100 * overall_result[:, 0] / overall_result[:, 1])
       


if __name__ == '__main__':
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    run_with_redirection(argh.dispatch_command)(evaluate_megan)
