import os
import numpy as np
import torch
import math
import json
import argparse

from rdkit import Chem
from tqdm import tqdm, trange
from gln.common.cmd_args import cmd_args
from gln.common.consts import DEVICE
from gln.test.model_inference import RetroGLN


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


def get_reaction_cost(task):
    product, ground_truth_reactants = task
    reaction = product + '>>' + ground_truth_reactants
    ground_truth_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant)) for reactant in ground_truth_reactants.split('.')]) 
    for rank, solution in enumerate(get_inference_answer(product, local_args.beam_size)):
        flag = False
        predict_reactants, cost = solution[0], solution[1]
        answer_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant)) for reactant in predict_reactants])
        if answer_keys == ground_truth_keys:
            return reaction, cost
        if flag: break
    return reaction, np.inf


if __name__ == "__main__":
    cmd_opt = argparse.ArgumentParser(description='Argparser for get reaction cost')
    cmd_opt.add_argument('-epoch_for_search', default=100, type=int, help='model for search')
    cmd_opt.add_argument("-beam_size", help="beam size", type=int, default=10)
    local_args, _ = cmd_opt.parse_known_args()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    model_dump = os.path.join(cmd_args.save_dir, 'model-%d.dump' % local_args.epoch_for_search)
    model = RetroGLN(model_dump)
    model.gln.to(DEVICE)

    train_products_list, train_reactants_list = get_dataset('train')
    tasks = []
    for epoch in trange(0, len(train_products_list)):
        product = train_products_list[epoch]
        ground_truth_reactants = train_reactants_list[epoch]
        tasks.append((product, ground_truth_reactants))

    reaction_cost = {}
    for task in tqdm(tasks):
        result = get_reaction_cost(task)
        reaction, cost = result
        if cost != np.inf:
            reaction_cost[reaction] = cost

    with open('reaction_cost.json', 'w') as f:
        f.write(json.dumps(reaction_cost, indent=4))
