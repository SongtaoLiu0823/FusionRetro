import numpy as np
import torch
import json
import os
import pandas as pd
import math
import rdkit.Chem.AllChem as AllChem
import argparse
import multiprocessing

from copy import deepcopy
from rdchiral.main import rdchiralReaction, rdchiralReactants, rdchiralRun
from tqdm import tqdm
from rdkit import Chem, DataStructs
from generate_retro_templates import process_an_example


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


def do_one(prod_smiles, beam_size):
    global jx_cache

    
    ex = Chem.MolFromSmiles(prod_smiles)
    rct = rdchiralReactants(prod_smiles)
    fp = getfp(prod_smiles)
    
    sims = similarity_metric(fp, [fp_ for fp_ in train_product_fps])
    js = np.argsort(sims)[::-1]

    # Get probability of precursors
    probs = {}
    
    for j in js[:100]:
        jx = j
        
        if jx in jx_cache:
            (template, rcts_ref_fp) = jx_cache[jx]
        else:
            try:
                template = '(' + process_an_example(train_rxn_smiles[jx], super_general=True).replace('>>', ')>>')
            except:
                return []
            rcts_ref_fp = getfp(train_rxn_smiles[jx].split('>')[0])
            jx_cache[jx] = (template, rcts_ref_fp)

        try:    
            rxn = rdchiralReaction(template)
        except:
            return []

        try:
            outcomes = rdchiralRun(rxn, rct, combine_enantiomers=False)
        except Exception as e:
            print(e)
            outcomes = []
            
        for precursors in outcomes:
            precursors_fp = getfp(precursors)
            precursors_sim = similarity_metric(precursors_fp, [rcts_ref_fp])[0]
            if precursors in probs:
                probs[precursors] = max(probs[precursors], precursors_sim * sims[j])
            else:
                probs[precursors] = precursors_sim * sims[j]
        
    testlimit = 50
    mols = []
    legends = []
    score = []
    found_rank = 9999
    for r, (prec, prob) in enumerate(sorted(probs.items(), key=lambda x:x[1], reverse=True)[:testlimit]):
        mols.append(Chem.MolFromSmiles(prec))
        legends.append('overall score: {:.3f}'.format(prob))
        score.append(-math.log10(prob))
    
    answer = []
    aim_size = beam_size
    for i in range(len(mols)):
        if aim_size == 0:
            break
        answer.append([sorted(Chem.MolToSmiles(mols[i], True).split(".")), score[i]])
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
    answer_list = []
    queue = {
        "routes_info": [{"route": [task["product"]], "depth": 0}],  # List of routes information
        "starting_materials": [],
    }
    while True:
        if len(queue) == 0:
            break
        nxt_queue = {}
        routes_info = queue["routes_info"]
        starting_materials = queue["starting_materials"]
        first_route_info = routes_info[0]
        first_route, depth = first_route_info["route"], first_route_info["depth"]
        if depth > max_depth:
            break
        if len(do_one(first_route[-1], args.beam_size)) == 0:
            break
        expansion_solution = do_one(first_route[-1], args.beam_size)[0]
        iter_routes = deepcopy(routes_info)
        iter_routes.pop(0)
        iter_starting_materials = deepcopy(starting_materials)
        expansion_reactants, _ = expansion_solution[0], expansion_solution[1]
        expansion_reactants = sorted(expansion_reactants)
        if check_reactants_are_material(expansion_reactants) and len(iter_routes) == 0:
            answer_list = iter_starting_materials+expansion_reactants
        else:
            for reactant in expansion_reactants:
                if check_reactant_is_material(reactant):
                    iter_starting_materials.append(reactant)
                else:
                    iter_routes = [{"route": first_route+[reactant], "depth": depth+1}] + iter_routes
            nxt_queue = {
                "routes_info": iter_routes,
                "starting_materials": iter_starting_materials
            }
        queue = nxt_queue
            
    answer_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(m))[:14] for m in answer_list])

    # Calculate answers
    ground_truth_keys_list = [
        set([
            Chem.MolToInchiKey(Chem.MolFromSmiles(target))[:14] for target in targets
        ]) for targets in task["targets"]
    ]

    for ground_truth_keys in ground_truth_keys_list:
        if ground_truth_keys == answer_keys:
            return max_depth, True

    return max_depth, False


if __name__ == "__main__":
    parser = argparse.ArgumentParser("dfs_bfs_search.py")
    parser.add_argument("--beam_size", help="beam size", type=int, default=5)
    parser.add_argument("--num_cores", help="number of cores", type=int, default=64)
    args = parser.parse_args()
    device = torch.device("cpu")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity
    similarity_label = 'Tanimoto'
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=True)
    getfp_label = 'Morgan2Feat'

    train_products_list, train_reactants_list = get_dataset('train')
    train_product_fps = [getfp(smi) for smi in train_products_list]
    train_rxn_smiles = [train_reactants_list[i]+'>>'+train_products_list[i] for i in range(len(train_reactants_list))]
    jx_cache = {}

    stock = pd.read_hdf('zinc_stock_17_04_20.hdf5', key="table")  
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    overall_result = np.zeros((2))
    depth_hit = np.zeros((2, 15))
    tasks = load_dataset("test")
    pool = multiprocessing.Pool(args.num_cores)
    for result in tqdm(pool.imap_unordered(get_route_result, tasks), total=len(tasks)):
        max_depth, match = result
        overall_result[1] += 1
        depth_hit[1, max_depth] += 1
        if match:
            overall_result[0] += 1
            depth_hit[0, max_depth] += 1

    print("overall_result: ", overall_result, 100 * overall_result[0] / overall_result[1])
    print("depth_hit: ", depth_hit, 100 * depth_hit[0, :] / depth_hit[1, :])
