import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import numpy as np
import json
import math
import argparse
import multiprocessing
from tqdm import trange, tqdm
from generate_retro_templates import process_an_example
from rdkit import DataStructs
from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(4)


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


def get_reaction_cost(task):
    product, ground_truth_reactants = task
    reaction = product + '>>' + ground_truth_reactants
    product = Chem.MolToSmiles(Chem.MolFromSmiles(product))
    _, product = cano_smiles(product)
    ground_truth_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in ground_truth_reactants.split('.')]) 
    for rank, solution in enumerate(do_one(product, beam_size)):
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
    beam_size = args.beam_size
    
    np.random.seed(42)

    similarity_metric = DataStructs.BulkTanimotoSimilarity # BulkDiceSimilarity or BulkTanimotoSimilarity
    similarity_label = 'Tanimoto'
    getfp = lambda smi: AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2, useFeatures=True)
    getfp_label = 'Morgan2Feat'

    train_products_list, train_reactants_list = get_dataset('train')
    train_product_fps = [getfp(smi) for smi in train_products_list]
    train_rxn_smiles = [train_reactants_list[i]+'>>'+train_products_list[i] for i in range(len(train_reactants_list))]
    jx_cache = {}

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

