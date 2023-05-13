import csv
import pickle 
import argparse
import os
import numpy as np
import scipy
import multiprocessing
import json

from functools import partial
from pathlib import Path
from scipy import sparse
from tqdm import tqdm

from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdchiral.template_extractor import extract_from_reaction

sparse_fp = scipy.sparse.csr_matrix


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


def valid_template_to_get_reaction(args, id_reactants_products):
    idx, react, prod = id_reactants_products[0], id_reactants_products[1], id_reactants_products[2]
    reaction = {'_id': idx, 'reactants': react, 'products': prod}
    template = extract_from_reaction(reaction)
    # https://github.com/connorcoley/rdchiral/blob/master/rdchiral/template_extractor.py
    if 'reaction_smarts' in template:
        return react+'>>'+prod
    else:
        return None



'''' Generate fingerprints and non-map SMILES '''

def mol_smi_to_count_fp(mol_smi: str, radius: int = 2, fp_size: int = 32681, dtype: str = "int32") -> scipy.sparse.csr_matrix:
    fp_gen = GetMorganGenerator(radius=radius, useCountSimulation=True, includeChirality=True, fpSize=fp_size)
    mol = Chem.MolFromSmiles(mol_smi)
    uint_count_fp = fp_gen.GetCountFingerprint(mol)
    count_fp = np.empty((1, fp_size), dtype=dtype)
    DataStructs.ConvertToNumpyArray(uint_count_fp, count_fp)
    return sparse.csr_matrix(count_fp, dtype=dtype)

def gen_prod_fps_helper(args, rxn_smi):
    prod_smi_map = rxn_smi.split('>>')[-1]
    prod_mol = Chem.MolFromSmiles(prod_smi_map)
    [atom.ClearProp('molAtomMapNumber') for atom in prod_mol.GetAtoms()]
    prod_smi_nomap = Chem.MolToSmiles(prod_mol, True)
    # Sometimes stereochem takes another canonicalization... (just in case)
    prod_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(prod_smi_nomap), True)
    
    prod_fp = mol_smi_to_count_fp(prod_smi_nomap, args.radius, args.fp_size)
    return prod_smi_nomap, prod_fp

def gen_prod_fps(args):
    # parallelizing makes it very slow for some reason
    for phase in ['train', 'valid', 'test']:
        
        num_cores = len(dir(os))
        pool = multiprocessing.Pool(num_cores)
        valid_template_to_get_reaction_partial_ = partial(valid_template_to_get_reaction, args)
        products_list, reactants_list = get_dataset(phase)
        id_reactants_products_list = []
        for i in range(len(reactants_list)):
            id_reactants_products_list.append([i, reactants_list[i], products_list[i]])
        rxn_smiles = []
        for result in tqdm(pool.imap(valid_template_to_get_reaction_partial_, id_reactants_products_list), total=len(id_reactants_products_list), desc='Getting rxn_smi'):
            if result is not None:
                rxn_smiles.append(result)
    
        num_cores = len(dir(os))
        pool = multiprocessing.Pool(num_cores)

        phase_prod_smi_nomap = []
        phase_rxn_prod_fps = []
        gen_prod_fps_partial = partial(gen_prod_fps_helper, args)
        for result in tqdm(pool.imap(gen_prod_fps_partial, rxn_smiles), total=len(rxn_smiles), desc='Processing rxn_smi'):
            prod_smi_nomap, prod_fp = result
            phase_prod_smi_nomap.append(prod_smi_nomap)
            phase_rxn_prod_fps.append(prod_fp)

        # these are the input data into the network
        phase_rxn_prod_fps = sparse.vstack(phase_rxn_prod_fps)
        sparse.save_npz(f"{args.data_folder}/{args.output_file_prefix}_prod_fps_{phase}.npz", phase_rxn_prod_fps)

        with open(f"{args.data_folder}/{args.output_file_prefix}_to_{args.final_fp_size}_prod_smis_nomap_{phase}.smi", 'wb') as f:
            pickle.dump(phase_prod_smi_nomap, f, protocol=4)



'''Dimensionality reduction for fingerprints'''

def log_row(row):
    return sparse.csr_matrix(np.log(row.toarray() + 1))

def var_col(col):
    return np.var(col.toarray())

def variance_cutoff(args):
    for phase in ['train', 'valid', 'test']:
        prod_fps = sparse.load_npz(f"{args.data_folder}/{args.output_file_prefix}_prod_fps_{phase}.npz")

        num_cores = len(dir(os))
        pool = multiprocessing.Pool(num_cores)

        logged = []
        # imap is much, much faster than map
        # take log(x+1), ~2.5 min for 1mil-dim on 8 cores (parallelized)
        for result in tqdm(pool.imap(log_row, prod_fps), total=prod_fps.shape[0], desc='Taking log(x+1)'):
            logged.append(result)
        logged = sparse.vstack(logged)

        # collect variance statistics by column index from training product fingerprints
        # VERY slow with 2 for-loops to access each element individually.
        # idea: tranpose the sparse matrix, then go through 1 million rows using pool.imap 
        # massive speed up from 280 hours to 1 hour on 8 cores
        logged = logged.transpose()     # [39713, 1 mil] -> [1 mil, 39713]

        if phase == 'train':
            # no need to store all the values from each col_idx (results in OOM). just calc variance immediately, and move on
            vars = []
            # imap directly on csr_matrix is the fastest!!! from 1 hour --> ~2 min on 20 cores (parallelized)
            for result in tqdm(pool.imap(var_col, logged), total=logged.shape[0], desc='Collecting fingerprint values by indices'):
                vars.append(result)
            indices_ordered = list(range(logged.shape[0])) # should be 1,000,000
            indices_ordered.sort(key=lambda x: vars[x], reverse=True)

        # need to save sorted indices for infer_one API
        indices_np = np.array(indices_ordered[:args.final_fp_size])
        np.savetxt(f"{args.data_folder}/variance_indices.txt", indices_np)

        logged = logged.transpose() # [1 mil, 39713] -> [39713, 1 mil]
        # build and save final thresholded fingerprints
        thresholded = []
        for row_idx in tqdm(range(logged.shape[0]), desc='Building thresholded fingerprints'):
            thresholded.append(logged[row_idx, indices_ordered[:args.final_fp_size]]) # should be 32,681
        thresholded = sparse.vstack(thresholded)
        sparse.save_npz(f"{args.data_folder}/{args.output_file_prefix}_to_{args.final_fp_size}_prod_fps_{phase}.npz", thresholded)
        


'''Get templates'''

def get_tpl(task):
    idx, react, prod = task
    reaction = {'_id': idx, 'reactants': react, 'products': prod}
    template = extract_from_reaction(reaction)
    # https://github.com/connorcoley/rdchiral/blob/master/rdchiral/template_extractor.py
    return idx, template

def cano_smarts(smarts):
    tmp = Chem.MolFromSmarts(smarts)
    if tmp is None:
        return smarts
    # do not remove atom map number
    # [a.ClearProp('molAtomMapNumber') for a in tmp.GetAtoms()]
    cano = Chem.MolToSmarts(tmp)
    if '[[se]]' in cano:  # strange parse error
        cano = smarts
    return cano

def get_train_templates(args):
    '''
    For the expansion rules, a more general rule definition was employed. Here, only
    the reaction centre was extracted. Rules occurring at least three times
    were kept. The two sets encompass 17,134 and 301,671 rules, and cover
    52% and 79% of all chemical reactions from 2015 and after, respectively.
    '''
    phase = 'train'
    
    num_cores = len(dir(os))
    pool = multiprocessing.Pool(num_cores)
    valid_template_to_get_reaction_partial_ = partial(valid_template_to_get_reaction, args)
    products_list, reactants_list = get_dataset(phase)
    id_reactants_products_list = []
    for i in range(len(reactants_list)):
        id_reactants_products_list.append([i, reactants_list[i], products_list[i]])
    rxn_smiles = []
    for result in tqdm(pool.imap(valid_template_to_get_reaction_partial_, id_reactants_products_list), total=len(id_reactants_products_list), desc='Getting rxn_smi'):
        if result is not None:
            rxn_smiles.append(result)

    templates = {}
    rxns = []
    for idx, rxn_smi in enumerate(rxn_smiles):
        r = rxn_smi.split('>>')[0]
        p = rxn_smi.split('>>')[-1]
        rxns.append((idx, r, p))

    num_cores = len(dir(os))
    pool = multiprocessing.Pool(num_cores)
    invalid_temp = 0
    # here the order doesn't matter since we just want a dictionary of templates
    for result in tqdm(pool.imap_unordered(get_tpl, rxns), total=len(rxns)):
        idx, template = result
        if 'reaction_smarts' not in template:
            invalid_temp += 1
            continue # no template could be extracted

        # canonicalize template (needed, bcos q a number of templates are equivalent, 10247 --> 10198)
        p_temp = cano_smarts(template['products'])
        r_temp = cano_smarts(template['reactants'])
        cano_temp = p_temp + '>>' + r_temp
        # NOTE: 'reaction_smarts' is actually: p_temp >> r_temp !!!!! 

        if cano_temp not in templates:
            templates[cano_temp] = 1
        else:
            templates[cano_temp] += 1

    templates = sorted(templates.items(), key=lambda x: x[1], reverse=True)
    templates = ['{}: {}\n'.format(p[0], p[1]) for p in templates]
    with open(f"{args.data_folder}/{args.templates_file}", 'w') as f:
        f.writelines(templates)



'''Get data'''

def get_template_idx(temps_dict, task):
    rxn_idx, r, p = task    
    ############################################################
    # original label generation pipeline
    # extract template for this rxn_smi, and match it to template dictionary from training data
    rxn = (rxn_idx, r, p) # r & p must be atom-mapped
    rxn_idx, rxn_template = get_tpl(task)

    if 'reaction_smarts' not in rxn_template:
        return rxn_idx, -1 # unable to extract template
    p_temp = cano_smarts(rxn_template['products'])
    r_temp = cano_smarts(rxn_template['reactants'])
    cano_temp = p_temp + '>>' + r_temp

    if cano_temp in temps_dict:
        return rxn_idx, temps_dict[cano_temp]
    else:
        return rxn_idx, len(temps_dict) # no template matching
    
def match_templates(args):
    with open(f"{args.data_folder}/{args.templates_file}", 'r') as f:
        lines = f.readlines()
    temps_filtered = []
    temps_dict = {} # build mapping from temp to idx for O(1) find
    temps_idx = 0
    for l in lines:
        pa, cnt = l.strip().split(': ')
        if int(cnt) >= args.min_freq:
            temps_filtered.append(pa)
            temps_dict[pa] = temps_idx
            temps_idx += 1

    for phase in ['train', 'valid', 'test']:
        with open(f"{args.data_folder}/{args.output_file_prefix}_prod_smis_nomap_{phase}.smi", 'rb') as f:
            phase_prod_smi_nomap = pickle.load(f)
            
        num_cores = len(dir(os))
        pool = multiprocessing.Pool(num_cores)
        valid_template_to_get_reaction_partial_ = partial(valid_template_to_get_reaction, args)
        products_list, reactants_list = get_dataset(phase)
        id_reactants_products_list = []
        for i in range(len(reactants_list)):
            id_reactants_products_list.append([i, reactants_list[i], products_list[i]])
        rxn_smiles = []
        for result in tqdm(pool.imap(valid_template_to_get_reaction_partial_, id_reactants_products_list), total=len(id_reactants_products_list), desc='Getting rxn_smi'):
            if result is not None:
                rxn_smiles.append(result)
        
        tasks = []
        for idx, rxn_smi in tqdm(enumerate(rxn_smiles), desc='Building tasks', total=len(rxn_smiles)):
            r = rxn_smi.split('>>')[0]
            p = rxn_smi.split('>>')[1]
            tasks.append((idx, r, p))

        num_cores = len(dir(os))
        pool = multiprocessing.Pool(num_cores)

        # make CSV file to save labels (template_idx) & rxn data for monitoring training
        col_names = ['rxn_idx', 'prod_smi', 'rcts_smi', 'template', 'temp_idx']
        rows = []
        labels = []
        found = 0
        get_template_partial = partial(get_template_idx, temps_dict)
        # don't use imap_unordered!!!! it doesn't guarantee the order, or we can use it and then sort by rxn_idx
        for result in tqdm(pool.imap(get_template_partial, tasks), total=len(tasks)):
            rxn_idx, template_idx = result
            rcts_smi_map = rxn_smiles[rxn_idx].split('>>')[0]
            rcts_mol = Chem.MolFromSmiles(rcts_smi_map)
            [atom.ClearProp('molAtomMapNumber') for atom in rcts_mol.GetAtoms()]
            rcts_smi_nomap = Chem.MolToSmiles(rcts_mol, True)
            # Sometimes stereochem takes another canonicalization...
            rcts_smi_nomap = Chem.MolToSmiles(Chem.MolFromSmiles(rcts_smi_nomap), True)

            template = temps_filtered[template_idx] if template_idx != len(temps_filtered) else ''
            if template_idx != -1:
                rows.append([
                    rxn_idx,
                    phase_prod_smi_nomap[rxn_idx],
                    rcts_smi_nomap, # tasks[rxn_idx][1],
                    template, 
                    template_idx,
                ])
                labels.append(template_idx)
                found += (template_idx != len(temps_filtered))
        
        labels = np.array(labels)
        np.save(f"{args.data_folder}/{args.output_file_prefix}_labels_{phase}", labels)
        with open(f"{args.data_folder}/{args.output_file_prefix}_csv_{phase}.csv", 'w') as out_csv:
            writer = csv.writer(out_csv)
            writer.writerow(col_names) # header
            for row in rows:
                writer.writerow(row)


parser = argparse.ArgumentParser("prepare_data.py")
# file names
parser.add_argument("--data_folder", help="Path to data folder (do not change)", type=str, default=None) 
parser.add_argument("--rxnsmi_file_prefix", help="Prefix of the 3 pickle files containing the train/valid/test reaction SMILES strings (do not change)", type=str, default='clean_rxnsmi_noreagent_allmapped_canon') 
parser.add_argument("--output_file_prefix", help="Prefix of output files", type=str)
parser.add_argument("--templates_file", help="Filename of templates extracted from training data", type=str, default='training_templates')
parser.add_argument("--min_freq", help="Minimum frequency of template in training data to be retained", type=int, default=1)
parser.add_argument("--radius", help="Fingerprint radius", type=int, default=2)
parser.add_argument("--fp_size", help="Fingerprint size", type=int, default=1000000)
parser.add_argument("--final_fp_size", help="Fingerprint size", type=int, default=32681)


args = parser.parse_args()


if args.data_folder is None:
    args.data_folder = Path(__file__).resolve().parents[0] / 'data'
else:
    args.data_folder = Path(args.data_folder)

if args.output_file_prefix is None:
    args.output_file_prefix = f"{args.fp_size}dim_{args.radius}rad"

if not os.path.exists(f"{args.data_folder}/{args.output_file_prefix}_prod_fps_test.npz"):
    # ~2 min on 40k train prod_smi on 16 cores for 32681-dim
    gen_prod_fps(args)
if not os.path.exists(f"{args.data_folder}/{args.output_file_prefix}_to_{args.final_fp_size}_prod_fps_test.npz"):
    # for training dataset (40k rxn_smi):
    # ~1 min to do log(x+1) transformation on 16 cores, and then
    # ~2 min to gather variance statistics across 1 million indices on 16 cores, and then
    # ~5 min to build final 32681-dim fingerprint on 16 cores
    variance_cutoff(args)

args.output_file_prefix = f"{args.output_file_prefix}_to_{args.final_fp_size}"
if not os.path.exists(f"{args.data_folder}/{args.templates_file}"):
    # ~40 sec on 40k train rxn_smi on 16 cores
    get_train_templates(args)
if not os.path.exists(f"{args.data_folder}/{args.output_file_prefix}_csv_test.csv"):
    # ~3-4 min on 40k train rxn_smi on 16 cores
    match_templates(args)
