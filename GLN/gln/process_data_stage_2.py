import random
import os
import sys
import json
import pickle as cp
import numpy as np
import csv
import multiprocessing

from tqdm import tqdm
from rdkit import Chem
from collections import Counter, defaultdict
from gln.common.cmd_args import cmd_args
from gln.common.mol_utils import cano_smarts, smarts_has_useless_parentheses
from gln.mods.rdchiral.template_extractor import extract_from_reaction
from gln.data_process.data_info import DataInfo, load_train_reactions
from gln.common.reactor import Reactor
from gln.mods.mol_gnn.mol_utils import SmartsMols, SmilesMols


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


'''build raw template'''
def get_writer(fname, header):
    output_name = os.path.join(cmd_args.save_dir, fname)
    fout = open(output_name, 'w')
    writer = csv.writer(fout)
    writer.writerow(header)
    return fout, writer

def get_tpl(task):
    idx, rxn_smiles = task
    react, reagent, prod = rxn_smiles.split('>')
    reaction = {'_id': idx, 'reactants': react, 'products': prod}
    template = extract_from_reaction(reaction)
    return idx, template



'''get cannonical smarts'''
def process_centers():
    prod_cano_smarts = set()
    react_cano_smarts = set()

    smarts_cano_map = {}
    pbar = tqdm(retro_templates)
    for template in pbar:
        sm_prod, _, sm_react = template.split('>')
        if smarts_has_useless_parentheses(sm_prod):
            sm_prod = sm_prod[1:-1]
        
        smarts_cano_map[sm_prod] = cano_smarts(sm_prod)[1]
        prod_cano_smarts.add(smarts_cano_map[sm_prod])

        for r_smarts in sm_react.split('.'):            
            smarts_cano_map[r_smarts] = cano_smarts(r_smarts)[1]
            react_cano_smarts.add(smarts_cano_map[r_smarts])
        pbar.set_description('# prod centers: %d, # react centers: %d' % (len(prod_cano_smarts), len(react_cano_smarts)))
    print('# prod centers: %d, # react centers: %d' % (len(prod_cano_smarts), len(react_cano_smarts)))

    with open(os.path.join(cmd_args.save_dir, 'prod_cano_smarts.txt'), 'w') as f:
        for s in prod_cano_smarts:
            f.write('%s\n' % s)
    with open(os.path.join(cmd_args.save_dir, 'react_cano_smarts.txt'), 'w') as f:
        for s in react_cano_smarts:
            f.write('%s\n' % s)
    with open(os.path.join(cmd_args.save_dir, 'cano_smarts.pkl'), 'wb') as f:
        cp.dump(smarts_cano_map, f, cp.HIGHEST_PROTOCOL)



'''find centers'''
def find_edges(task):
    idx, rxn_type, smiles = task
    smiles = smiles_cano_map[smiles]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return idx, rxn_type, smiles, None
    list_centers = []
    for i, (sm_center, center_mol) in enumerate(prod_center_mols):
        if center_mol is None:
            continue
        if not rxn_type in smarts_type_set[sm_center]:
            continue
        if mol.HasSubstructMatch(center_mol):
            list_centers.append(str(i))
    if len(list_centers) == 0:
        return idx, rxn_type, smiles, None
    centers = ' '.join(list_centers)
    return idx, rxn_type, smiles, centers



'''build all reactions'''
def find_tpls(cur_task):
    idx, (rxn_type, rxn) = cur_task
    reactants, _, raw_prod = rxn.split('>')

    prod = DataInfo.get_cano_smiles(raw_prod)

    if not (rxn_type, prod) in DataInfo.prod_center_maps:
        return None
    reactants = DataInfo.get_cano_smiles(reactants)
    prod_center_cand_idx = DataInfo.prod_center_maps[(rxn_type, prod)]
    
    neg_reactants = set()
    pos_tpl_idx = {}
    tot_tpls = 0
    for center_idx in prod_center_cand_idx:
        c = DataInfo.prod_cano_smarts[center_idx]
        assert c in DataInfo.unique_tpl_of_prod_center

        tpl_indices = DataInfo.unique_tpl_of_prod_center[c][rxn_type]
        tot_tpls += len(tpl_indices)
        for tpl_idx in tpl_indices:
            cur_t, tpl = DataInfo.unique_templates[tpl_idx]
            assert cur_t == rxn_type
            pred_mols = Reactor.run_reaction(prod, tpl)
            if pred_mols is None or len(pred_mols) == 0:
                continue            
            for pred in pred_mols:
                if pred != reactants:
                    neg_reactants.add(pred)
                else:
                    pos_tpl_idx[tpl_idx] = (len(tpl_indices), len(pred_mols))
    return (idx, pos_tpl_idx, neg_reactants)


def get_writer_build_all_reactions(fname, header):
    f = open(os.path.join(cmd_args.save_dir, 'np-%d' % cmd_args.num_parts, fname), 'w')
    writer = csv.writer(f)
    writer.writerow(header) 
    return f, writer



##Start processing data


'''build raw template'''
if not os.path.exists(os.path.join(cmd_args.save_dir, 'proc_train_singleprod.csv')):
    rows = []
    products_list, reactants_list = get_dataset('train')
    for i in range(len(reactants_list)):
        rows.append(reactants_list[i]+">>"+products_list[i])

    pool = multiprocessing.Pool(cmd_args.num_cores)
    tasks = []
    for idx, row in tqdm(enumerate(rows)):
        tasks.append((idx, row))

    fout, writer = get_writer('proc_train_singleprod.csv', ['id', 'class', 'rxn_smiles', 'retro_templates'])
    fout_failed, failed_writer = get_writer('failed_template.csv', ['id', 'class', 'rxn_smiles', 'err_msg'])

    for result in tqdm(pool.imap(get_tpl, tasks), total=len(tasks)):
        idx, template = result
        rxn_smiles = rows[idx]

        if 'reaction_smarts' in template:
            writer.writerow([idx, "UNK", rxn_smiles, template['reaction_smarts']])
            fout.flush()
        else:
            failed_writer.writerow([idx, "UNK", rxn_smiles, template['err_msg']])
            fout_failed.flush()

    fout.close()
    fout_failed.close()



'''filter template'''
if not os.path.exists(os.path.join(cmd_args.save_dir, 'templates.csv')):
    proc_file = os.path.join(cmd_args.save_dir, 'proc_train_singleprod.csv')

    unique_tpls = Counter()
    tpl_types = defaultdict(set)
    with open(proc_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        print(header)
        for row in tqdm(reader):
            tpl = row[header.index('retro_templates')]
            rxn_type = row[header.index('class')]
            tpl_types[tpl].add(rxn_type)
            unique_tpls[tpl] += 1

    print('total # templates', len(unique_tpls))

    used_tpls = []
    for x in unique_tpls:
        if unique_tpls[x] >= cmd_args.tpl_min_cnt:
            used_tpls.append(x)
    print('num templates after filtering', len(used_tpls))

    out_file = os.path.join(cmd_args.save_dir, 'templates.csv')
    with open(out_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['class', 'retro_templates'])
        for x in used_tpls:
            for t in tpl_types[x]:
                writer.writerow([t, x])



'''get canonical smarts'''
if not os.path.exists(os.path.join(cmd_args.save_dir, 'cano_smarts.pkl')):
    tpl_file = os.path.join(cmd_args.save_dir, 'templates.csv')

    retro_templates = []
    
    with open(tpl_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in tqdm(reader):
            retro_templates.append(row[header.index('retro_templates')])

    rxn_smiles = []
    rxn_smiles_set = set()
    for phase in ['train', 'valid', 'test']:
        products_list, reactants_list = get_dataset(phase)
        for i in range(len(reactants_list)):
            if reactants_list[i]+">>"+products_list[i] not in rxn_smiles_set:
                rxn_smiles.append(reactants_list[i]+">>"+products_list[i])
                rxn_smiles_set.add(reactants_list[i]+">>"+products_list[i])

    process_centers()



'''find centers'''
if cmd_args.num_parts <= 0:
    num_parts = cmd_args.num_cores
else:
    num_parts = cmd_args.num_parts

if not os.path.exists(os.path.join(os.path.join(cmd_args.save_dir, 'np-%d' % num_parts), '%s-prod_center_maps-part-%d.csv' % ('test', num_parts-1))):
    with open(os.path.join(cmd_args.save_dir, 'cano_smiles.pkl'), 'rb') as f:
        smiles_cano_map = cp.load(f)

    with open(os.path.join(cmd_args.save_dir, 'cano_smarts.pkl'), 'rb') as f:
        smarts_cano_map = cp.load(f)

    with open(os.path.join(cmd_args.save_dir, 'prod_cano_smarts.txt'), 'r') as f:
        prod_cano_smarts = [row.strip() for row in f.readlines()]

    prod_center_mols = []
    for sm in tqdm(prod_cano_smarts):
        prod_center_mols.append((sm, Chem.MolFromSmarts(sm)))

    print('num of prod centers', len(prod_center_mols))
    print('num of smiles', len(smiles_cano_map))

    csv_file = os.path.join(cmd_args.save_dir, 'templates.csv')

    smarts_type_set = defaultdict(set)
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        tpl_idx = header.index('retro_templates')
        type_idx = header.index('class')
        for row in reader:
            rxn_type = row[type_idx]
            template = row[tpl_idx]
            sm_prod, _, _ = template.split('>')
            if smarts_has_useless_parentheses(sm_prod):
                sm_prod = sm_prod[1:-1]            
            sm_prod = smarts_cano_map[sm_prod]
            smarts_type_set[sm_prod].add(rxn_type)

    if cmd_args.num_parts <= 0:
        num_parts = cmd_args.num_cores
    else:
        num_parts = cmd_args.num_parts

    pool = multiprocessing.Pool(cmd_args.num_cores)

    for out_phase in ['train', 'valid', 'test']:
        rxn_smiles = []
        products_list, reactants_list = get_dataset(out_phase)
        for i in range(len(reactants_list)):
            rxn_smiles.append(("UNK", reactants_list[i]+">>"+products_list[i]))

        part_size = min(len(rxn_smiles) // num_parts + 1, len(rxn_smiles))

        for pid in range(num_parts):        
            idx_range = range(pid * part_size, min((pid + 1) * part_size, len(rxn_smiles)))

            local_results = [None] * len(idx_range)

            tasks = []            
            for i, idx in enumerate(idx_range):
                rxn_type, rxn = rxn_smiles[idx]
                reactants, _, prod = rxn.split('>')
                tasks.append((i, rxn_type, prod))                
            for result in tqdm(pool.imap(find_edges, tasks), total=len(tasks)):
                i, rxn_type, smiles, centers = result
                local_results[i] = (rxn_type, smiles, centers)
            out_folder = os.path.join(cmd_args.save_dir, 'np-%d' % num_parts)
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)
            fout = open(os.path.join(out_folder, '%s-prod_center_maps-part-%d.csv' % (out_phase, pid)), 'w')
            writer = csv.writer(fout)
            writer.writerow(['smiles', 'class', 'centers'])

            for i in range(len(local_results)):
                rxn_type, smiles, centers = local_results[i]
                if centers is not None:
                    writer.writerow([smiles, rxn_type, centers])
            fout.close()



'''build all reactions'''
if not os.path.exists(os.path.join(cmd_args.save_dir, 'np-%d' % cmd_args.num_parts, 'pos_tpls-part-0.csv')):
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    DataInfo.init(cmd_args)
    
    fn_pos = lambda idx: get_writer_build_all_reactions('pos_tpls-part-%d.csv' % idx, ['tpl_idx', 'pos_tpl_idx', 'num_tpl_compete', 'num_react_compete'])
    fn_neg = lambda idx: get_writer_build_all_reactions('neg_reacts-part-%d.csv' % idx, ['sample_idx', 'neg_reactants'])

    if cmd_args.num_parts <= 0:
        num_parts = cmd_args.num_cores
        DataInfo.load_cooked_part('train', load_graphs=False)
    else:
        num_parts = cmd_args.num_parts

    train_reactions = load_train_reactions(cmd_args)
    n_train = len(train_reactions)
    part_size = n_train // num_parts + 1

    if cmd_args.part_num > 0:
        prange = range(cmd_args.part_id, cmd_args.part_id + cmd_args.part_num)
    else:
        prange = range(num_parts)
    for pid in prange:
        f_pos, writer_pos = fn_pos(pid)
        f_neg, writer_neg = fn_neg(pid)
        if cmd_args.num_parts > 0:
            DataInfo.load_cooked_part('train', part=pid, load_graphs=False)
        part_tasks = []
        idx_range = list(range(pid * part_size, min((pid + 1) * part_size, n_train)))
        for i in idx_range:
            part_tasks.append((i, train_reactions[i]))

        pool = multiprocessing.Pool(cmd_args.num_cores)
        for result in tqdm(pool.imap(find_tpls, part_tasks), total=len(idx_range)):
            if result is None:
                continue
            idx, pos_tpl_idx, neg_reactions = result
            idx = str(idx)
            neg_keys = neg_reactions
            
            if cmd_args.max_neg_reacts > 0:
                neg_keys = list(neg_keys)
                random.shuffle(neg_keys)
                neg_keys = neg_keys[:cmd_args.max_neg_reacts]
            for pred in neg_keys:
                writer_neg.writerow([idx, pred])
            for key in pos_tpl_idx:
                nt, np = pos_tpl_idx[key]
                writer_pos.writerow([idx, key, nt, np])
            f_pos.flush()
            f_neg.flush()
        f_pos.close()
        f_neg.close()
        pool.close()
        pool.join()



'''dump graphs'''
if not os.path.exists(os.path.join(cmd_args.save_dir, 'graph_smarts')):
    file_root = cmd_args.save_dir
    if cmd_args.fp_degree > 0:
        SmilesMols.set_fp_degree(cmd_args.fp_degree)
        SmartsMols.set_fp_degree(cmd_args.fp_degree)

    if cmd_args.retro_during_train:
        part_folder = os.path.join(file_root, 'np-%d' % cmd_args.num_parts)
        if cmd_args.part_num > 0:
            prange = range(cmd_args.part_id, cmd_args.part_id + cmd_args.part_num)
        else:
            prange = range(cmd_args.num_parts)
        for pid in prange:
            with open(os.path.join(part_folder, 'neg_reacts-part-%d.csv' % pid), 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in tqdm(reader):
                    reacts = row[-1]
                    for t in reacts.split('.'):
                        SmilesMols.get_mol_graph(t)
                    SmilesMols.get_mol_graph(reacts)
            SmilesMols.save_dump(os.path.join(part_folder, 'neg_graphs-part-%d' % pid))
            SmilesMols.clear()
        sys.exit()

    with open(os.path.join(file_root, 'cano_smiles.pkl'), 'rb') as f:
        smiles_cano_map = cp.load(f)

    with open(os.path.join(file_root, 'prod_cano_smarts.txt'), 'r') as f:
        prod_cano_smarts = [row.strip() for row in f.readlines()]

    with open(os.path.join(file_root, 'react_cano_smarts.txt'), 'r') as f:
        react_cano_smarts = [row.strip() for row in f.readlines()]


    for mol in tqdm(smiles_cano_map):
        SmilesMols.get_mol_graph(smiles_cano_map[mol])
    SmilesMols.save_dump(os.path.join(cmd_args.save_dir, 'graph_smiles'))

    for smarts in tqdm(prod_cano_smarts + react_cano_smarts):
        SmartsMols.get_mol_graph(smarts)
    SmartsMols.save_dump(os.path.join(cmd_args.save_dir, 'graph_smarts'))
