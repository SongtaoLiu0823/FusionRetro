import os
import json
import pickle as cp

from tqdm import tqdm
from gln.common.cmd_args import cmd_args
from gln.common.mol_utils import cano_smiles


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


'''get_canonical_smiles.py'''
def process_smiles():
    all_symbols = set()

    smiles_cano_map = {}
    for rxn in tqdm(rxn_smiles):
        reactants, _, prod = rxn.split('>')
        mols = reactants.split('.') + [prod]
        for sm in mols:
            m, cano_sm = cano_smiles(sm)
            if m is not None:
                for a in m.GetAtoms():
                    all_symbols.add((a.GetAtomicNum(), a.GetSymbol()))
            if sm in smiles_cano_map:
                assert smiles_cano_map[sm] == cano_sm
            else:
                smiles_cano_map[sm] = cano_sm
    print('num of smiles', len(smiles_cano_map))
    set_mols = set()
    for s in smiles_cano_map:
        set_mols.add(smiles_cano_map[s])
    print('# unique smiles', len(set_mols))    
    with open(os.path.join(cmd_args.save_dir, 'cano_smiles.pkl'), 'wb') as f:
        cp.dump(smiles_cano_map, f, cp.HIGHEST_PROTOCOL)
    print('# unique atoms:', len(all_symbols))
    all_symbols = sorted(list(all_symbols))
    with open(os.path.join(cmd_args.save_dir, 'atom_list.txt'), 'w') as f:
        for a in all_symbols:
            f.write('%d\n' % a[0])



##Start processing data

'''get_canonical_smiles.py'''
if not os.path.exists(os.path.join(cmd_args.save_dir, 'atom_list.txt')):
    rxn_smiles = []
    rxn_smiles_set = set()
    for phase in ['train', 'valid', 'test']:
        products_list, reactants_list = get_dataset(phase)
        for i in range(len(reactants_list)):
            if reactants_list[i]+">>"+products_list[i] not in rxn_smiles_set:
                rxn_smiles.append(reactants_list[i]+">>"+products_list[i])
                rxn_smiles_set.add(reactants_list[i]+">>"+products_list[i])
    process_smiles()
