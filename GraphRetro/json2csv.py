import json
import os
import csv

def get_dataset(phase):
    file_name = os.path.join("datasets/uspto-50k", "%s_dataset.json" % phase)
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

if __name__ == "__main__":
    for phase in ['train', 'eval', 'test']:
        products_list, reactants_list = get_dataset(phase)
        rxn_smiles_set = set()
        f = open(os.path.join('datasets/uspto-50k', '%s.csv'%phase), 'w')
        writer = csv.writer(f)
        writer.writerow(['id', 'class', 'reactants>reagents>production'])
        for i in range(len(reactants_list)):
            rxn_smiles = reactants_list[i]+">>"+products_list[i]
            if rxn_smiles not in rxn_smiles_set:
                rxn_smiles_set.add(rxn_smiles)
                writer.writerow(['UNK', 1, rxn_smiles])
