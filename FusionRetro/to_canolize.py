import json
import argparse
from tqdm import tqdm
from rdkit import Chem
from rdkit.rdBase import DisableLog

DisableLog('rdApp.warning')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="train", type=str, help="which dataset for canolize.")

args = parser.parse_args()

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


canolize_dataset = {}
file_name = "data/%s_dataset.json"%args.dataset
with open(file_name, 'r') as f:
    dataset = json.load(f)
    for final_product_smiles, reaction_trees in tqdm(dataset.items()):
        canolize_dataset[final_product_smiles] = {}
        max_num_materials = 0
        final_retro_tree = None
        for i in range(1, int(reaction_trees['num_reaction_trees'])+1):
            if len(reaction_trees[str(i)]['materials']) > max_num_materials:
                max_num_materials = len(reaction_trees[str(i)]['materials'])
                final_retro_tree = reaction_trees[str(i)]

        canolize_dataset[final_product_smiles]['depth'] = reaction_trees['depth']
        retro_routes = []
        for retro_route in final_retro_tree['retro_routes']:
            current_retro_route = []
            for retro_reaction in retro_route:
                product = retro_reaction.split('>')[0]
                reactants = retro_reaction.split('>')[-1]
                product_mol = Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles(product)))
                product = Chem.MolToSmiles(product_mol)
                _, product = cano_smiles(product)
                reactants_list = []
                for reactant in reactants.split('.'):
                    reactant_mol = Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles(reactant)))
                    reactant = Chem.MolToSmiles(reactant_mol)
                    _, reactant = cano_smiles(reactant)
                    reactants_list.append(reactant)
                reactants = '.'.join(reactants_list)
                current_retro_route.append(product+">>"+reactants)
            retro_routes.append(current_retro_route)
        canolize_dataset[final_product_smiles]['retro_routes'] = retro_routes
        canolize_dataset[final_product_smiles]['materials'] = final_retro_tree['materials']

with open('%s_canolize_dataset.json'%args.dataset, 'w') as f:
    f.write(json.dumps(canolize_dataset, indent=4))
