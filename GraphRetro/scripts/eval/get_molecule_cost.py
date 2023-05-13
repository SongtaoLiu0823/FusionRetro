import numpy as np
import json
import networkx as nx
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem


def smiles_to_fp(s, fp_dim=2048, pack=False):
    mol = Chem.MolFromSmiles(s)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=fp_dim)
    onbits = list(fp.GetOnBits())
    arr = np.zeros(fp.GetNumBits(), dtype=np.bool)
    arr[onbits] = 1

    if pack:
        arr = np.packbits(arr)
    fp = 1 * np.array(arr)

    return fp


def smiles_to_inchikey(smi):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smi))[:14]


def get_tree_molecule_cost(G, inchikey):
    res = nx.descendants(G, inchikey)
    cost = 0
    for node in res:
        if ">.<" in node:
            cost += float(node.split(">.<")[-1])
    return cost


with open('reaction_cost.json', 'r') as f:
    reaction_cost = json.load(f)

molecule_cost = {}
file_name = "datasets/uspto-50k/train_dataset.json"

with open(file_name, 'r') as f:
    dataset = json.load(f)
    for _, reaction_trees in tqdm(dataset.items()):
        max_num_materials = 0
        retro_routes = None
        for i in range(1, int(reaction_trees['num_reaction_trees'])+1):
            if len(reaction_trees[str(i)]['materials']) > max_num_materials:
                max_num_materials = len(reaction_trees[str(i)]['materials'])
                retro_routes = reaction_trees[str(i)]['retro_routes']

        G = nx.DiGraph()
        reaction_set = set()
        molecule_set = set()
        for retro_route in retro_routes:
            for reaction in retro_route:
                if reaction not in reaction_set:
                    reaction_set.add(reaction)
                    product = reaction.split('>>')[0]
                    if smiles_to_inchikey(product) not in molecule_set:
                        molecule_set.add(smiles_to_inchikey(product))
                        G.add_node(smiles_to_inchikey(product))
                    for reactant in reaction.split('>>')[1].split('.'):
                        if smiles_to_inchikey(reactant) not in molecule_set:
                            molecule_set.add(smiles_to_inchikey(reactant))
                            G.add_node(smiles_to_inchikey(reactant))
        
        flag = False
        for reaction in reaction_set:
            if reaction not in reaction_cost.keys():
                flag = True
                break
            reaction_node = "%s>.<%s" %(reaction, reaction_cost[reaction])
            G.add_node(reaction_node)
            product_inchikey = smiles_to_inchikey(reaction.split('>>')[0])
            G.add_edge(product_inchikey, reaction_node)
            for reactant in reaction.split('>>')[1].split('.'):
                reactant_inchikey = smiles_to_inchikey(reactant)
                G.add_edge(reaction_node, reactant_inchikey)
        if flag:
            continue
        for reaction in reaction_set:
            product_inchikey = smiles_to_inchikey(reaction.split('>>')[0])
            cost = get_tree_molecule_cost(G, product_inchikey)
            if product_inchikey not in molecule_cost.keys():
                molecule_cost[product_inchikey] = cost
            else:
                if cost < molecule_cost[product_inchikey]:
                    molecule_cost[product_inchikey] = cost

molecule_visited = set()
retro_reaction_set = set()
final_fps = []
final_costs = []
with open(file_name, 'r') as f:
    dataset = json.load(f)
    for _, reaction_trees in tqdm(dataset.items()):
        max_num_materials = 0
        retro_routes_list = None
        for i in range(1, int(reaction_trees['num_reaction_trees'])+1):
            if len(reaction_trees[str(i)]['materials']) > max_num_materials:
                max_num_materials = len(reaction_trees[str(i)]['materials'])
                retro_routes_list = reaction_trees[str(i)]['retro_routes']

        for retro_route in retro_routes_list:
            for retro_reaction in retro_route:
                if retro_reaction not in retro_reaction_set:
                    retro_reaction_set.add(retro_reaction)
                    product_inchikey = smiles_to_inchikey(retro_reaction.split('>>')[0])
                    if product_inchikey not in molecule_visited and product_inchikey in molecule_cost.keys():
                        molecule_visited.add(product_inchikey)
                        final_fps.append(smiles_to_fp(retro_reaction.split('>>')[0]))
                        final_costs.append(molecule_cost[product_inchikey])

torch.save(torch.FloatTensor(final_fps), 'fps.pt')
torch.save(torch.FloatTensor(final_costs), 'costs.pt')
