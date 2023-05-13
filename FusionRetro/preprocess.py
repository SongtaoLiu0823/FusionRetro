import numpy as np
import json
from collections import defaultdict

chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
vocab_size = len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }


def get_chars():
    return chars

def get_vocab_size():
    return vocab_size

def get_char_to_ix():
    return char_to_ix

def get_ix_to_char():
    return ix_to_char

def get_train_dataset():
    file_name = "train_canolize_dataset.json"
    depth_products_list = defaultdict(list)
    depth_reactants_list = defaultdict(list)
    with open(file_name, 'r') as f:
        dataset = json.load(f)
        for _, reaction_tree in dataset.items():
            retro_routes = reaction_tree['retro_routes']
            depth = reaction_tree['depth']
            for retro_route in retro_routes:
                products, reactants = [], []
                for reaction in retro_route:
                    products.append(reaction.split('>')[0])
                    reactants.append(reaction.split('>')[-1])
                depth_products_list[depth].append(products)
                depth_reactants_list[depth].append(reactants)

    return depth_products_list, depth_reactants_list


def convert_symbols_to_inputs(products_list, reactants_list, max_depth, max_length):
    #products
    products_input = np.zeros((len(products_list), max_depth, max_length))
    products_input_mask = np.zeros((len(products_list), max_depth, max_length))

    #reactants
    reactants_input = np.zeros((len(products_list), max_depth, max_length))
    reactants_input_mask = np.zeros((len(products_list), max_depth, max_length))

    #for output
    label_input = np.zeros((len(products_list), max_depth, max_length))

    #memory
    memory_input_mask = np.zeros((len(products_list), max_depth))
    for index, products in enumerate(products_list):
        reactants = reactants_list[index]
        memory_input_mask[index, :len(products)] = 1
        for i, product in enumerate(products):
            reactant = reactants[i]
            product = '^' + product + '$'
            reactant = '^' + reactant + '$'
        
            for j, symbol in enumerate(product):
                products_input[index, i, j] = char_to_ix[symbol]
            products_input_mask[index, i, :len(product)] = 1

            for j in range(len(reactant)-1):
                reactants_input[index, i, j] = char_to_ix[reactant[j]]
                label_input[index, i, j] = char_to_ix[reactant[j+1]]
            reactants_input_mask[index, i, :len(reactant)-1] = 1
    return (products_input, products_input_mask, reactants_input, reactants_input_mask, memory_input_mask, label_input)
