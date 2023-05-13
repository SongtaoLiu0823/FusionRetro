import numpy as np
import json

chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$"
vocab_size = len(chars)

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
max_lenth = 200


def get_chars():
    return chars

def get_vocab_size():
    return vocab_size

def get_char_to_ix():
    return char_to_ix

def get_ix_to_char():
    return ix_to_char

def get_dataset(phase):
    file_name = "%s_canolize_dataset.json" %phase
    products_list = []
    reactants_list = []
    retro_reaction_set = set()
    with open(file_name, 'r') as f:
        dataset = json.load(f)
        for _, reaction_tree in dataset.items():
            retro_routes_list = reaction_tree['retro_routes']
            for retro_route in retro_routes_list:
                for retro_reaction in retro_route:
                    if retro_reaction not in retro_reaction_set:
                        retro_reaction_set.add(retro_reaction)
                        products_list.append(retro_reaction.split('>>')[0])
                        reactants_list.append(retro_reaction.split('>>')[1])
    return products_list, reactants_list

def convert_symbols_to_inputs(products_list, reactants_list, max_length):
    num_samples = len(products_list)
    #products
    products_input_ids = np.zeros((num_samples, max_length))
    products_input_mask = np.zeros((num_samples, max_length))

    #reactants
    reactants_input_ids = np.zeros((num_samples, max_length))
    reactants_input_mask = np.zeros((num_samples, max_length))

    #for output
    label_ids = np.zeros((num_samples, max_length))

    for cnt in range(num_samples):
        products = '^' + products_list[cnt] + '$'
        reactants = '^' + reactants_list[cnt] + '$'
        
        for i, symbol in enumerate(products):
            products_input_ids[cnt, i] = char_to_ix[symbol]
        products_input_mask[cnt, :len(products)] = 1

        for i in range(len(reactants)-1):
            reactants_input_ids[cnt, i] = char_to_ix[reactants[i]]
            label_ids[cnt, i] = char_to_ix[reactants[i+1]]
        reactants_input_mask[cnt, :len(reactants)-1] = 1
    return (products_input_ids, products_input_mask, reactants_input_ids, reactants_input_mask, label_ids)
