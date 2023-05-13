import argparse
import os
import random
import json
import torch
import numpy as np
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
from preprocess import get_vocab_size, get_char_to_ix, get_ix_to_char
from modeling import TransformerConfig, Transformer, get_padding_mask, get_mutual_mask, get_tril_mask, get_mem_tril_mask
from rdkit import Chem
from rdkit.rdBase import DisableLog

DisableLog('rdApp.warning')


def convert_symbols_to_inputs(products_list, reactants_list, max_depth, max_length):
    #products
    products_input = torch.zeros((len(products_list), max_depth, max_length), device=device, dtype=torch.long)
    products_input_mask = torch.zeros((len(products_list), max_depth, max_length), device=device)

    #reactants
    reactants_input = torch.zeros((len(products_list), max_depth, max_length), device=device, dtype=torch.long)
    reactants_input_mask = torch.zeros((len(products_list), max_depth, max_length), device=device)

    #memory
    memory_input_mask = torch.zeros((len(products_list), max_depth), device=device)
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
            reactants_input_mask[index, i, :len(reactant)-1] = 1
    return (products_input, products_input_mask, reactants_input, reactants_input_mask, memory_input_mask)


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


def get_output_probs(products, res, max_depth, max_length):
    products_ids, products_mask, reactants_ids, reactants_mask, memory_mask = convert_symbols_to_inputs([products], [res], max_depth, max_length)
    mutual_mask = get_mutual_mask([reactants_mask, products_mask])
    products_mask = get_padding_mask(products_mask)
    reactants_mask = get_tril_mask(reactants_mask)
    memory_mask = get_mem_tril_mask(memory_mask)
        
    logits = predict_model(products_ids, reactants_ids, products_mask, reactants_mask, mutual_mask, memory_mask)
    k = len(products) - 1
    prob = logits[0, k, len(res[k]), :] / args.temperature
    prob = torch.exp(prob) / torch.sum(torch.exp(prob))
    return prob.detach()


def get_beam(products, beam_size):
    lines = []
    scores = []
    final_beams = []
    object_size = beam_size

    res = ['' for _ in range(len(products))]
    for i in range(object_size):
        lines.append("")
        scores.append(0.0)
    
    for step in range(args.max_length):
        if step == 0:
            prob = get_output_probs(products, res, len(products), args.max_length)
            result = torch.zeros((vocab_size, 2), device=device)
            for i in range(vocab_size):
                result[i, 0] = -torch.log10(prob[i])
                result[i, 1] = i
        else:
            num_candidate = len(lines)
            result = torch.zeros((num_candidate * vocab_size, 2), device=device)
            for i in range(num_candidate):
                prob = get_output_probs(products, res[:-1]+[lines[i]], len(products), args.max_length)
                for j in range(vocab_size):
                    result[i*vocab_size+j, 0] = -torch.log10(prob[j]) + scores[i]
                    result[i*vocab_size+j, 1] = i * 100 + j
        
        ranked_result = result[result[:, 0].argsort()]
        
        new_beams = []
        new_scores = []
        if len(lines) == 0: 
            break

        for i in range(object_size):
            symbol = ix_to_char[ranked_result[i, 1].item()%100]
            beam_index = int(ranked_result[i, 1]) // 100 

            if symbol == '$':
                added = lines[beam_index] + symbol
                if added != "$":
                    final_beams.append([lines[beam_index] + symbol, ranked_result[i,0]])
                object_size -= 1
            else:
                new_beams.append(lines[beam_index] + symbol)
                new_scores.append(ranked_result[i, 0])

        lines = new_beams
        scores = new_scores

        if len(lines) == 0:
            break

    for i in range(len(final_beams)):
        final_beams[i][1] = final_beams[i][1] / len(final_beams[i][0])

    final_beams = list(sorted(final_beams, key=lambda x:x[1]))
    answer = []
    aim_size = beam_size
    for k in range(len(final_beams)):
        if aim_size == 0:
            break
        reactants = set(final_beams[k][0].split("."))
        num_valid_reactant = 0
        sms = set()
        for r in reactants:
            r = r.replace("$", "")
            m = Chem.MolFromSmiles(r)
            if m is not None:
                num_valid_reactant += 1
                sms.add(Chem.MolToSmiles(m))
        if num_valid_reactant != len(reactants):
            continue
        if len(sms):
            answer.append([sorted(list(sms)), final_beams[k][1]])
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
            product_mol = Chem.MolFromInchi(Chem.MolToInchi(Chem.MolFromSmiles(product)))
            product = Chem.MolToSmiles(product_mol)
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
        if len(get_beam(first_route, args.beam_size)) == 0:
            break
        expansion_solution = get_beam(first_route, args.beam_size)[0]
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max_length', type=int, default=200, help='The max length of a molecule.')
    parser.add_argument('--max_depth', type=int, default=14, help='The max depth of a synthesis route.')
    parser.add_argument('--embedding_size', type=int, default=64, help='The size of embeddings')
    parser.add_argument('--hidden_size', type=int, default=640, help='The size of hidden units')
    parser.add_argument('--num_hidden_layers', type=int, default=3, help='Number of layers in encoder\'s module. Default 3.')
    parser.add_argument('--num_attention_heads', type=int, default=10, help='Number of attention heads. Default 10.')
    parser.add_argument('--intermediate_size', type=int, default=512, help='The size of hidden units of position-wise layer.')
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--temperature', type=float, default=1.2, help='Temperature for decoding. Default 1.2')
    parser.add_argument('--beam_size', type=int, default=5, help='Beams size. Default 5. Must be 1 meaning greedy search or greater or equal 5.')

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = TransformerConfig(vocab_size=get_vocab_size(),
                            max_length=args.max_length,
                            embedding_size=args.embedding_size,
                            hidden_size=args.hidden_size,
                            num_hidden_layers=args.num_hidden_layers,
                            num_attention_heads=args.num_attention_heads,
                            intermediate_size=args.intermediate_size,
                            hidden_dropout_prob=args.hidden_dropout_prob)

    predict_model = Transformer(config)
    checkpoint = torch.load("models/model.pkl")
    if isinstance(checkpoint, torch.nn.DataParallel):
        checkpoint = checkpoint.module
    predict_model.load_state_dict(checkpoint.state_dict())

    predict_model.to(device)
    predict_model.eval()

    char_to_ix = get_char_to_ix()
    ix_to_char = get_ix_to_char()
    vocab_size = get_vocab_size()

    stock = pd.read_hdf('zinc_stock_17_04_20.hdf5', key="table")  
    stockinchikey_list = stock.inchi_key.values
    stock_inchikeys = set([x[:14] for x in stockinchikey_list])

    tasks = load_dataset('test')
    overall_result = np.zeros((2))
    depth_hit = np.zeros((2, 15))
    for epoch in tqdm(range(0, len(tasks))):
        max_depth, match = get_route_result(tasks[epoch])
        overall_result[1] += 1
        depth_hit[1, max_depth] += 1
        if match:
            overall_result[0] += 1
            depth_hit[0, max_depth] += 1

    print("overall_result: ", overall_result, 100 * overall_result[0] / overall_result[1])
    print("depth_hit: ", depth_hit, 100 * depth_hit[0, :] / depth_hit[1, :])

