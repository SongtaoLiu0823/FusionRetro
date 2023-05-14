import argparse
import random
import json
import torch
import numpy as np
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


def get_reaction_cost(task):
    reaction, _input = task
    product_list, ground_truth_reactants = _input
    ground_truth_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in ground_truth_reactants.split('.')]) 
    for rank, solution in enumerate(get_beam(product_list, args.beam_size)):
        flag = False
        predict_reactants, cost = solution[0], solution[1]
        try:
            answer_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in predict_reactants])
        except:
            return reaction, np.inf
        if answer_keys == ground_truth_keys:
            return reaction, cost
        if flag: break
    return reaction, np.inf


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
    parser.add_argument('--beam_size', type=int, default=10, help='Beams size. Default 5. Must be 1 meaning greedy search or greater or equal 5.')
    parser.add_argument("--batch", help="batch", type=int, default=0)

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

    reaction_to_input = {}
    file_name = "train_canolize_dataset.json"
    with open(file_name, 'r') as f:
        dataset = json.load(f)
        for _, reaction_tree in dataset.items():
            retro_routes = reaction_tree['retro_routes']
            for retro_route in retro_routes:
                products, reactants = [], []
                for reaction in retro_route:
                    products.append(reaction.split('>')[0])
                    reactants.append(reaction.split('>')[-1])
                for i in range(len(products)):
                    reaction = products[i] + ">>" + reactants[i]
                    if reaction not in reaction_to_input.keys():
                        _input = (products[:i+1], reactants[i])
                        reaction_to_input[reaction] = _input

    tasks = []
    for reaction, _input in reaction_to_input.items():
        tasks.append((reaction, _input))
    
    reaction_cost = {}
    for task in tqdm(tasks):
        result = get_reaction_cost(task)
        reaction, cost = result
        if cost != np.inf:
            reaction_cost[reaction] = cost.item()
    with open('reaction_cost.json', 'w') as f:
        f.write(json.dumps(reaction_cost, indent=4))

