import numpy as np
import torch
import random
import json
import argparse

from tqdm import trange, tqdm
from rdkit import Chem
from rdkit.rdBase import DisableLog
from preprocess import get_vocab_size, get_char_to_ix, get_ix_to_char, get_dataset
from modeling import TransformerConfig, Transformer, get_products_mask, get_reactants_mask, get_mutual_mask


DisableLog('rdApp.warning')


def convert_symbols_to_inputs(products_list, reactants_list, max_length):
    num_samples = len(products_list)
    #products
    products_input_ids = torch.zeros((num_samples, max_length), device=device, dtype=torch.long)
    products_input_mask = torch.zeros((num_samples, max_length), device=device)

    #reactants
    reactants_input_ids = torch.zeros((num_samples, max_length), device=device, dtype=torch.long)
    reactants_input_mask = torch.zeros((num_samples, max_length), device=device)

    for cnt in range(num_samples):
        products = '^' + products_list[cnt] + '$'
        reactants = '^' + reactants_list[cnt] + '$'
        
        for i, symbol in enumerate(products):
            products_input_ids[cnt, i] = char_to_ix[symbol]
        products_input_mask[cnt, :len(products)] = 1

        for i in range(len(reactants)-1):
            reactants_input_ids[cnt, i] = char_to_ix[reactants[i]]
        reactants_input_mask[cnt, :len(reactants)-1] = 1
    return (products_input_ids, products_input_mask, reactants_input_ids, reactants_input_mask)


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


def get_output_probs(product, res):
    test_products_ids, test_products_mask, test_res_ids, test_res_mask = convert_symbols_to_inputs([product], [res], args.max_length)
    # To Tensor
    test_mutual_mask = get_mutual_mask([test_res_mask, test_products_mask])
    test_products_mask = get_products_mask(test_products_mask)
    test_res_mask = get_reactants_mask(test_res_mask)

    logits = predict_model(test_products_ids, test_res_ids, test_products_mask, test_res_mask, test_mutual_mask)
    prob = logits[0, len(res), :] / args.temperature
    prob = torch.exp(prob) / torch.sum(torch.exp(prob))
    return prob.detach()


def get_beam(product, beam_size):
    lines = []
    scores = []
    final_beams = []
    object_size = beam_size

    for i in range(object_size):
        lines.append("")
        scores.append(0.0)

    for step in range(args.max_length):
        if step == 0:
            prob = get_output_probs(product, "")
            result = torch.zeros((vocab_size, 2), device=device)
            for i in range(vocab_size):
                result[i, 0] = -torch.log10(prob[i])
                result[i, 1] = i
        else:
            num_candidate = len(lines)
            result = torch.zeros((num_candidate * vocab_size, 2), device=device)
            for i in range(num_candidate):
                prob = get_output_probs(product, lines[i])
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
    product, ground_truth_reactants = task
    reaction = product + '>>' + ground_truth_reactants
    product = Chem.MolToSmiles(Chem.MolFromSmiles(product))
    _, product = cano_smiles(product)
    ground_truth_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant)) for reactant in ground_truth_reactants.split('.')]) 
    for rank, solution in enumerate(get_beam(product, args.beam_size)):
        flag = False
        predict_reactants, cost = solution[0], solution[1]
        answer_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant)) for reactant in predict_reactants])
        if answer_keys == ground_truth_keys:
            return reaction, cost
        if flag: break
    return reaction, np.inf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--max_length', type=int, default=200, help='The max length of a molecule.')
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


    train_products_list, train_reactants_list = get_dataset('train')
    tasks = []
    for epoch in trange(0, len(train_products_list)):
        product = train_products_list[epoch]
        ground_truth_reactants = train_reactants_list[epoch]
        tasks.append((product, ground_truth_reactants))

    reaction_cost = {}
    for epoch in trange(args.batch*3000, min(args.batch*3000 + 3000, len(tasks))):
    #for task in tqdm(tasks):
        result = get_reaction_cost(tasks[epoch])
        reaction, cost = result
        if cost != np.inf:
            reaction_cost[reaction] = cost.item()
    with open('cost/reaction_cost_%s.json'%args.batch, 'w') as f:
        f.write(json.dumps(reaction_cost, indent=4))

