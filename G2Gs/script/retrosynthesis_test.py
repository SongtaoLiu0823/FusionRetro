import os 
import sys
import numpy as np
import json
import argparse
import torch
import logging

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from g2g import dataset
from rdkit import Chem
from tqdm import trange, tqdm
from torchdrug import core, tasks, models, utils, data


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


def get_dataset(phase):
    file_name = "datasets/%s_dataset.json" % phase
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


def load_model(beam_size):
    # Center Identification
    reaction_dataset = dataset.USPTOFull("datasets", 
                atom_feature="center_identification", kekulize=True)
    reaction_train, reaction_valid, reaction_test = reaction_dataset.split()

    reaction_model = models.RGCN(input_dim=reaction_dataset.node_feature_dim,
                    hidden_dims=[512, 512, 512, 512],
                    # hidden_dims=[10, 10],
                    num_relation=reaction_dataset.num_bond_type,
                    concat_hidden=True)
    reaction_task = tasks.CenterIdentification(reaction_model,
                                            feature=("graph", "atom", "bond"))

    # Synthon Completion
    synthon_dataset = dataset.USPTOFull("datasets/", as_synthon=True,
                atom_feature="synthon_completion", kekulize=True)
    synthon_train, synthon_valid, synthon_test = synthon_dataset.split()

    synthon_model = models.RGCN(input_dim=synthon_dataset.node_feature_dim,
                            hidden_dims=[512, 512, 512, 512],
                            # hidden_dims=[10, 10],
                            num_relation=synthon_dataset.num_bond_type,
                            concat_hidden=True)
    synthon_task = tasks.SynthonCompletion(synthon_model, feature=("graph",))

    # Retrosynthesis
    reaction_task.preprocess(reaction_train, None, None)
    synthon_task.preprocess(synthon_train, None, None)
    task = tasks.Retrosynthesis(reaction_task, synthon_task, center_topk=beam_size,
                                num_synthon_beam=beam_size, max_prediction=beam_size)
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
    solver = core.Engine(task, reaction_train, reaction_valid, reaction_test,
                        optimizer, gpus=args.gpus, batch_size=32)
    solver.load("g2gs_reaction_model.pth", load_optimizer=False)
    solver.load("g2gs_synthon_model.pth", load_optimizer=False)

    return task, reaction_dataset


def get_batch(product_smiles, kwargs):
    batch = []
    for i, smiles in enumerate(product_smiles):
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if not mol:
            continue
        product = data.Molecule.from_molecule(mol, **kwargs)
        with product.node():
            product.node_label = torch.zeros(product.num_node)
            product.atom_map = torch.arange(product.num_node) + 1
        with product.edge():
            product.edge_label = torch.zeros(product.num_edge)
            product.bond_stereo[:] = 0
        batch.append({
            "graph": (product, product),    # Fake reactant
            "reaction": 0,
            "sample id": i,
        })

    batch = data.graph_collate(batch)
    if args.gpus:
        batch = utils.cuda(batch)
    return batch


def get_prediction(model, batch):
    reactants, num_prediction = model.predict(batch)
    num_prediction = num_prediction.cumsum(0)
    answer = [[]]
    for i, graph in enumerate(reactants):
        if i == num_prediction[len(answer)-1]: 
            answer.append([])
        _reactants = graph.connected_components()[0]
        answer[-1].append([
            sorted([reactant.to_smiles(isomeric=False, atom_map=False, canonical=True) for reactant in _reactants]),
            -graph.logps.item()
        ])
    assert len(answer) == num_prediction.shape[0]
    return answer


def get_prediction_result(task):
    product, ground_truth_reactants = task
    ground_truth_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in ground_truth_reactants.split('.')]) 
    _, product = cano_smiles(product)
    try:
        batch = get_batch([product], reaction_dataset.kwargs)
    except:
        return None
    for rank, solution in enumerate(get_prediction(model, batch)[0]):
        flag = False
        predict_reactants, _ = solution[0], solution[1]
        answer_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in predict_reactants])
        if answer_keys == ground_truth_keys:
            return rank
        if flag: break
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--beam_size", help="beam size", type=int, default=10)
    parser.add_argument("-b", "--batch_size", help="batch size", type=int, default=16)
    parser.add_argument("-g", "--gpus", help="device", default=None)

    args, unparsed = parser.parse_known_args()
    args.gpus = utils.literal_eval(args.gpus)
    beam_size = args.beam_size

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    logger = logging.getLogger()
    logger.disabled = True
    model, reaction_dataset = load_model(beam_size)

    overall_result = np.zeros((args.beam_size, 2))
    test_products_list, test_reactants_list = get_dataset('test')
    test_tasks = []
    for epoch in trange(0, len(test_products_list)):
        ground_truth_reactants = test_reactants_list[epoch]
        product = test_products_list[epoch]
        product = Chem.MolToSmiles(Chem.MolFromSmiles(product))
        _, product = cano_smiles(product)
        test_tasks.append((product, ground_truth_reactants))

    for task in tqdm(test_tasks):
        rank = get_prediction_result(task)
        overall_result[:, 1] += 1
        if rank is not None:
            overall_result[rank:, 0] += 1

    print("overall_result: ", overall_result, 100 * overall_result[:, 0] / overall_result[:, 1])
