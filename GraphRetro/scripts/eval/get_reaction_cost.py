import numpy as np
import pandas as pd
import torch
import os
import json
import argparse
from tqdm import trange, tqdm
from rdkit import RDLogger, Chem
import yaml

from seq_graph_retro.utils.parse import get_reaction_info, extract_leaving_groups
from seq_graph_retro.utils.chem import apply_edits_to_mol
from seq_graph_retro.utils.edit_mol import canonicalize, generate_reac_set
from seq_graph_retro.models import EditLGSeparate
from seq_graph_retro.search import BeamSearch
from seq_graph_retro.molgraph import MultiElement
lg = RDLogger.logger()
lg.setLevel(4)

try:
    ROOT_DIR = os.environ["SEQ_GRAPH_RETRO"]
    DATA_DIR = os.path.join(ROOT_DIR, "datasets", "uspto-50k")
    EXP_DIR = os.path.join(ROOT_DIR, "experiments")

except KeyError:
    ROOT_DIR = "./"
    DATA_DIR = os.path.join(ROOT_DIR, "datasets", "uspto-50k")
    EXP_DIR = os.path.join(ROOT_DIR, "local_experiments")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_TEST_FILE = f"{DATA_DIR}/canonicalized_test.csv"


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


def canonicalize_prod(pcanon):
    pmol = Chem.MolFromSmiles(pcanon)
    [atom.SetAtomMapNum(atom.GetIdx()+1) for atom in pmol.GetAtoms()]
    p = Chem.MolToSmiles(pmol)
    return p


def load_edits_model(args):
    edits_step = args.edits_step
    if edits_step is None:
        edits_step = "best_model"

    if "run" in args.edits_exp:
        # This addition because some of the new experiments were run using wandb
        edits_loaded = torch.load(os.path.join(args.exp_dir, "wandb", args.edits_exp, "files", edits_step + ".pt"), map_location=DEVICE)
        with open(f"{args.exp_dir}/wandb/{args.edits_exp}/files/config.yaml", "r") as f:
            tmp_loaded = yaml.load(f, Loader=yaml.FullLoader)

        model_name = tmp_loaded['model']['value']

    else:
        edits_loaded = torch.load(os.path.join(args.exp_dir, args.edits_exp,
                                  "checkpoints", edits_step + ".pt"),
                                  map_location=DEVICE)
        model_name = args.edits_exp.split("_")[0]

    return edits_loaded, model_name


def load_lg_model(args):
    lg_step = args.lg_step
    if lg_step is None:
        lg_step = "best_model"

    if "run" in args.lg_exp:
        # This addition because some of the new experiments were run using wandb
        lg_loaded = torch.load(os.path.join(args.exp_dir, "wandb", args.lg_exp, "files", lg_step + ".pt"), map_location=DEVICE)
        with open(f"{args.exp_dir}/wandb/{args.lg_exp}/files/config.yaml", "r") as f:
            tmp_loaded = yaml.load(f, Loader=yaml.FullLoader)

        model_name = tmp_loaded['model']['value']

    else:
        lg_loaded = torch.load(os.path.join(args.exp_dir, args.lg_exp,
                               "checkpoints", lg_step + ".pt"),
                                map_location=DEVICE)
        model_name = args.lg_exp.split("_")[0]

    return lg_loaded, model_name


def get_dataset(phase):
    file_name = "%s/%s_dataset.json" % (DATA_DIR, phase)
    file_name = os.path.expanduser(file_name)
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


def get_prediction(p):
    p = canonicalize_prod(p)
    rxn_class = None
    try:
        answer = []
        if lg_toggles.get("use_rxn_class", False):
            top_k_nodes = beam_model.run_search(p, max_steps=6, rxn_class=rxn_class)
        else:
            top_k_nodes = beam_model.run_search(p, max_steps=6)

        for beam_idx, node in enumerate(top_k_nodes):
            
            if len(answer) == args.beam_size:
                return answer

            pred_edit = node.edit
            pred_label = node.lg_groups
            score = -node.prob

            if isinstance(pred_edit, list):
                pred_edit = pred_edit[0]
            try:
                pred_set = generate_reac_set(p, pred_edit, pred_label, verbose=False)
                num_valid_reactant = 0
                sms = set()
                for r in pred_set:
                    m = Chem.MolFromSmiles(r)
                    if m is not None:
                        num_valid_reactant += 1
                        sms.add(Chem.MolToSmiles(m))
                if num_valid_reactant != len(pred_set):
                    continue
                if len(sms):
                    answer.append([sorted(list(sms)), score])

            except BaseException as e:
                print(e, flush=True)
                pred_set = None

        return answer[:args.beam_size]
    
    except Exception as e:
        return []


def get_reaction_cost(task):
    product, ground_truth_reactants = task
    reaction = product + '>>' + ground_truth_reactants
    product = Chem.MolToSmiles(Chem.MolFromSmiles(product))
    _, product = cano_smiles(product)
    ground_truth_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in ground_truth_reactants.split('.')]) 
    for rank, solution in enumerate(get_prediction(product)):
        flag = False
        predict_reactants, cost = solution[0], solution[1]
        answer_keys = set([Chem.MolToInchiKey(Chem.MolFromSmiles(reactant))[:14] for reactant in predict_reactants])
        if answer_keys == ground_truth_keys:
            return reaction, cost
        if flag: break
    return reaction, np.inf


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default=DATA_DIR, help="Data directory")
parser.add_argument("--exp_dir", default=EXP_DIR, help="Experiments directory.")
parser.add_argument("--test_file", default=DEFAULT_TEST_FILE, help="Test file.")
parser.add_argument("--edits_exp", default="SingleEdit_21-03-2020--20-33-05",
                    help="Name of edit prediction experiment.")
parser.add_argument("--edits_step", default=None,
                    help="Checkpoint for edit prediction experiment.")
parser.add_argument("--lg_exp", default="LGClassifier_02-04-2020--02-06-17",
                    help="Name of synthon completion experiment.")
parser.add_argument("--lg_step", default=None,
                    help="Checkpoint for synthon completion experiment.")
parser.add_argument("--beam_width", default=20, type=int, help="Beam width")
parser.add_argument("--beam_size", default=10, type=int, help="Beam Size")
parser.add_argument("--use_rxn_class", action='store_true', help="Whether to use reaction class.")
parser.add_argument("--rxn_class_acc", action="store_true",
                    help="Whether to print reaction class accuracy.")

args = parser.parse_args()

test_df = pd.read_csv(args.test_file)

edits_loaded, edit_net_name = load_edits_model(args)
lg_loaded, lg_net_name = load_lg_model(args)

edits_config = edits_loaded["saveables"]
lg_config = lg_loaded['saveables']
lg_toggles = lg_config['toggles']

if 'tensor_file' in lg_config:
    if not os.path.isfile(lg_config['tensor_file']):
        if not lg_toggles.get("use_rxn_class", False):
            tensor_file = os.path.join(args.data_dir, "train/h_labels/without_rxn/lg_inputs.pt")
        else:
            tensor_file = os.path.join(args.data_dir, "train/h_labels/with_rxn/lg_inputs.pt")
        lg_config['tensor_file'] = tensor_file

rm = EditLGSeparate(edits_config=edits_config, lg_config=lg_config, edit_net_name=edit_net_name,
                    lg_net_name=lg_net_name, device=DEVICE)
rm.load_state_dict(edits_loaded['state'], lg_loaded['state'])
rm.to(DEVICE)
rm.eval()

beam_model = BeamSearch(model=rm, beam_width=args.beam_width, max_edits=1)


train_products_list, train_reactants_list = get_dataset('train')
tasks = []
for epoch in trange(0, len(train_products_list)):
    product = train_products_list[epoch]
    ground_truth_reactants = train_reactants_list[epoch]
    tasks.append((product, ground_truth_reactants))

reaction_cost = {}
for task in tqdm(tasks):
    result = get_reaction_cost(task)
    reaction, cost = result
    if cost != np.inf:
        reaction_cost[reaction] = cost
        
with open('reaction_cost.json', 'w') as f:
    f.write(json.dumps(reaction_cost, indent=4))

