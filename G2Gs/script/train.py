import os
import sys
import logging
import argparse
import torch

from torchdrug import core, tasks, models, utils
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from g2g import dataset


logger = logging.getLogger(__file__)

def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)
parser.add_argument("-g", "--gpus", help="device", default=None)

args, unparsed = parser.parse_known_args()
args.gpus = utils.literal_eval(args.gpus)

if __name__ == "__main__":
    torch.manual_seed(args.seed + comm.get_rank())

    logger = get_root_logger()
    
    
    # Center Identification
    reaction_dataset = dataset.USPTOFull("datasets", 
                atom_feature="center_identification", kekulize=True)
    reaction_train, reaction_valid, reaction_test = reaction_dataset.split()
    if comm.get_rank() == 0:
        logger.warning(reaction_dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(reaction_train), len(reaction_valid), len(reaction_test)))
    
    reaction_model = models.RGCN(input_dim=reaction_dataset.node_feature_dim,
                    hidden_dims=[512, 512, 512, 512],
                    num_relation=reaction_dataset.num_bond_type,
                    concat_hidden=True)
    reaction_task = tasks.CenterIdentification(reaction_model,
                                           feature=("graph", "atom", "bond"))
    
    reaction_optimizer = torch.optim.Adam(reaction_task.parameters(), lr=1e-3)
    reaction_solver = core.Engine(reaction_task, reaction_train, reaction_valid,
                                reaction_test, reaction_optimizer,
                                gpus=args.gpus, batch_size=128)
    reaction_solver.train(num_epoch=100)
    reaction_solver.evaluate("valid")
    reaction_solver.save("g2gs_reaction_model.pth")

    # Synthon Completion
    synthon_dataset = dataset.USPTOFull("datasets/", as_synthon=True,
                atom_feature="synthon_completion", kekulize=True)
    synthon_train, synthon_valid, synthon_test = synthon_dataset.split()
    if comm.get_rank() == 0:
        logger.warning(synthon_dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(synthon_train), len(synthon_valid), len(synthon_test)))
    
    synthon_model = models.RGCN(input_dim=synthon_dataset.node_feature_dim,
                            hidden_dims=[512, 512, 512, 512],
                            num_relation=synthon_dataset.num_bond_type,
                            concat_hidden=True)
    synthon_task = tasks.SynthonCompletion(synthon_model, feature=("graph",))
    
    synthon_optimizer = torch.optim.Adam(synthon_task.parameters(), lr=1e-4)
    synthon_solver = core.Engine(synthon_task, synthon_train, synthon_valid,
                                synthon_test, synthon_optimizer,
                                gpus=args.gpus, batch_size=64)
    synthon_solver.train(num_epoch=100)
    synthon_solver.evaluate("valid")
    synthon_solver.save("g2gs_synthon_model.pth")

    
    # Retrosynthesis
    task = tasks.Retrosynthesis(reaction_task, synthon_task, center_topk=2,
                                num_synthon_beam=5, max_prediction=10)
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
    solver = core.Engine(task, reaction_train, reaction_valid, reaction_test,
                        optimizer, gpus=args.gpus, batch_size=32)
    solver.load("g2gs_reaction_model.pth", load_optimizer=False)
    solver.load("g2gs_synthon_model.pth", load_optimizer=False)
    solver.evaluate("valid")
