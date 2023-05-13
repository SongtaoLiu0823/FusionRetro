import os
import json
import logging
from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem

from torch.utils import data as torch_data
from torchdrug import data, datasets, utils
from torchdrug.core import Registry as R

logger = logging.getLogger(__name__)


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


@R.register("datasets.USPTOFull")
class USPTOFull(datasets.USPTO50k):
    """
    Chemical reactions extracted from USPTO patents.

    Statistics:
        - #Reaction: 
        - #Reaction class: 1

    Parameters:
        path (str): path to store the dataset
        as_synthon (bool, optional): whether decompose (reactant, product) pairs into (reactant, synthon) pairs
        verbose (int, optional): output verbose level
        **kwargs
    """

    target_fields = ["reaction"]

    reaction_names = ["Placeholder"]


    def __init__(self, path, as_synthon=False, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.as_synthon = as_synthon
        self.kwargs = kwargs

        self.data = []
        self.targets = defaultdict(list)
        self.num_samples = []
        for split in ["train", "valid", "test"]:
            file_name = 'datasets/%s_dataset.json'%split
            self.load_json(file_name, verbose=verbose, **kwargs)
            self.num_samples.append(len(self.data))

        if as_synthon:
            prefix = "Computing synthons"
            process_fn = self._get_synthon
        else:
            prefix = "Computing reaction centers"
            process_fn = self._get_reaction_center

        data = self.data
        targets = self.targets
        num_samples = self.num_samples
        idx = 0
        self.data = []
        self.targets = defaultdict(list)
        self.num_samples = []
        indexes = range(len(data))
        if verbose:
            indexes = tqdm(indexes, prefix)
        invalid = 0
        for i in indexes:
            if i >= num_samples[idx]: 
                self.num_samples.append(len(self.data))
                idx += 1
            reactant, product = data[i]
            reactant.bond_stereo[:] = 0
            product.bond_stereo[:] = 0

            reactants, products = process_fn(reactant, product)
            if not reactants:
                invalid += 1
                continue

            self.data += zip(reactants, products)
            for k in targets:
                self.targets[k] += [targets[k][i]] * len(reactants)
            self.targets["sample id"] += [i] * len(reactants)
        self.num_samples.append(len(self.data))

        self.valid_rate = 1 - invalid / len(data)

    def load_json(self, json_file, verbose=0, **kwargs):
        with open(json_file, "r") as fin:
            dataset = json.load(fin)
            if verbose:
                reader = iter(tqdm(dataset.items(), "Loading %s" % json_file))
            else:
                reader = dataset.items()
            smiles = []
            targets = defaultdict(list)
            retro_reaction_set = set()
            for _, reaction_trees in reader:
                target = reaction_trees['1']['retro_routes'][0][0].split('>')[0]
                target = Chem.MolToSmiles(Chem.MolFromSmiles(target))
                _, target = cano_smiles(target)
                for k in range(1, int(reaction_trees['num_reaction_trees'])+1):
                    retro_routes_list = reaction_trees[str(k)]['retro_routes']
                    for retro_route in retro_routes_list:
                        for retro_reaction in retro_route:
                            if retro_reaction not in retro_reaction_set:
                                retro_reaction_set.add(retro_reaction)
                                smiles.append(retro_reaction)
                                targets["target"].append(target)
                                targets["reaction"].append(0)   # Fake target                        

        self.load_smiles(smiles, targets, verbose=verbose, **kwargs)

    def load_smiles(self, smiles_list, targets, transform=None, verbose=0, **kwargs):
        """
        Load the dataset from SMILES and targets.

        Parameters:
            smiles_list (list of str): SMILES strings
            targets (dict of list): prediction targets
            transform (Callable, optional): data transformation function
            verbose (int, optional): output verbose level
            **kwargs
        """
        num_sample = len(smiles_list)
        for field, target_list in targets.items():
            if len(target_list) != num_sample:
                raise ValueError("Number of target `%s` doesn't match with number of molecules. "
                                 "Expect %d but found %d" % (field, num_sample, len(target_list)))

        if verbose:
            smiles_list = tqdm(smiles_list, "Constructing molecules from SMILES")
        for i, smiles in enumerate(smiles_list):
            smiles_product, agent, smiles_reactant = smiles.split(">")
            mols = []
            for _smiles in [smiles_reactant, smiles_product]:
                mol = Chem.MolFromSmiles(_smiles)
                if not mol:
                    logger.debug("Can't construct molecule from SMILES `%s`. Ignore this sample." % _smiles)
                    break
                mol = data.Molecule.from_molecule(mol, **kwargs)
                mols.append(mol)
            else:
                self.data.append(mols)
                for field in targets:
                    self.targets[field].append(targets[field][i])
        self.transform = transform

    def split(self):
        offset = 0
        splits = []
        for num_sample in self.num_samples:
            split = torch_data.Subset(self, range(offset, num_sample))
            splits.append(split)
            offset = num_sample
        return splits
        
    @property
    def num_reaction_type(self):
        return len(self.reaction_types)

    @utils.cached_property
    def reaction_types(self):
        """All reaction types."""
        return sorted(set(self.target["class"]))
