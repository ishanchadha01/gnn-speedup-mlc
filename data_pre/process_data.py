import os
import random
import numpy as np
import torch
from rdkit import Chem, rdBase
import pickle
from mordred import Calculator, descriptors
import warnings
from tqdm import tqdm
from multiprocessing import Pool
import json
from itertools import islice

from utils import add_mol

rdBase.DisableLog("rdApp.error")
rdBase.DisableLog("rdApp.warning")
warnings.filterwarnings("ignore")


def preprocess_mol(mol):
    try:
        Chem.SanitizeMol(mol)
        si = Chem.FindPotentialStereo(mol)
        for element in si:
            if (str(element.type) == "Atom_Tetrahedral" and str(element.specified) == "Specified"):
                mol.GetAtomWithIdx(element.centeredOn).SetProp("Chirality", str(element.descriptor))
            elif (str(element.type) == "Bond_Double" and str(element.specified) == "Specified"):
                mol.GetBondWithIdx(element.centeredOn).SetProp("Stereochemistry", str(element.descriptor))
        
        assert "." not in Chem.MolToSmiles(mol)
    except:
        return None
    
    mol = Chem.RemoveHs(mol)
    return mol


def preprocess(molsuppl, graph_save_path="../../data_1m", num_processes=2):
    length = len(molsuppl)

    mol_dict = {
        "n_node": [],
        "n_edge": [],
        "node_attr": [],
        "edge_attr": [],
        "src": [],
        "dst": [],
    }

    print("Creating molecule dict")
    

    # for i, mol in tqdm(enumerate(molsuppl)):
 
    #     try:
    #         Chem.SanitizeMol(mol)
    #         si = Chem.FindPotentialStereo(mol)
    #         for element in si:
    #             if (
    #                 str(element.type) == "Atom_Tetrahedral"
    #                 and str(element.specified) == "Specified"
    #             ):
    #                 mol.GetAtomWithIdx(element.centeredOn).SetProp(
    #                     "Chirality", str(element.descriptor)
    #                 )
    #             elif (
    #                 str(element.type) == "Bond_Double"
    #                 and str(element.specified) == "Specified"
    #             ):
    #                 mol.GetBondWithIdx(element.centeredOn).SetProp(
    #                     "Stereochemistry", str(element.descriptor)
    #                 )
    #         assert "." not in Chem.MolToSmiles(mol)
    #     except:
    #         continue

    #     mol = Chem.RemoveHs(mol)
    #     mol_dict = add_mol(mol_dict, mol)

    with Pool(num_processes) as pool:
        results = list(tqdm(pool.imap(preprocess_mol, molsuppl), total=len(molsuppl)))

    for mol in tqdm(filter(None, results)):  # Filter out None results (failed molecules)
        mol_dict = add_mol(mol_dict, mol)

    mol_dict["n_node"] = np.array(mol_dict["n_node"]).astype(int)
    mol_dict["n_edge"] = np.array(mol_dict["n_edge"]).astype(int)
    mol_dict["node_attr"] = np.vstack(mol_dict["node_attr"]).astype(bool)
    mol_dict["edge_attr"] = np.vstack(mol_dict["edge_attr"]).astype(bool)
    mol_dict["src"] = np.hstack(mol_dict["src"]).astype(int)
    mol_dict["dst"] = np.hstack(mol_dict["dst"]).astype(int)

    for key in mol_dict.keys():
        print(key, mol_dict[key].shape, mol_dict[key].dtype)

    with open(os.path.join(graph_save_path, "pubchem_graph.npz"), "wb") as f:
        pickle.dump([mol_dict], f, protocol=5)


def get_mordred_mol(mol):
    try:
        Chem.SanitizeMol(mol)
        si = Chem.FindPotentialStereo(mol)
        for element in si:
            if (
                str(element.type) == "Atom_Tetrahedral"
                and str(element.specified) == "Specified"
            ):
                mol.GetAtomWithIdx(element.centeredOn).SetProp(
                    "Chirality", str(element.descriptor)
                )
            elif (
                str(element.type) == "Bond_Double"
                and str(element.specified) == "Specified"
            ):
                mol.GetBondWithIdx(element.centeredOn).SetProp(
                    "Stereochemistry", str(element.descriptor)
                )
        assert "." not in Chem.MolToSmiles(mol)
        return mol
    except:
        return None

def process_batch(batch, calc):
    mol_list = [get_mordred_mol(mol) for mol in batch]
    mol_list = [mol for mol in mol_list if mol is not None]
    if mol_list:
        return calc.pandas(mol_list, nproc=1).fill_missing(np.nan).to_numpy(dtype=float)
    else:
        return []

def get_mordred(molsuppl, mordred_save_path="../../data_1m", num_processes=64, batch_size=100_000):
    calc = Calculator(descriptors, ignore_3D=True)
    num_mols = len(molsuppl)
    pool = Pool(num_processes)
    
    for i in tqdm(range(0, num_mols, batch_size)):
        batch = list(islice(molsuppl, i, min(i + batch_size, num_mols)))
        results = pool.apply_async(process_batch, (batch, calc))
        
        mordred_batch = results.get()
        if mordred_batch:
            with open(os.path.join(mordred_save_path, f"pubchem_mordred_{i//batch_size}.npz"), "wb") as f:
                pickle.dump([mordred_batch], f, protocol=5)


# def get_mordred(molsuppl, mordred_save_path="../../data_1m", num_processes=256):
    
#     calc = Calculator(descriptors, ignore_3D=True)

#     mol_list = []

#     print("Creating mordred")
#     pool = Pool(num_processes)
#     mol_list = list(tqdm(pool.imap(get_mordred_mol, molsuppl), total=len(molsuppl)))
#     mol_list = [mol for mol in mol_list if mol is not None]
#     mordred_list = calc.pandas(mol_list).fill_missing(np.nan).to_numpy(dtype=float)

#     with open(
#         os.path.join(mordred_save_path, "pubchem_mordred.npz"),
#         "wb",
#     ) as f:
#         pickle.dump([mordred_list], f, protocol=5)

    


if __name__ == "__main__":
    seed = 27407
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False

    # We use the demo dataset (10k mols) for convenience in github repo.
    # The full dataset (10M mols collected from Pubchem) can be downloaded from
    # https://arxiv.org/pdf/2010.09885.pdf
    molsuppl = Chem.SmilesMolSupplier(
        "../../data_1m/pubchem-1m.txt", delimiter=","
    )

    if not os.path.exists("../../data_1m/pubchem_graph.npz"):
        preprocess(molsuppl)

    if not os.path.exists("../../data_1m/pubchem_mordred.npz"):
        get_mordred(molsuppl)
