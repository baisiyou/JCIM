"""Fingerprint featurization for molecules and BRICS fragments."""

from __future__ import annotations

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

EDIT_TYPE_TO_ID = {"replace": 0, "add": 1, "remove": 2}


def smiles_to_fp(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Morgan fingerprint; zeros if invalid / empty."""
    fp = np.zeros(n_bits, dtype=np.float32)
    if not smiles:
        return fp
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fp
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr.astype(np.float32)


def batch_smiles_to_fp(
    smiles_list: list[str],
    n_bits: int = 2048,
    radius: int = 2,
) -> np.ndarray:
    return np.stack([smiles_to_fp(s, n_bits=n_bits, radius=radius) for s in smiles_list])
