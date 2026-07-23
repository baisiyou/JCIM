"""Chemistry-facing utilities: SA score, fragment highlighting, reaction checks."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdMolDescriptors


@dataclass
class ChemistryAudit:
    sa_score: float | None
    valid: bool
    highlight_atoms: list[int]
    highlight_bonds: list[int]
    note: str


def synthetic_accessibility(smiles: str) -> float | None:
    """Ertl SA score when RDKit contrib is available."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem import RDConfig
        import os
        import sys

        sa_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if sa_path not in sys.path:
            sys.path.append(sa_path)
        import sascorer  # type: ignore

        return float(sascorer.calculateScore(mol))
    except Exception:
        # fallback proxy: lower is easier; use Bertz complexity normalized
        try:
            complexity = float(rdMolDescriptors.BertzCT(mol))
            return min(10.0, complexity / 150.0)
        except Exception:
            return None


def _bonds_within(mol: Chem.Mol, atoms: list[int]) -> list[int]:
    atom_set = set(atoms)
    return [
        bond.GetIdx()
        for bond in mol.GetBonds()
        if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set
    ]


def _brics_query_variants(frag_smiles: str) -> list[str]:
    """Generate RDKit-parseable queries from BRICS attachment SMILES like [16*]c1ccc(F)c(F)c1."""
    import re

    variants = [frag_smiles]
    # Keep dummy atoms: [16*] -> [*]
    variants.append(re.sub(r"\[\d+\*\]", "[*]", frag_smiles))
    # Drop attachment markers entirely (may leave empty parens — filtered below)
    stripped = re.sub(r"\[\d+\*\]", "", frag_smiles)
    stripped = re.sub(r"\(\)", "", stripped)
    if stripped and stripped not in variants:
        variants.append(stripped)
    # SMARTS-friendly: numbered dummies as wildcards
    variants.append(re.sub(r"\[\d+\*\]", "*", frag_smiles))
    return variants


def fragment_highlight_indices(mol: Chem.Mol, frag_smiles: str) -> tuple[list[int], list[int]]:
    """Return atom/bond indices matching a BRICS fragment SMARTS/SMILES."""
    if not frag_smiles:
        return [], []
    for candidate in _brics_query_variants(frag_smiles):
        query = Chem.MolFromSmiles(candidate)
        if query is None:
            query = Chem.MolFromSmarts(candidate)
        if query is None:
            continue
        matches = mol.GetSubstructMatches(query)
        if matches:
            atoms = sorted(set(matches[0]))
            return atoms, _bonds_within(mol, atoms)
    return [], []


def changed_atom_highlight(
    prev_smiles: str, curr_smiles: str
) -> tuple[list[int], list[int]]:
    """Highlight atoms in curr that are outside the MCS with prev (edit locus)."""
    prev = Chem.MolFromSmiles(prev_smiles)
    curr = Chem.MolFromSmiles(curr_smiles)
    if prev is None or curr is None:
        return [], []
    try:
        from rdkit.Chem import rdFMCS

        res = rdFMCS.FindMCS(
            [prev, curr],
            timeout=2,
            matchValences=True,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
        )
        if res.numAtoms < 1:
            return [], []
        mcs = Chem.MolFromSmarts(res.smartsString)
        if mcs is None:
            return [], []
        match = curr.GetSubstructMatch(mcs)
        if not match:
            return [], []
        kept = set(match)
        atoms = [i for i in range(curr.GetNumAtoms()) if i not in kept]
        return atoms, _bonds_within(curr, atoms)
    except Exception:
        return [], []


def audit_edit_step(source_smiles: str, target_smiles: str, frag_old: str, frag_new: str) -> ChemistryAudit:
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        return ChemistryAudit(None, False, [], [], "invalid target")
    atoms, bonds = fragment_highlight_indices(mol, frag_new)
    sa = synthetic_accessibility(target_smiles)
    note = "fragment change audited"
    if frag_old and frag_new:
        note = f"replace `{frag_old}` -> `{frag_new}`"
    return ChemistryAudit(sa, True, atoms, bonds, note)


def draw_highlighted_molecule(
    smiles: str,
    highlight_atoms: list[int],
    highlight_bonds: list[int],
    size: tuple[int, int] = (320, 240),
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    return Draw.MolToImage(
        mol,
        size=size,
        highlightAtoms=highlight_atoms or None,
        highlightBonds=highlight_bonds or None,
    )
