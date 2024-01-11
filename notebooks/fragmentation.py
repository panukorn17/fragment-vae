import numpy as np
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import MolToSmiles, MolFromSmiles


def break_on_bond(mol, bond, min_length=3):
    if mol.GetNumAtoms() - bond <= min_length:
        return [mol]

    broken = Chem.FragmentOnBonds(
        mol, bondIndices=[bond],
        dummyLabels=[(0, 0)])

    res = Chem.GetMolFrags(
        broken, asMols=True, sanitizeFrags=False)

    return res

def fragment_iterative(mol, min_length=3):

    bond_data = list(BRICS.FindBRICSBonds(mol))

    try:
        idxs, labs = zip(*bond_data)
    except Exception:
        return []

    bonds = []
    for a1, a2 in idxs:
        bond = mol.GetBondBetweenAtoms(a1, a2)
        bonds.append(bond.GetIdx())

    order = np.argsort(bonds).tolist()
    bonds = [bonds[i] for i in order]

    frags, temp = [], deepcopy(mol)
    for bond in bonds:
        res = break_on_bond(temp, bond)

        if len(res) == 1:
            frags.append(temp)
            break

        head, tail = res

        frags.append(head)
        temp = deepcopy(tail)

    return frags

def break_into_fragments(mol, smi):
    frags = fragment_iterative(mol)

    if len(frags) == 0:
        return smi, np.nan, 0

    if len(frags) == 1:
        return smi, smi, 1

    return smi, np.nan, 0

def fragment_recursive(mol, frags):
    fragComplete = False
    try:
        bonds = list(BRICS.FindBRICSBonds(mol))
        if bonds == []:
            frags.append(mol)
            fragComplete = True
            return fragComplete

        idxs, labs = list(zip(*bonds))

        bond_idxs = []
        for a1, a2 in idxs:
            bond = mol.GetBondBetweenAtoms(a1, a2)
            bond_idxs.append(bond.GetIdx())

        order = np.argsort(bond_idxs).tolist()
        bond_idxs = [bond_idxs[i] for i in order]
        for bond in bond_idxs:
            print(fragComplete)
            broken = Chem.FragmentOnBonds(mol,
                                        bondIndices=[bond],
                                        dummyLabels=[(0, 0)])
            head, tail = Chem.GetMolFrags(broken, asMols=True)
            head_smi = MolToSmiles(head)
            tail_smi = MolToSmiles(tail)
            if not list(BRICS.FindBRICSBonds(head)):
                frags.append(head)
                print("Terminal:", head_smi, "Recurse:", tail_smi)
                fragComplete = fragment_recursive(tail, frags)  
                if fragComplete:
                    return frags
            elif not list(BRICS.FindBRICSBonds(tail)):
                frags.append(tail)
                print("Terminal:", tail_smi, "Recurse:", head_smi)
                fragComplete = fragment_recursive(head, frags)  
                if fragComplete:
                    return frags
        
    except Exception:
        pass

smiles = 'CCCN(CCc1cccc(-c2ccccc2)c1)C(=O)C1OC(C(=O)O)=CC(N)C1NC(C)=O'
mol = MolFromSmiles(smiles)
frag = []
fragment_recursive(mol, frag)
print([MolToSmiles(fragment) for fragment in frag])
