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

def replace_last(s, old, new):
    s_reversed = s[::-1]
    old_reversed = old[::-1]
    new_reversed = new[::-1]

    # Replace the first occurrence in the reversed string
    s_reversed = s_reversed.replace(old_reversed, new_reversed, 1)

    # Reverse the string back to original order
    return s_reversed[::-1]

def check_reconstruction(frag_1, frag_2, smi):
    print("Reconstructing...")
    recomb = replace_last(frag_2, "*", frag_1.replace("*","",1))
    #print(recomb)
    try:
        recomb_canon = Chem.CanonSmiles(MolToSmiles(MolFromSmiles(recomb), rootedAtAtom=0))
        #print(recomb_canon)
        if recomb_canon == smi:
            print("Reconstruction successful")
            print("True Smiles:", smi, "Fragment 1:" , frag_1, "Fragment 2: ", frag_2, "Reconstruction: ", recomb_canon)
            return True
        else:
            #print("Reconstruction failed")
            #print("True Smiles:", smi, "Fragment 1:" , frag_1, "Fragment 2: ", frag_2, "Reconstruction: ", recomb_canon)
            return False
    except:
        #print("Reconstruction failed")
        #print("True Smiles:", smi, "Fragment 1:" , frag_1, "Fragment 2: ", frag_2, "Reconstruction: ", recomb_canon)
        return False

def fragment_recursive(mol_smi, frags):
    fragComplete = False
    try:
        mol = MolFromSmiles(mol_smi)
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
            #print(fragComplete)
            broken = Chem.FragmentOnBonds(mol,
                                        bondIndices=[bond],
                                        dummyLabels=[(0, 0)])
            head, tail = Chem.GetMolFrags(broken, asMols=True)
            if not list(BRICS.FindBRICSBonds(head)):
                head_smi = Chem.CanonSmiles(MolToSmiles(head))
                tail_smi = MolToSmiles(tail, rootedAtAtom=0)
                if check_reconstruction(head_smi,tail_smi,mol_smi):
                    frags.append(head_smi)
                    print("Bond: ", bond, "Terminal: ", head_smi, "Recurse: ", tail_smi)
                    fragComplete = fragment_recursive(tail_smi, frags)  
                    if fragComplete:
                        return frags
            elif not list(BRICS.FindBRICSBonds(tail)):
                tail_smi = Chem.CanonSmiles(MolToSmiles(tail))
                head_smi = MolToSmiles(head, rootedAtAtom=0)
                if check_reconstruction(tail_smi,head_smi,mol_smi):
                    frags.append(tail_smi)
                    print("Bond: ", bond,  "Terminal: ", tail_smi, "Recurse: ", head_smi)
                    fragComplete = fragment_recursive(head_smi, frags)  
                    if fragComplete:
                        return frags
                    
        
    except Exception:
        pass

smiles = Chem.CanonSmiles('CCCN(CCc1cccc(-c2ccccc2)c1)C(=O)C1OC(C(=O)O)=CC(N)C1NC(C)=O')
print(smiles)
frag = []
fragment_recursive(smiles, frag)
print(frag)
