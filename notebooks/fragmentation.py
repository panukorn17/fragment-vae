import numpy as np
import re

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

def replace_square_brackets(text):
    # This function will replace anything within square brackets, including the brackets themselves
    return re.sub(r"\[.*?\]", "*", text)

def check_reconstruction(frag_1, frag_2, smi):
    print("Reconstructing...")
    smi_re = replace_square_brackets(smi)
    smi_canon = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(smi_re)),rootedAtAtom = 1)
    frag_1_re = replace_square_brackets(frag_1)
    frag_2_re = replace_square_brackets(frag_2)
    recomb = replace_last(frag_2_re, "*", frag_1_re.replace("*", "",1))
    #recomb = frag_2.replace("*", frag_1.replace("*","",1),1)
    #print(recomb)
    try:
        recomb_canon = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(recomb)),rootedAtAtom = 1)
        #print(recomb_canon)
        if recomb_canon == smi_canon:
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

def fragment_recursive(mol_smi, frags, counter):
    fragComplete = False
    try:
        counter += 1
        mol = MolFromSmiles(mol_smi)
        bonds = list(BRICS.FindBRICSBonds(mol))
        if bonds == []:
            frags.append(MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1))
            print("Final Fragment: ", mol_smi)
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
                broken = Chem.FragmentOnBonds(mol,
                                            bondIndices=[bond],
                                            dummyLabels=[(counter, 0)])
                head, tail = Chem.GetMolFrags(broken, asMols=True)
                head_smi = Chem.CanonSmiles(MolToSmiles(head))
                tail_smi = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(MolToSmiles(tail))), rootedAtAtom=1)
                if check_reconstruction(head_smi,tail_smi,mol_smi):
                    frags.append(head_smi)
                    print("Bond: ", bond, "Terminal: ", head_smi, "Recurse: ", tail_smi)
                    fragComplete = fragment_recursive(tail_smi, frags, counter)  
                    if fragComplete:
                        return frags
                elif len(bond_idxs) == 1:
                    frags.append(MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1))
                    print("Final Fragment: ", mol_smi)
                    fragComplete = True
                    return frag
                elif bond == bond_idxs[-1]:
                    frags.append(MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1))
                    print("Final Fragment: ", mol_smi)
                    fragComplete = True
                    return frags
            elif not list(BRICS.FindBRICSBonds(tail)):
                broken = Chem.FragmentOnBonds(mol,
                                            bondIndices=[bond],
                                            dummyLabels=[(0, counter)])
                head, tail = Chem.GetMolFrags(broken, asMols=True)
                tail_smi = Chem.CanonSmiles(MolToSmiles(tail))
                head_smi = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(MolToSmiles(head))), rootedAtAtom=1)
                if check_reconstruction(tail_smi,head_smi,mol_smi):
                    frags.append(tail_smi)
                    print("Bond: ", bond,  "Terminal: ", tail_smi, "Recurse: ", head_smi)
                    fragComplete = fragment_recursive(head_smi, frags, counter)  
                    if fragComplete:
                        return frags
                elif len(bond_idxs) == 1:
                    frags.append(MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1))
                    print("Final Fragment: ", mol_smi)
                    fragComplete = True
                    return frags
                elif bond == bond_idxs[-1]:
                    frags.append(MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1))
                    print("Final Fragment: ", mol_smi)
                    fragComplete = True
                    return frags
                    
        
    except Exception:
        pass

#smiles = Chem.CanonSmiles('Oc1cccc(C(C(=O)NC2CCCC2)N(C(=O)c2ccco2)c2ccccc2F)c1OC')
#smiles = Chem.CanonSmiles('CCCN(CCc1cccc(-c2ccccc2)c1)C(=O)C1OC(C(=O)O)=CC(N)C1NC(C)=O')
#smiles = Chem.CanonSmiles('CCC(CC)N1CCN(C(CN2CCN(CCCCc3c(OC)ccc4ccccc34)CC2)c2ccc(F)cc2)CC1')
smiles = Chem.CanonSmiles('CCOC(=O)CCCSc1nc(O)c2c(C)c(C)sc2n1')

print(smiles)
frag = []
fragment_recursive(smiles, frag, 0)
print(frag)
