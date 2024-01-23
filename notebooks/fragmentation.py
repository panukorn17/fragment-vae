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

def check_reconstruction(frags, frag_1, frag_2, orig_smi):
    try:
        print("Reconstructing...")
        frags_test = frags.copy()
        frags_test.append(frag_1)
        frags_test.append(frag_2)
        frag_2_re = replace_square_brackets(frags_test[-1])
        for i in range(len(frags_test)-1):
            frag_1_re = replace_square_brackets(frags_test[-1*i-2])
            recomb = replace_last(frag_2_re, "*", frag_1_re.replace("*", "",1))
            recomb_canon = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(recomb)),rootedAtAtom = 1)
            frag_2_re = recomb_canon
        #print(recomb_canon)
        orig_smi_canon = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(orig_smi)),rootedAtAtom = 1)
        if recomb_canon == orig_smi_canon:
            print("Reconstruction successful")
            print("Original Smiles:", orig_smi, "Fragment 1:" , frag_1, "Fragment 2: ", frag_2, "Reconstruction: ", recomb_canon)
            return True
        else:
            #print("Reconstruction failed")
            #print("True Smiles:", smi, "Fragment 1:" , frag_1, "Fragment 2: ", frag_2, "Reconstruction: ", recomb_canon)
            return False
    except:
        #print("Reconstruction failed")
        #print("True Smiles:", smi, "Fragment 1:" , frag_1, "Fragment 2: ", frag_2, "Reconstruction: ", recomb_canon)
        return False

def fragment_recursive(mol_smi_orig, mol_smi, frags, counter, frag_list_len):
    fragComplete = False
    try:
        counter += 1
        mol = MolFromSmiles(mol_smi)
        bonds = list(BRICS.FindBRICSBonds(mol))
        if len(bonds) <= frag_list_len:
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
            if len(list(BRICS.FindBRICSBonds(head))) <= frag_list_len:
                #broken = Chem.FragmentOnBonds(mol,
                #                            bondIndices=[bond],
                #                            dummyLabels=[(counter, 0)])
                #head, tail = Chem.GetMolFrags(broken, asMols=True)
                head_smi = Chem.CanonSmiles(MolToSmiles(head))
                tail_smi = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(MolToSmiles(tail))), rootedAtAtom=1)
                if check_reconstruction(frags, head_smi, tail_smi, mol_smi_orig):
                    frags.append(head_smi)
                    print("Recursed: ", mol_smi, "Bond: ", bond, "Terminal: ", head_smi, "Recurse: ", tail_smi)
                    fragComplete = fragment_recursive(mol_smi_orig, tail_smi, frags, counter, frag_list_len)  
                    if fragComplete:
                        return frags
                elif len(bond_idxs) == 1:
                    frags.append(MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1))
                    print("Final Fragment: ", mol_smi)
                    fragComplete = True
                    return frag
                elif bond == bond_idxs[-1]:
                    fragComplete = fragment_recursive(mol_smi_orig, MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1), frags, counter, frag_list_len + 1)
                    if fragComplete:
                        return frags
            elif len(list(BRICS.FindBRICSBonds(tail))) <= frag_list_len:
                #broken = Chem.FragmentOnBonds(mol,
                #                            bondIndices=[bond],
                #                            dummyLabels=[(0, counter)])
                #head, tail = Chem.GetMolFrags(broken, asMols=True)
                tail_smi = Chem.CanonSmiles(MolToSmiles(tail))
                head_smi = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(MolToSmiles(head))), rootedAtAtom=1)
                if check_reconstruction(frags, tail_smi, head_smi, mol_smi_orig):
                    frags.append(tail_smi)
                    print("Recursed: ", mol_smi, "Bond: ", bond,  "Terminal: ", tail_smi, "Recurse: ", head_smi)
                    fragComplete = fragment_recursive(mol_smi_orig, head_smi, frags, counter, frag_list_len)  
                    if fragComplete:
                        return frags
                elif len(bond_idxs) == 1:
                    frags.append(MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1))
                    print("Final Fragment: ", mol_smi)
                    fragComplete = True
                    return frag
                elif bond == bond_idxs[-1]:
                    fragComplete = fragment_recursive(mol_smi_orig, MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1), frags, counter, frag_list_len + 1)
                    if fragComplete:
                        return frags
                    
        
    except Exception:
        pass

#smiles = Chem.CanonSmiles('Oc1cccc(C(C(=O)NC2CCCC2)N(C(=O)c2ccco2)c2ccccc2F)c1OC')
#smiles = Chem.CanonSmiles('CCCN(CCc1cccc(-c2ccccc2)c1)C(=O)C1OC(C(=O)O)=CC(N)C1NC(C)=O')
#smiles = Chem.CanonSmiles('CCC(CC)N1CCN(C(CN2CCN(CCCCc3c(OC)ccc4ccccc34)CC2)c2ccc(F)cc2)CC1')
#smiles = Chem.CanonSmiles('CCOC(=O)CCCSc1nc(O)c2c(C)c(C)sc2n1')
#smiles = Chem.CanonSmiles('Cc1nc(Oc2ccc(NS(C)(=O)=O)cc2Cl)ccc1CN1CCC(N(C(=O)Nc2ccc(C(N)=O)nc2)c2cccc(F)c2)CC1')
smiles = Chem.CanonSmiles('COc1nc(N)nc2c1ncn2C1OC(COP(=O)(NCCC(=O)OCc2ccccc2)NCCC(=O)OCc2ccccc2)C(O)C1(C)O')


print(smiles)
frag = []
fragment_recursive(smiles, smiles, frag, 0, 0)
print(frag)
