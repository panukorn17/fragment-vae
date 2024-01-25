import numpy as np

from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import MolToSmiles, MolFromSmiles

def replace_last(s, old, new):
    s_reversed = s[::-1]
    old_reversed = old[::-1]
    new_reversed = new[::-1]

    # Replace the first occurrence in the reversed string
    s_reversed = s_reversed.replace(old_reversed, new_reversed, 1)

    # Reverse the string back to original order
    return s_reversed[::-1]

def check_reconstruction(frags, frag_1, frag_2, orig_smi):
    try:
        print("Reconstructing...")
        frags_test = frags.copy()
        frags_test.append(frag_1)
        frags_test.append(frag_2)
        frag_2_re = frags_test[-1]
        for i in range(len(frags_test)-1):
            frag_1_re = frags_test[-1*i-2]
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
            print("Final Fragment: ", mol_smi, "Number of BRIC bonds: ", len(bonds))
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
            broken = Chem.FragmentOnBonds(mol,
                                        bondIndices=[bond],
                                        dummyLabels=[(0, 0)])
            head, tail = Chem.GetMolFrags(broken, asMols=True)
            head_bric_bond_no = len(list(BRICS.FindBRICSBonds(head)))
            tail_bric_bond_no = len(list(BRICS.FindBRICSBonds(tail)))
            if head_bric_bond_no <= frag_list_len:
                head_smi = Chem.CanonSmiles(MolToSmiles(head))
                tail_smi = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(MolToSmiles(tail))), rootedAtAtom=1)
                if check_reconstruction(frags, head_smi, tail_smi, mol_smi_orig):
                    frags.append(head_smi)
                    print("Recursed: ", mol_smi, "Bond: ", bond, "Terminal: ", head_smi, "Number of BRIC bonds: ", head_bric_bond_no, "Recurse: ", tail_smi)
                    fragComplete = fragment_recursive(mol_smi_orig, tail_smi, frags, counter, frag_list_len = 0)  
                    if fragComplete:
                        return frags
                elif len(bond_idxs) == 1:
                    frags.append(MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1))
                    print("Final Fragment: ", mol_smi, "Number of BRIC bonds: ", len(bonds))
                    fragComplete = True
                    return frags
                elif bond == bond_idxs[-1]:
                    fragComplete = fragment_recursive(mol_smi_orig, MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1), frags, counter, frag_list_len + 1)
                    if fragComplete:
                        return frags
            elif tail_bric_bond_no <= frag_list_len:
                tail_smi = Chem.CanonSmiles(MolToSmiles(tail))
                head_smi = MolToSmiles(MolFromSmiles(Chem.CanonSmiles(MolToSmiles(head))), rootedAtAtom=1)
                if check_reconstruction(frags, tail_smi, head_smi, mol_smi_orig):
                    frags.append(tail_smi)
                    print("Recursed: ", mol_smi, "Bond: ", bond,  "Terminal: ", tail_smi, "Number of BRIC bonds: ", tail_bric_bond_no, "Recurse: ", head_smi)
                    fragComplete = fragment_recursive(mol_smi_orig, head_smi, frags, counter, frag_list_len = 0)  
                    if fragComplete:
                        return frags
                elif len(bond_idxs) == 1:
                    frags.append(MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1))
                    print("Final Fragment: ", mol_smi, "Number of BRIC bonds: ", len(bonds))
                    fragComplete = True
                    return frags
                elif bond == bond_idxs[-1]:
                    fragComplete = fragment_recursive(mol_smi_orig, MolToSmiles(MolFromSmiles(Chem.CanonSmiles(mol_smi)), rootedAtAtom=1), frags, counter, frag_list_len + 1)
                    if fragComplete:
                        return frags              
    except Exception:
        pass

#smiles = Chem.CanonSmiles('Oc1cccc(C(C(=O)NC2CCCC2)N(C(=O)c2ccco2)c2ccccc2F)c1OC')
smiles = Chem.CanonSmiles('CCCN(CCc1cccc(-c2ccccc2)c1)C(=O)C1OC(C(=O)O)=CC(N)C1NC(C)=O') # fragment demonstration in the thesis
#smiles = Chem.CanonSmiles('CCC(CC)N1CCN(C(CN2CCN(CCCCc3c(OC)ccc4ccccc34)CC2)c2ccc(F)cc2)CC1')
#smiles = Chem.CanonSmiles('CCOC(=O)CCCSc1nc(O)c2c(C)c(C)sc2n1')
#smiles = Chem.CanonSmiles('Cc1nc(Oc2ccc(NS(C)(=O)=O)cc2Cl)ccc1CN1CCC(N(C(=O)Nc2ccc(C(N)=O)nc2)c2cccc(F)c2)CC1')
#smiles = Chem.CanonSmiles('O=C(O)CCC(NC(=O)NC(CSC1CC(=O)N(CCCCC(NC(=O)CNC(=O)c2cccc(I)c2)C(=O)O)C1=O)C(=O)O)C(=O)O')
#smiles = Chem.CanonSmiles('CN(C)c1ccc(C(=O)Nc2ccc(C(=O)Nc3cc(C(=O)Nc4cc(C(=O)NCCN5CCOCC5)n(C)c4)n(C)c3)cc2)cc1')
#smiles = Chem.CanonSmiles('CCCCCCCCCCCCCCCCCCN(CCCCCCCCCCCCCCCCCC)C(=O)c1ccccc1C(=O)OCC1OC(OC)C(OCCN)C(OCCN)C1OCCN')
#smiles = Chem.CanonSmiles('CC(C)CN(C(CO)CCCCNC(=O)C(NC(=O)c1ccc([N+](=O)[O-])c(O)c1)C(c1ccccc1)c1ccccc1)S(=O)(=O)c1ccc(N)cc1')
#smiles = Chem.CanonSmiles('Cc1ccccc1NC(=O)NCCCCC(NC(=O)C(Cc1c[nH]c2ccccc12)NC(=O)OC(C)(C)C)C(=O)N(C)CC(=O)NC(Cc1ccccc1)C(N)=O')
#smiles = Chem.CanonSmiles('CCNCCCCNCCCCNCCCCNCCCCNCC=C(CS)CNCCCCNCCCCNCCCCNCCCCNCC')
#smiles = Chem.CanonSmiles('CCC(NC(=O)C(CC1CCCCC1)NC(=O)C(NC(=O)C(CC(C)C)NC(=O)c1cnccn1)C(C)CC)C(=O)C(=O)NCC(N)=O') # wrong intermediatary molecules
#smiles = Chem.CanonSmiles('CCOC(=O)c1c(NC(=O)NS(=O)(=O)c2ccnn2C)sc2c1CCC(C)(C)C2') # not sure if same molecule or can it rotate?
#smiles = Chem.CanonSmiles('CN1Cc2c(Cl)cc(Cl)cc2C(c2cccc(S(=O)(=O)NCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCN=[N+]=[N-])c2)C1') # 23 fragments

print(smiles)
frag = []
fragment_recursive(smiles, smiles, frag, 0, 0)
print(frag)
