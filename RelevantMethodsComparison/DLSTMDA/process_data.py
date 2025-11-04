import pandas as pd
import numpy as np
import os
import json, pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx
from utils import *


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index

def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        # print(line[0][0])
        # print(i,"========", ch)
        X[i] = smi_ch_ind[ch]
    return X

def seq_cat(prot,max_seq_len):
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]):
        x[i] = seq_dict[ch]
    return x

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, ">": 65, "<": 66}

CHARISOSMILEN = 66

# Load data from data0 folder
print("Loading data from data0 folder...")
drugs = pd.read_csv('../../Dataset/mi_drug_with_SMILES.csv')
rna = pd.read_excel('../../Dataset/miRNA_with_sequence.xlsx')
pairs = pd.read_csv('../../Dataset/MiPair.csv')

print(f"Drug data shape: {drugs.shape}")
print(f"miRNA data shape: {rna.shape}")
print(f"Pair data shape: {pairs.shape}")

# Process drug data
ligands = drugs['SMILES']
drug_names = drugs['Drug_Name']

# Process miRNA data
proteins = rna['Sequence']
mirna_names = rna['ncRNA_Name']

# Create drug name to SMILES mapping
drug_name_to_smiles = dict(zip(drug_names, ligands))

# Create miRNA name to sequence mapping
mirna_name_to_seq = dict(zip(mirna_names, proteins))

print(f"Unique drugs: {len(drug_name_to_smiles)}")
print(f"Unique miRNAs: {len(mirna_name_to_seq)}")

# Process association pair data
print("Processing association pairs...")
positive_pairs = []
negative_pairs = []

# Extract positive and negative samples from pair.csv
for idx, row in pairs.iterrows():
    drug_name = row['Drug_Name']
    mirna_name = row['ncRNA_Name']
    
    # Check if drug and miRNA exist in our data
    if drug_name in drug_name_to_smiles and mirna_name in mirna_name_to_seq:
        positive_pairs.append((drug_name, mirna_name))

print(f"Positive pairs found: {len(positive_pairs)}")

# Generate negative samples - randomly select non-existing drug-miRNA pairs
print("Generating negative samples...")
all_drug_names = list(drug_name_to_smiles.keys())
all_mirna_names = list(mirna_name_to_seq.keys())
positive_set = set(positive_pairs)

negative_pairs = []
max_negative = len(positive_pairs)  # Generate same number of negative samples as positive samples
attempts = 0
max_attempts = max_negative * 10

while len(negative_pairs) < max_negative and attempts < max_attempts:
    drug_name = np.random.choice(all_drug_names)
    mirna_name = np.random.choice(all_mirna_names)
    
    if (drug_name, mirna_name) not in positive_set:
        negative_pairs.append((drug_name, mirna_name))
    
    attempts += 1

print(f"Negative pairs generated: {len(negative_pairs)}")

# Create single fold data (no cross-validation)
print("Creating single fold data...")
np.random.seed(42)  # Set random seed for reproducibility

# Randomly shuffle positive and negative samples
np.random.shuffle(positive_pairs)
np.random.shuffle(negative_pairs)

# Use 80% for training and 20% for testing
train_split = 0.8
pos_train_size = int(len(positive_pairs) * train_split)
neg_train_size = int(len(negative_pairs) * train_split)

# Split data into train and test
pos_train = positive_pairs[:pos_train_size]
pos_test = positive_pairs[pos_train_size:]
neg_train = negative_pairs[:neg_train_size]
neg_test = negative_pairs[neg_train_size:]

print(f"Training positive samples: {len(pos_train)}")
print(f"Training negative samples: {len(neg_train)}")
print(f"Test positive samples: {len(pos_test)}")
print(f"Test negative samples: {len(neg_test)}")

# Create sequence vocabulary
all_prots = list(set(proteins))
seq_voc = "ACGU"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)

print(f"Sequence vocabulary: {seq_voc}")
print(f"Sequence dictionary length: {seq_dict_len}")

# Get all unique SMILES
compound_iso_smiles = []
for drug_name in drug_names:
    if drug_name in drug_name_to_smiles:
        compound_iso_smiles.append(drug_name_to_smiles[drug_name])

compound_iso_smiles = set(compound_iso_smiles)
print(f"Unique SMILES compounds: {len(compound_iso_smiles)}")

# Create SMILES to graph mapping
print("Creating SMILES to graph mappings...")
smile_graph = {}
for smile in compound_iso_smiles:
    try:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    except:
        print(f"Failed to process SMILES: {smile}")
        continue

print(f"Successfully processed {len(smile_graph)} SMILES to graphs")

# Create data directory
if not os.path.exists('data0/processed'):
    os.makedirs('data0/processed')

# Create training and test data
opts = ['train', 'test']

print(f"\nProcessing single fold...")

for opt in opts:
    if opt == 'train':
        pos_pairs = pos_train
        neg_pairs = neg_train
    else:  # test
        pos_pairs = pos_test
        neg_pairs = neg_test
    
    # Create CSV file
    csv_filename = f'processed/{opt}0.csv'
    with open(csv_filename, 'w') as f:
        f.write('compound_iso_smiles,target_sequence,affinity\n')
        
        # Write positive samples
        for drug_name, mirna_name in pos_pairs:
            if drug_name in drug_name_to_smiles and mirna_name in mirna_name_to_seq:
                smiles = drug_name_to_smiles[drug_name]
                sequence = mirna_name_to_seq[mirna_name]
                f.write(f'{smiles},{sequence},1\n')
        
        # Write negative samples
        for drug_name, mirna_name in neg_pairs:
            if drug_name in drug_name_to_smiles and mirna_name in mirna_name_to_seq:
                smiles = drug_name_to_smiles[drug_name]
                sequence = mirna_name_to_seq[mirna_name]
                f.write(f'{smiles},{sequence},0\n')
    
    print(f"Created {csv_filename}")

# Create PyTorch format data
print(f"\nCreating PyTorch format data...")

processed_data_file_train = f'processed/_train0.pt'
processed_data_file_test = f'processed/_test0.pt'

if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
    # Process training data
    df_train = pd.read_csv(f'processed/train0.csv')
    train_drugs, train_prots, train_Y = list(df_train['compound_iso_smiles']), list(df_train['target_sequence']), list(df_train['affinity'])
    
    XT = [seq_cat(t, 24) for t in train_prots]
    train_sdrugs = [label_smiles(t, CHARISOSMISET, 100) for t in train_drugs]
    train_drugs, train_prots, train_Y, train_seqdrugs = np.asarray(train_drugs), np.asarray(XT), np.asarray(train_Y), np.asarray(train_sdrugs)
    
    # Process test data
    df_test = pd.read_csv(f'processed/test0.csv')
    test_drugs, test_prots, test_Y = list(df_test['compound_iso_smiles']), list(df_test['target_sequence']), list(df_test['affinity'])
    
    XT = [seq_cat(t, 24) for t in test_prots]
    test_sdrugs = [label_smiles(t, CHARISOSMISET, 100) for t in test_drugs]
    test_drugs, test_prots, test_Y, test_seqdrugs = np.asarray(test_drugs), np.asarray(XT), np.asarray(test_Y), np.asarray(test_seqdrugs)

    print(f'Preparing train0.pt in pytorch format!')
    train_data = TestbedDataset(root='./', dataset=f'train0', xd=train_drugs, xt=train_prots, y=train_Y, z=train_seqdrugs,
                                smile_graph=smile_graph)
    
    print(f'Preparing test0.pt in pytorch format!')
    test_data = TestbedDataset(root='./', dataset=f'test0', xd=test_drugs, xt=test_prots, y=test_Y, z=test_seqdrugs,
                               smile_graph=smile_graph)
    
    print(f'{processed_data_file_train} and {processed_data_file_test} have been created')
else:
    print(f'{processed_data_file_train} and {processed_data_file_test} are already created')

print("\nData processing completed successfully!")
print(f"Processed data saved in: data0/processed/")
print(f"Single fold data created (80% train, 20% test)")

# Script runs directly - no main() function needed
