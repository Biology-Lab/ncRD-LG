#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified data preprocessing for Drug-ncRNA interaction prediction

"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Batch
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore')

def create_dir(directory):

    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_dataset_files(dataset_type='miRNA'):
   
    print(f"Loading {dataset_type} dataset files...")
    
  
    if dataset_type == 'miRNA':
       
        try:
            drugs_df = pd.read_csv('../../Dataset/mi_drug_with_SMILES.csv')
            print(f"Loaded drugs: {len(drugs_df)} samples")
            print(f"Drug columns: {drugs_df.columns.tolist()}")
            
           
            if 'Drug_Name' not in drugs_df.columns or 'SMILES' not in drugs_df.columns:
                print("Error: Missing required columns in drug data")
                return None, None, None
                
        except Exception as e:
            print(f"Error loading drugs: {e}")
            return None, None, None
        
    
        try:
            rna_df = pd.read_excel('../../Dataset/miRNA_with_sequence.xlsx')
            print(f"Loaded miRNAs: {len(rna_df)} samples")
            print(f"miRNA columns: {rna_df.columns.tolist()}")
            
            
            if 'miRNA_Name' in rna_df.columns:
                print("Found miRNA_Name column")
            elif 'ncRNA_Name' in rna_df.columns:
                print("Found ncRNA_Name column, will use this")
            else:
                print("Error: No miRNA name column found")
                return None, None, None
                
            if 'Sequence' not in rna_df.columns:
                print("Error: No Sequence column found in miRNA data")
                return None, None, None
                
        except Exception as e:
            print(f"Error loading miRNAs: {e}")
            return None, None, None
        
       
        try:
            pairs_df = pd.read_csv('../../Dataset/MiPair.csv')
            print(f"Loaded pairs: {len(pairs_df)} samples")
            print(f"Pair columns: {pairs_df.columns.tolist()}")
            
           
            if 'Drug_Name' not in pairs_df.columns or 'ncRNA_Name' not in pairs_df.columns:
                print("Error: Missing required columns in pair data")
                return None, None, None
                
        except Exception as e:
            print(f"Error loading pairs: {e}")
            return None, None, None
    
    
    else:  # dataset_type == 'lncRNA'
      
        # try:
        #     drugs_df = pd.read_excel('../../Dataset/lnc_drug_with_SMILES.xlsx')
        #     print(f"Loaded drugs: {len(drugs_df)} samples")
        #     print(f"Drug columns: {drugs_df.columns.tolist()}")
        #     
        #   
        #     if 'Drug_Name' not in drugs_df.columns or 'SMILES' not in drugs_df.columns:
        #         print("Error: Missing required columns in drug data")
        #         return None, None, None
        #         
        # except Exception as e:
        #     print(f"Error loading drugs: {e}")
        #     return None, None, None
        
       
        # try:
        #     rna_df = pd.read_excel('../../Dataset/lncRNA_with_sequence.xlsx')
        #     print(f"Loaded lncRNAs: {len(rna_df)} samples")
        #     print(f"lncRNA columns: {rna_df.columns.tolist()}")
        #  
        #     if 'lncRNA_Name' in rna_df.columns:
        #         print("Found lncRNA_Name column")
        #     elif 'ncRNA_Name' in rna_df.columns:
        #         print("Found ncRNA_Name column, will use this")
        #     else:
        #         print("Error: No lncRNA name column found")
        #         return None, None, None
        #         
        #     if 'Sequence' not in rna_df.columns:
        #         print("Error: No Sequence column found in lncRNA data")
        #         return None, None, None
        #         
        # except Exception as e:
        #     print(f"Error loading lncRNAs: {e}")
        #     return None, None, None
        
        # try:
        #     pairs_df = pd.read_csv('../../Dataset/LncPair.csv')
        #     print(f"Loaded pairs: {len(pairs_df)} samples")
        #     print(f"Pair columns: {pairs_df.columns.tolist()}")
        #     
       
        #     if 'Drug_Name' not in pairs_df.columns or 'ncRNA_Name' not in pairs_df.columns:
        #         print("Error: Missing required columns in pair data")
        #         return None, None, None
        #         
        # except Exception as e:
        #     print(f"Error loading pairs: {e}")
        #     return None, None, None
        pass
    
    return drugs_df, rna_df, pairs_df

def validate_smiles(smiles):
  
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
       
        return mol.GetNumAtoms() > 0
    except:
        return False

def smiles_to_graph(smiles, max_atoms=100):
   
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
      
        if mol.GetNumAtoms() > max_atoms:
            return None
        
        
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                float(atom.GetAtomicNum()),     
                float(atom.GetDegree()),         
                float(atom.GetFormalCharge()),   
                float(atom.GetIsAromatic())     
            ]
            atom_features.append(features)
        
      
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
          
            edge_indices.extend([[i, j], [j, i]])
            
          
            bond_type = float(bond.GetBondTypeAsDouble())
            edge_features.extend([[bond_type], [bond_type]])
        
        if len(edge_indices) == 0:
            return None
        
        
        x = torch.tensor(atom_features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {e}")
        return None

def rna_to_onehot(sequence, max_length=50, dataset_type='miRNA'):
   
    if pd.isna(sequence) or sequence is None:
        sequence = ""
    
    sequence = str(sequence).upper().strip()
    
   
    if dataset_type == 'miRNA':
       
        valid_chars = set('AUGC')
        if not all(c in valid_chars for c in sequence):
        
            sequence = ''.join([c for c in sequence if c in valid_chars])
        
       
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
        elif len(sequence) < max_length:
            sequence = sequence + 'N' * (max_length - len(sequence))
        
       
        vocab = {'A': 0, 'U': 1, 'G': 2, 'C': 3, 'N': 4}  
        onehot = torch.zeros(max_length, 5) 
        for i, char in enumerate(sequence):
            if char in vocab:
                onehot[i, vocab[char]] = 1.0
            else:
                onehot[i, vocab['N']] = 1.0  
    
   
    else:  # dataset_type == 'lncRNA'
       
        # base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'U': 1} 
        
       
        # num_seq = []
        # for base in sequence:
        #     if base in base_map:
        #         num_seq.append(base_map[base])
        #     else:
        #        
        #         continue
        
        # if len(num_seq) == 0:
        #     return None
        
      
        # onehot = np.zeros((len(num_seq), 4))
        # for i, base_num in enumerate(num_seq):
        #     onehot[i, base_num] = 1
        
        # return torch.tensor(onehot, dtype=torch.float)
        pass
    
    return onehot

def process_drugs(drugs_df):
   
    print("Processing drug data...")
    
    drug_dict = {}
    valid_drugs = 0
    invalid_drugs = 0
    
    for idx, row in drugs_df.iterrows():
        drug_name = row['Drug_Name']
        smiles = row['SMILES']
        
       
        if pd.isna(smiles) or smiles is None or str(smiles).strip() == '':
            print(f"Empty or invalid SMILES for {drug_name}: {smiles}")
            invalid_drugs += 1
            continue
        
       
        if not validate_smiles(smiles):
            print(f"Invalid SMILES for {drug_name}: {smiles}")
            invalid_drugs += 1
            continue
    
        graph = smiles_to_graph(smiles)
        if graph is None:
            print(f"Failed to convert SMILES to graph for {drug_name}")
            invalid_drugs += 1
            continue
        
        drug_dict[drug_name] = {
            'smiles': smiles,
            'graph': graph,
            'num_atoms': graph.x.size(0),
            'num_bonds': graph.edge_index.size(1) // 2
        }
        valid_drugs += 1
    
    print(f"Valid drugs: {valid_drugs}, Invalid drugs: {invalid_drugs}")
    return drug_dict

def process_rnas(rna_df, dataset_type='miRNA'):
    
    print(f"Processing {dataset_type} data...")
    

    if dataset_type == 'miRNA':
        if 'miRNA_Name' in rna_df.columns:
            name_col = 'miRNA_Name'
        elif 'ncRNA_Name' in rna_df.columns:
            name_col = 'ncRNA_Name'
        else:
            print("Error: No miRNA name column found")
            return {}
    else:  # lncRNA
        if 'lncRNA_Name' in rna_df.columns:
            name_col = 'lncRNA_Name'
        elif 'ncRNA_Name' in rna_df.columns:
            name_col = 'ncRNA_Name'
        else:
            print("Error: No lncRNA name column found")
            return {}
    
    rna_dict = {}
    valid_rnas = 0
    invalid_rnas = 0
    
    for idx, row in rna_df.iterrows():
        rna_name = row[name_col]
        sequence = row['Sequence']
        
     
        if pd.isna(sequence) or sequence is None or str(sequence).strip() == '':
            print(f"Empty or invalid sequence for {rna_name}: {sequence}")
            invalid_rnas += 1
            continue
        
      
        onehot = rna_to_onehot(sequence, dataset_type=dataset_type)
        if onehot is None:
            print(f"Failed to process sequence for {rna_name}: {sequence}")
            invalid_rnas += 1
            continue
        
        rna_dict[rna_name] = {
            'sequence': sequence,
            'onehot': onehot,
            'length': len(str(sequence).strip())
        }
        valid_rnas += 1
    
    print(f"Valid {dataset_type}s: {valid_rnas}, Invalid {dataset_type}s: {invalid_rnas}")
    return rna_dict

def create_interaction_dataset(pairs_df, drug_dict, rna_dict, dataset_type='miRNA'):
  
    print("Creating interaction dataset...")
    
    positive_samples = []
    missing_drugs = set()
    missing_rnas = set()
    
   
    if dataset_type == 'miRNA':
        for idx, row in pairs_df.iterrows():
            drug_name = row['Drug_Name']
            rna_name = row['ncRNA_Name']
            
        
            if drug_name not in drug_dict:
                missing_drugs.add(drug_name)
                continue
            
            if rna_name not in rna_dict:
                missing_rnas.add(rna_name)
                continue
            
         
            sample = {
                'drug_name': drug_name,
                'rna_name': rna_name,
                'drug_graph': drug_dict[drug_name]['graph'],
                'rna_onehot': rna_dict[rna_name]['onehot'],
                'label': 1, 
                'drug_smiles': drug_dict[drug_name]['smiles'],
                'rna_sequence': rna_dict[rna_name]['sequence']
            }
            positive_samples.append(sample)
    
 
    else:  # dataset_type == 'lncRNA'
      
        # drug_name_mapping = {}
        # for drug_name in drug_dict.keys():
       
        #     drug_name_mapping[drug_name.upper()] = drug_name
        #     drug_name_mapping[drug_name.lower()] = drug_name
        #     drug_name_mapping[drug_name.strip()] = drug_name
        #     drug_name_mapping[drug_name.upper().strip()] = drug_name
        
        # for idx, row in pairs_df.iterrows():
        #     drug_name = row['Drug_Name']
        #     rna_name = row['ncRNA_Name']
        #     
        #    
        #     matched_drug_name = None
        #     if drug_name in drug_dict:
        #         matched_drug_name = drug_name
        #     elif drug_name.upper() in drug_name_mapping:
        #         matched_drug_name = drug_name_mapping[drug_name.upper()]
        #     elif drug_name.lower() in drug_name_mapping:
        #         matched_drug_name = drug_name_mapping[drug_name.lower()]
        #     elif drug_name.strip() in drug_name_mapping:
        #         matched_drug_name = drug_name_mapping[drug_name.strip()]
        #     
        #     if matched_drug_name is None:
        #         missing_drugs.add(drug_name)
        #         continue
        #     
        #     if rna_name not in rna_dict:
        #         missing_rnas.add(rna_name)
        #         continue
        #     
        #     
        #     sample = {
        #         'drug_name': drug_name,
        #         'rna_name': rna_name,
        #         'drug_graph': drug_dict[matched_drug_name]['graph'],
        #         'rna_onehot': rna_dict[rna_name]['onehot'],
        #         'label': 1, 
        #         'drug_smiles': drug_dict[matched_drug_name]['smiles'],
        #         'rna_sequence': rna_dict[rna_name]['sequence']
        #     }
        #     positive_samples.append(sample)
        pass
    
    print(f"Positive samples: {len(positive_samples)}")
    print(f"Missing drugs: {len(missing_drugs)}")
    print(f"Missing RNAs: {len(missing_rnas)}")
    
    if missing_drugs:
        print(f"Sample missing drugs: {list(missing_drugs)[:5]}")
    if missing_rnas:
        print(f"Sample missing RNAs: {list(missing_rnas)[:5]}")
    
    return positive_samples

def generate_negative_samples(positive_samples, drug_dict, rna_dict, ratio=1.0):
    
    print(f"Generating negative samples (ratio: {ratio})...")
    
    drug_names = list(drug_dict.keys())
    rna_names = list(rna_dict.keys())
    
 
    positive_pairs = set()
    for sample in positive_samples:
        pair = (sample['drug_name'], sample['rna_name'])
        positive_pairs.add(pair)
    
    negative_samples = []
    num_negatives = int(len(positive_samples) * ratio)
    
    attempts = 0
    max_attempts = num_negatives * 10 
    
    while len(negative_samples) < num_negatives and attempts < max_attempts:
        attempts += 1
        
       
        drug_name = np.random.choice(drug_names)
        rna_name = np.random.choice(rna_names)
        
        if (drug_name, rna_name) not in positive_pairs:
            sample = {
                'drug_name': drug_name,
                'rna_name': rna_name,
                'drug_graph': drug_dict[drug_name]['graph'],
                'rna_onehot': rna_dict[rna_name]['onehot'],
                'label': 0,  
                'drug_smiles': drug_dict[drug_name]['smiles'],
                'rna_sequence': rna_dict[rna_name]['sequence']
            }
            negative_samples.append(sample)
    
    print(f"Generated negative samples: {len(negative_samples)}")
    return negative_samples

def split_dataset(samples, train_ratio=0.8, val_ratio=0.1):
    
    print("Splitting dataset...")
    
    np.random.shuffle(samples)
    n_samples = len(samples)
    
    train_size = int(n_samples * train_ratio)
    val_size = int(n_samples * val_ratio)
    
    train_samples = samples[:train_size]
    val_samples = samples[train_size:train_size + val_size]
    test_samples = samples[train_size + val_size:]
    
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    return train_samples, val_samples, test_samples

def save_processed_data(drug_dict, rna_dict, train_samples, val_samples, test_samples, dataset_type='miRNA'):

    print("Saving processed data...")
    
  
    output_dir = f'data/{dataset_type.lower()}1'
    create_dir(output_dir)
    
  
    with open(f'{output_dir}/drugs.pkl', 'wb') as f:
        pickle.dump(drug_dict, f)
    
   
    rna_key = 'mirnas' if dataset_type == 'miRNA' else 'lncrnas'
    with open(f'{output_dir}/{rna_key}.pkl', 'wb') as f:
        pickle.dump(rna_dict, f)
    
  
    with open(f'{output_dir}/train.pkl', 'wb') as f:
        pickle.dump(train_samples, f)
    
    with open(f'{output_dir}/val.pkl', 'wb') as f:
        pickle.dump(val_samples, f)
    
    with open(f'{output_dir}/test.pkl', 'wb') as f:
        pickle.dump(test_samples, f)
    
  
    stats = {
        'num_drugs': len(drug_dict),
        f'num_{rna_key}': len(rna_dict),
        'num_train': len(train_samples),
        'num_val': len(val_samples),
        'num_test': len(test_samples),
        'num_positive_train': sum(1 for s in train_samples if s['label'] == 1),
        'num_negative_train': sum(1 for s in train_samples if s['label'] == 0),
    }

   
    stats_compat = dict(stats)
    stats_compat.update({
        'drugs': stats['num_drugs'],
        rna_key: stats[f'num_{rna_key}'],
        'train_samples': stats['num_train'],
        'val_samples': stats['num_val'],
        'test_samples': stats['num_test'],
    })

    with open(f'{output_dir}/stats.pkl', 'wb') as f:
        pickle.dump(stats_compat, f)
    
    print(f"Data saved to {output_dir}/")
    print(f"Statistics: {stats_compat}")

def main():

    print("=" * 60)
    print("Drug-ncRNA Interaction Data Preprocessing")
    print("=" * 60)
    
 
    dataset_type = 'miRNA'
    
 
    drugs_df, rna_df, pairs_df = load_dataset_files(dataset_type)
    if drugs_df is None or rna_df is None or pairs_df is None:
        print("Failed to load data files!")
        return
    
   
    drug_dict = process_drugs(drugs_df)
    if not drug_dict:
        print("No valid drugs found!")
        return
    
   
    rna_dict = process_rnas(rna_df, dataset_type)
    if not rna_dict:
        print(f"No valid {dataset_type}s found!")
        return
    

    positive_samples = create_interaction_dataset(pairs_df, drug_dict, rna_dict, dataset_type)
    if not positive_samples:
        print("No valid interactions found!")
        return
    
 
    negative_samples = generate_negative_samples(positive_samples, drug_dict, rna_dict, ratio=1.0)
    
   
    all_samples = positive_samples + negative_samples
    print(f"Total samples: {len(all_samples)}")
    
  
    train_samples, val_samples, test_samples = split_dataset(all_samples)
    
   
    save_processed_data(drug_dict, rna_dict, train_samples, val_samples, test_samples, dataset_type)
    
    print("=" * 60)
    print("Data preprocessing completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
