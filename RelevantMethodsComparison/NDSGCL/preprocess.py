#!/usr/bin/env python3
"""
mi dataset preprocessor
Convert mi dataset to the format expected by original code
"""

import pandas as pd
import numpy as np
import os

def load_mi_data():
    """Load mi dataset"""
    print("Loading mi dataset...")
    
    # Load interaction data
    pair_data = pd.read_csv("./mi/pair.csv")
    print(f"Interaction data: {len(pair_data)} records")
    
    # Load drug features
    drug_features = pd.read_csv("./mi/mi_drug_features.csv")
    print(f"Drug features: {drug_features.shape}")
    
    # Load miRNA features
    rna_features = pd.read_csv("./mi/head_pooled.csv")
    print(f"miRNA features: {rna_features.shape}")
    
    return pair_data, drug_features, rna_features

def create_id_mapping(drug_features, rna_features):
    """Create name to ID mapping"""
    # Drug name to ID mapping (starting from 1, consistent with original format)
    drug_names = drug_features.iloc[:, 0].tolist()  # First column is drug name
    drug_name_to_id = {name: idx+1 for idx, name in enumerate(drug_names)}  # Start from 1

    # miRNA name to ID mapping (starting from 1, consistent with original format)
    rna_names = rna_features.iloc[:, 0].tolist()    # First column is miRNA name
    rna_name_to_id = {name: idx+1 for idx, name in enumerate(rna_names)}  # Start from 1

    print(f"Drug count: {len(drug_name_to_id)}")
    print(f"miRNA count: {len(rna_name_to_id)}")
    print(f"Drug ID range: 1-{len(drug_name_to_id)}")
    print(f"miRNA ID range: 1-{len(rna_name_to_id)}")
    
    return drug_name_to_id, rna_name_to_id

def filter_valid_pairs(pair_data, drug_name_to_id, rna_name_to_id):
    """Filter valid interactions"""
    valid_pairs = []
    for _, row in pair_data.iterrows():
        rna_name = row['ncRNA_Name']
        drug_name = row['Drug_Name']
        
        if rna_name in rna_name_to_id and drug_name in drug_name_to_id:
            rna_id = rna_name_to_id[rna_name]
            drug_id = drug_name_to_id[drug_name]
            valid_pairs.append([rna_id, drug_id])
    
    print(f"Valid interactions: {len(valid_pairs)} records")
    return valid_pairs

def create_index_files(drug_name_to_id, rna_name_to_id, output_dir):
    """Create index files"""
    # Drug index file
    with open(f"{output_dir}/drug_name_index.txt", 'w') as f:
        for name, idx in drug_name_to_id.items():
            f.write(f"{name} {idx}\n")
    
    # miRNA index file
    with open(f"{output_dir}/rna_name_index.txt", 'w') as f:
        for name, idx in rna_name_to_id.items():
            f.write(f"{name} {idx}\n")

def create_cv_splits(valid_pairs, output_dir):
    """Create 5-fold cross validation splits"""
    np.random.seed(42)
    indices = np.random.permutation(len(valid_pairs))
    fold_size = len(valid_pairs) // 5
    
    for i in range(5):
        # Test set
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < 4 else len(valid_pairs)
        test_indices = indices[test_start:test_end]
        train_indices = np.concatenate([indices[:test_start], indices[test_end:]])
        
        # Write training file (format: miRNA_id drug_id)
        with open(f"{output_dir}/train_{i}.txt", 'w') as f:
            for idx in train_indices:
                rna_id, drug_id = valid_pairs[idx]
                f.write(f"{rna_id} {drug_id}\n")
        
        # Write test file (format: miRNA_id drug_id)
        with open(f"{output_dir}/test_{i}.txt", 'w') as f:
            for idx in test_indices:
                rna_id, drug_id = valid_pairs[idx]
                f.write(f"{rna_id} {drug_id}\n")
        
        print(f"Fold {i}: train {len(train_indices)} records, test {len(test_indices)} records")

def main():
    """Main function"""
    print("=== mi dataset preprocessor ===")
    
    # Create output directory
    output_dir = "./dataset/miRNA_drug_not_Mutation_all_mi/"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    pair_data, drug_features, rna_features = load_mi_data()
    
    # 2. Create ID mapping
    drug_name_to_id, rna_name_to_id = create_id_mapping(drug_features, rna_features)
    
    # 3. Filter valid interactions
    valid_pairs = filter_valid_pairs(pair_data, drug_name_to_id, rna_name_to_id)
    
    # 4. Create index files
    create_index_files(drug_name_to_id, rna_name_to_id, output_dir)
    
    # 5. Create 5-fold cross validation splits
    create_cv_splits(valid_pairs, output_dir)
    
    print(f"\nDataset files created: {output_dir}")
    print("Data preprocessing completed!")

if __name__ == "__main__":
    main()
