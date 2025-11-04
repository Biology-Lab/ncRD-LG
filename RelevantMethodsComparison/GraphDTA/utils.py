#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified utility functions for Drug-ncRNA interaction prediction
Unified utility functions supporting both miRNA and lncRNA datasets
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Batch
from rdkit import Chem
import pickle
import os

def load_processed_data(data_dir='data/mirna1', dataset_type='miRNA'):
    """Load preprocessed data, supporting both miRNA and lncRNA"""
    print(f"Loading processed {dataset_type} data from {data_dir}...")
    
    # miRNA data loading logic (main code)
    if dataset_type == 'miRNA':
        # Check if files exist
        required_files = ['drugs.pkl', 'mirnas.pkl', 'train.pkl', 'val.pkl', 'test.pkl']
        for file in required_files:
            if not os.path.exists(f'{data_dir}/{file}'):
                raise FileNotFoundError(f"Required file not found: {data_dir}/{file}")
        
        # Load data
        with open(f'{data_dir}/drugs.pkl', 'rb') as f:
            drug_dict = pickle.load(f)
        
        with open(f'{data_dir}/mirnas.pkl', 'rb') as f:
            rna_dict = pickle.load(f)
        
        with open(f'{data_dir}/train.pkl', 'rb') as f:
            train_samples = pickle.load(f)
        
        with open(f'{data_dir}/val.pkl', 'rb') as f:
            val_samples = pickle.load(f)
        
        with open(f'{data_dir}/test.pkl', 'rb') as f:
            test_samples = pickle.load(f)
        
        # Load/build statistics, fill missing values with defaults to avoid KeyError
        stats = {}
        if os.path.exists(f'{data_dir}/stats.pkl'):
            with open(f'{data_dir}/stats.pkl', 'rb') as f:
                stats = pickle.load(f)

        # Default values for required statistics fields
        default_stats = {
            'drugs': len(drug_dict),
            'mirnas': len(rna_dict),
            'train_samples': len(train_samples),
            'val_samples': len(val_samples),
            'test_samples': len(test_samples),
        }
        # Only fill missing values, preserve existing values
        for key, value in default_stats.items():
            if key not in stats:
                stats[key] = value
        
        print(f"Loaded {len(drug_dict)} drugs, {len(rna_dict)} miRNAs")
        print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    
    # lncRNA data loading logic (preserved as comments)
    else:  # dataset_type == 'lncRNA'
        # Check if files exist
        # required_files = ['drugs.pkl', 'lncrnas.pkl', 'train.pkl', 'val.pkl', 'test.pkl']
        # for file in required_files:
        #     if not os.path.exists(f'{data_dir}/{file}'):
        #         raise FileNotFoundError(f"Required file not found: {data_dir}/{file}")
        # 
        # Load data
        # with open(f'{data_dir}/drugs.pkl', 'rb') as f:
        #     drug_dict = pickle.load(f)
        # 
        # with open(f'{data_dir}/lncrnas.pkl', 'rb') as f:
        #     lncrna_dict = pickle.load(f)
        # 
        # with open(f'{data_dir}/train.pkl', 'rb') as f:
        #     train_samples = pickle.load(f)
        # 
        # with open(f'{data_dir}/val.pkl', 'rb') as f:
        #     val_samples = pickle.load(f)
        # 
        # with open(f'{data_dir}/test.pkl', 'rb') as f:
        #     test_samples = pickle.load(f)
        # 
        # Load/build statistics, fill missing values with defaults to avoid KeyError
        # stats = {}
        # if os.path.exists(f'{data_dir}/stats.pkl'):
        #     with open(f'{data_dir}/stats.pkl', 'rb') as f:
        #         stats = pickle.load(f)
        # 
        # Default values for required statistics fields
        # default_stats = {
        #     'drugs': len(drug_dict),
        #     'lncrnas': len(lncrna_dict),
        #     'train_samples': len(train_samples),
        #     'val_samples': len(val_samples),
        #     'test_samples': len(test_samples),
        # }
        # Only fill missing values, preserve existing values
        # for key, value in default_stats.items():
        #     if key not in stats:
        #         stats[key] = value
        # 
        # print(f"Loaded {len(drug_dict)} drugs, {len(lncrna_dict)} lncRNAs")
        # print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
        pass
    
    return drug_dict, rna_dict, train_samples, val_samples, test_samples, stats

def get_data_loaders(train_samples, val_samples, test_samples, batch_size=32, shuffle=True, dataset_type='miRNA'):
    """Create data loaders, supporting both miRNA and lncRNA"""
    from torch.utils.data import DataLoader
    from torch_geometric.data import Batch
    
    # miRNA data loader logic (main code)
    if dataset_type == 'miRNA':
        # Create custom dataset class
        class GraphDataset(torch.utils.data.Dataset):
            def __init__(self, samples):
                self.samples = samples
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                sample = self.samples[idx]
                return sample['drug_graph'], sample['rna_onehot'], sample['label']
        
        # Create dataset
        train_dataset = GraphDataset(train_samples)
        val_dataset = GraphDataset(val_samples)
        test_dataset = GraphDataset(test_samples)
        
        # Custom batch processing function
        def collate_fn(batch):
            drug_graphs = [item[0] for item in batch]  # Extract all graph data
            rna_onehots = torch.stack([item[1] for item in batch])  # Stack RNA features
            labels = torch.tensor([item[2] for item in batch], dtype=torch.long)  # Create label tensor
            
            # Use Batch.from_data_list to process graph data
            drug_batch = Batch.from_data_list(drug_graphs)
            
            return drug_batch, rna_onehots, labels
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            collate_fn=collate_fn,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=collate_fn,
            num_workers=0
        )
    
    # lncRNA data loader logic (preserved as comments)
    else:  # dataset_type == 'lncRNA'
        # Create custom dataset class
        # class GraphDataset(torch.utils.data.Dataset):
        #     def __init__(self, samples):
        #         self.samples = samples
        #     
        #     def __len__(self):
        #         return len(self.samples)
        #     
        #     def __getitem__(self, idx):
        #         sample = self.samples[idx]
        #         return sample['drug_graph'], sample['lncrna_onehot'], sample['label']
        # 
        # Create dataset
        # train_dataset = GraphDataset(train_samples)
        # val_dataset = GraphDataset(val_samples)
        # test_dataset = GraphDataset(test_samples)
        # 
        # Custom batch processing function
        # def collate_fn(batch):
        #     drug_graphs = [item[0] for item in batch]  # Extract all graph data
        #     lncrna_onehots = [item[1] for item in batch]  # Keep as list, don't stack
        #     labels = torch.tensor([item[2] for item in batch], dtype=torch.long)  # Create label tensor
        #     
        #     # Use Batch.from_data_list to process graph data
        #     drug_batch = Batch.from_data_list(drug_graphs)
        #     
        #     return drug_batch, lncrna_onehots, labels
        # 
        # Create data loaders
        # train_loader = DataLoader(
        #     train_dataset, 
        #     batch_size=batch_size, 
        #     shuffle=shuffle, 
        #     collate_fn=collate_fn,
        #     num_workers=0
        # )
        # 
        # val_loader = DataLoader(
        #     val_dataset, 
        #     batch_size=batch_size, 
        #     shuffle=False, 
        #     collate_fn=collate_fn,
        #     num_workers=0
        # )
        # 
        # test_loader = DataLoader(
        #     test_dataset, 
        #     batch_size=batch_size, 
        #     shuffle=False, 
        #     collate_fn=collate_fn,
        #     num_workers=0
        # )
        pass
    
    return train_loader, val_loader, test_loader

def calculate_class_weights(train_loader, device='cuda'):
    """Calculate class weights"""
    all_labels = []
    for batch in train_loader:
        _, _, labels = batch
        all_labels.extend(labels.numpy())
    
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    total_samples = len(all_labels)
    
    # Calculate weights
    weights = []
    for label in unique_labels:
        weight = total_samples / (len(unique_labels) * counts[label])
        weights.append(weight)
    
    # Convert to tensor
    weights = torch.FloatTensor(weights).to(device)
    print(f"Class weights: {weights}")
    return weights

def print_data_statistics(drug_dict, rna_dict, train_samples, val_samples, test_samples, dataset_type='miRNA'):
    """Print data statistics"""
    print("=" * 50)
    print(f"{dataset_type} Dataset Statistics")
    print("=" * 50)
    
    # Basic statistics
    print(f"Drugs: {len(drug_dict)}")
    print(f"RNAs: {len(rna_dict)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")
    
    # Positive/negative sample statistics
    train_positive = sum(1 for s in train_samples if s['label'] == 1)
    train_negative = sum(1 for s in train_samples if s['label'] == 0)
    val_positive = sum(1 for s in val_samples if s['label'] == 1)
    val_negative = sum(1 for s in val_samples if s['label'] == 0)
    test_positive = sum(1 for s in test_samples if s['label'] == 1)
    test_negative = sum(1 for s in test_samples if s['label'] == 0)
    
    print(f"\nTrain - Positive: {train_positive}, Negative: {train_negative}")
    print(f"Validation - Positive: {val_positive}, Negative: {val_negative}")
    print(f"Test - Positive: {test_positive}, Negative: {test_negative}")
    
    # Sequence length statistics
    if dataset_type == 'miRNA':
        rna_lengths = [len(s['rna_sequence']) for s in train_samples]
    else:  # lncRNA
        rna_lengths = [len(s['rna_sequence']) for s in train_samples]
    
    print(f"\nRNA sequence length statistics:")
    print(f"Mean: {np.mean(rna_lengths):.2f}")
    print(f"Median: {np.median(rna_lengths):.2f}")
    print(f"Min: {np.min(rna_lengths)}")
    print(f"Max: {np.max(rna_lengths)}")
    
    # Molecular size statistics
    drug_sizes = [s['drug_graph'].x.size(0) for s in train_samples]
    print(f"\nDrug molecule size statistics:")
    print(f"Mean atoms: {np.mean(drug_sizes):.2f}")
    print(f"Median atoms: {np.median(drug_sizes):.2f}")
    print(f"Min atoms: {np.min(drug_sizes)}")
    print(f"Max atoms: {np.max(drug_sizes)}")
    
    print("=" * 50)

def save_model(model, path, epoch, val_acc, val_auc, val_aupr):
    """Save model"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': epoch,
        'val_acc': val_acc,
        'val_auc': val_auc,
        'val_aupr': val_aupr,
    }, path)
    print(f"Model saved to {path}")

def load_model(model, path):
    """Load model"""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    val_acc = checkpoint['val_acc']
    val_auc = checkpoint['val_auc']
    val_aupr = checkpoint['val_aupr']
    print(f"Model loaded from {path}")
    print(f"Epoch: {epoch}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val AUPR: {val_aupr:.4f}")
    return model, epoch, val_acc, val_auc, val_aupr

def set_seed(seed=42):
    """Set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    """Get available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def count_parameters(model):
    """Calculate model parameter count"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    return total_params, trainable_params

def create_output_dir(base_dir='results'):
    """Create output directory"""
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"{base_dir}/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    return output_dir

def save_results(results, output_dir):
    """Save results"""
    import json
    
    # Save as JSON
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save as CSV
    df = pd.DataFrame([results])
    df.to_csv(f"{output_dir}/results.csv", index=False)
    
    print(f"Results saved to {output_dir}")

def plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'], save_path=None):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_roc_curve(y_true, y_scores, save_path=None):
    """Plot ROC curve"""
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()
    
    return roc_auc

def plot_pr_curve(y_true, y_scores, save_path=None):
    """Plot PR curve"""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib.pyplot as plt
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'PR curve (AP = {avg_precision:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"PR curve saved to {save_path}")
    
    plt.show()
    
    return avg_precision
