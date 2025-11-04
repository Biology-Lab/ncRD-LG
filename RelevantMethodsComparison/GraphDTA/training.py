#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified training script for Drug-ncRNA interaction prediction
Unified training script supporting both miRNA and lncRNA datasets
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Import models (unified naming)
from models.gcn import GCNNet_Simple
from models.gat import GATNet_Simple
from models.rna_net import DrugRNANet_Simple, DrugRNANet, DrugRNANet_Ensemble

# Import utility functions
from utils import load_processed_data, get_data_loaders, calculate_class_weights, print_data_statistics

class Trainer:
    """Unified trainer class supporting both miRNA and lncRNA"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, device='cuda', lr=0.001, epochs=50, dataset_type='miRNA'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.dataset_type = dataset_type
        
        # Loss function and optimizer - use class weights to handle imbalance
        class_weights = self._calculate_class_weights()
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        # Use cosine annealing learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs//2, eta_min=1e-6)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.val_aucs = []
        self.val_auprs = []
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_val_auc = 0.0
        self.best_val_aupr = 0.0
        self.best_epoch = 0
        self.patience = 10
        self.patience_counter = 0
        self.best_model_state = None
    
    def _calculate_class_weights(self):
        """Calculate class weights"""
        # miRNA version calculation logic (main code)
        if self.dataset_type == 'miRNA':
            # Count class distribution in training set
            labels = []
            for _, _, batch_labels in self.train_loader:
                labels.extend(batch_labels.numpy())
            
            labels = np.array(labels)
            class_counts = np.bincount(labels)
            total_samples = len(labels)
            
            # Calculate weights: total_samples / (num_classes * class_samples)
            weights = []
            for count in class_counts:
                if count > 0:
                    weight = total_samples / (len(class_counts) * count)
                    weights.append(weight)
                else:
                    weights.append(1.0)
            
            weights = torch.FloatTensor(weights).to(self.device)
            print(f"Class weights: {weights}")
            return weights
        
        # lncRNA version calculation logic (preserved as comments)
        else:  # dataset_type == 'lncRNA'
            # Calculate class distribution from training data
            # all_labels = []
            # for batch in self.train_loader:
            #     _, _, labels = batch
            #     all_labels.extend(labels.numpy())
            # 
            # unique_labels, counts = np.unique(all_labels, return_counts=True)
            # total_samples = len(all_labels)
            # 
            # Calculate weights
            # weights = []
            # for label in unique_labels:
            #     weight = total_samples / (len(unique_labels) * counts[label])
            #     weights.append(weight)
            # 
            # Convert to tensor
            # weights = torch.FloatTensor(weights).to(self.device)
            # print(f"Class weights: {weights}")
            # return weights
            pass
        
    def train_epoch(self):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # For monitoring prediction distribution
        all_predictions = []
        all_labels = []
        
        # miRNA training logic (main code)
        if self.dataset_type == 'miRNA':
            for batch_idx, (drug_batch, rna_onehot, labels) in enumerate(self.train_loader):
                # Move data to device
                drug_batch = drug_batch.to(self.device)
                rna_onehot = rna_onehot.to(self.device)
                labels = labels.to(self.device)
                
                # Create data object - add RNA features to drug_batch
                drug_batch.target = rna_onehot
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(drug_batch)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Collect predictions for analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # lncRNA training logic (preserved as comments)
        else:  # dataset_type == 'lncRNA'
            # for batch_idx, (drug_batch, lncrna_onehots, labels) in enumerate(self.train_loader):
            #     # Move to device
            #     drug_batch = drug_batch.to(self.device)
            #     labels = labels.to(self.device)
            #     
            #     # Handle variable-length lncRNA sequences - use padding to unify length
            #     max_len = max(onehot.size(0) for onehot in lncrna_onehots)
            #     padded_lncrna = []
            #     for onehot in lncrna_onehots:
            #         if onehot.size(0) < max_len:
            #             # Use zero padding
            #             padding = torch.zeros(max_len - onehot.size(0), onehot.size(1))
            #             padded = torch.cat([onehot, padding], dim=0)
            #         else:
            #             padded = onehot
            #         padded_lncrna.append(padded)
            #     
            #     # Convert to batch tensor
            #     lncrna_batch = torch.stack(padded_lncrna).to(self.device)
            #     
            #     # Forward pass
            #     self.optimizer.zero_grad()
            #     outputs = self.model(drug_batch, lncrna_batch)
            #     loss = self.criterion(outputs, labels)
            #     
            #     # Backward pass
            #     loss.backward()
            #     self.optimizer.step()
            #     
            #     # Statistics
            #     total_loss += loss.item()
            #     _, predicted = torch.max(outputs.data, 1)
            #     total += labels.size(0)
            #     correct += (predicted == labels).sum().item()
            pass
        
        # Calculate accuracy
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss, accuracy
    
    def validate_epoch(self):
        """Validate one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            # miRNA validation logic (main code)
            if self.dataset_type == 'miRNA':
                for drug_batch, rna_onehot, labels in self.val_loader:
                    drug_batch = drug_batch.to(self.device)
                    rna_onehot = rna_onehot.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Create data object
                    drug_batch.target = rna_onehot
                    
                    # Forward pass
                    outputs = self.model(drug_batch)
                    loss = self.criterion(outputs, labels)
                    
                    # Statistics
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Collect results
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            
            # lncRNA validation logic (preserved as comments)
            else:  # dataset_type == 'lncRNA'
                # for drug_batch, lncrna_onehots, labels in self.val_loader:
                #     drug_batch = drug_batch.to(self.device)
                #     labels = labels.to(self.device)
                #     
                #     # Handle variable-length lncRNA sequences
                #     max_len = max(onehot.size(0) for onehot in lncrna_onehots)
                #     padded_lncrna = []
                #     for onehot in lncrna_onehots:
                #         if onehot.size(0) < max_len:
                #             padding = torch.zeros(max_len - onehot.size(0), onehot.size(1))
                #             padded = torch.cat([onehot, padding], dim=0)
                #         else:
                #             padded = onehot
                #         padded_lncrna.append(padded)
                #     
                #     lncrna_batch = torch.stack(padded_lncrna).to(self.device)
                #     
                #     # Forward pass
                #     outputs = self.model(drug_batch, lncrna_batch)
                #     loss = self.criterion(outputs, labels)
                #     
                #     # Statistics
                #     total_loss += loss.item()
                #     _, predicted = torch.max(outputs.data, 1)
                #     total += labels.size(0)
                #     correct += (predicted == labels).sum().item()
                #     
                #     # Collect results
                #     all_predictions.extend(predicted.cpu().numpy())
                #     all_labels.extend(labels.cpu().numpy())
                #     all_probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                pass
        
        # Calculate metrics
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate AUC and AUPR
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
            aupr = average_precision_score(all_labels, all_probabilities)
        except:
            auc = 0.0
            aupr = 0.0
        
        return avg_loss, accuracy, auc, aupr
    
    def train(self):
        """Complete training process"""
        print(f"Starting training for {self.dataset_type} dataset...")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.epochs}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Training
            train_loss, train_acc = self.train_epoch()
            
            # Validation
            val_loss, val_acc, val_auc, val_aupr = self.validate_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            self.val_aucs.append(val_auc)
            self.val_auprs.append(val_aupr)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_auc = val_auc
                self.best_val_aupr = val_aupr
                self.best_epoch = epoch
                self.patience_counter = 0
                self.best_model_state = self.model.state_dict().copy()
            else:
                self.patience_counter += 1
            
            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{self.epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}, Val AUPR: {val_aupr:.4f} - "
                  f"Time: {epoch_time:.2f}s")
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model from epoch {self.best_epoch+1}")
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation AUC: {self.best_val_auc:.4f}")
        print(f"Best validation AUPR: {self.best_val_aupr:.4f}")
    
    def test(self):
        """Test model"""
        print("Testing model...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            # miRNA test logic (main code)
            if self.dataset_type == 'miRNA':
                for drug_batch, rna_onehot, labels in self.test_loader:
                    drug_batch = drug_batch.to(self.device)
                    rna_onehot = rna_onehot.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Create data object
                    drug_batch.target = rna_onehot
                    
                    # Forward pass
                    outputs = self.model(drug_batch)
                    
                    # Collect results
                    _, predicted = torch.max(outputs.data, 1)
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
            
            # lncRNA test logic (preserved as comments)
            else:  # dataset_type == 'lncRNA'
                # for drug_batch, lncrna_onehots, labels in self.test_loader:
                #     drug_batch = drug_batch.to(self.device)
                #     labels = labels.to(self.device)
                #     
                #     # Handle variable-length lncRNA sequences
                #     max_len = max(onehot.size(0) for onehot in lncrna_onehots)
                #     padded_lncrna = []
                #     for onehot in lncrna_onehots:
                #         if onehot.size(0) < max_len:
                #             padding = torch.zeros(max_len - onehot.size(0), onehot.size(1))
                #             padded = torch.cat([onehot, padding], dim=0)
                #         else:
                #             padded = onehot
                #         padded_lncrna.append(padded)
                #     
                #     lncrna_batch = torch.stack(padded_lncrna).to(self.device)
                #     
                #     # Forward pass
                #     outputs = self.model(drug_batch, lncrna_batch)
                #     
                #     # Collect results
                #     _, predicted = torch.max(outputs.data, 1)
                #     all_predictions.extend(predicted.cpu().numpy())
                #     all_labels.extend(labels.cpu().numpy())
                #     all_probabilities.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                pass
        
        # Calculate test metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
            aupr = average_precision_score(all_labels, all_probabilities)
        except:
            auc = 0.0
            aupr = 0.0
        
        print(f"Test Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"AUPR: {aupr:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'aupr': aupr
        }
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        axes[0, 1].plot(self.train_accs, label='Train Acc')
        axes[0, 1].plot(self.val_accs, label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC curves
        axes[1, 0].plot(self.val_aucs, label='Val AUC')
        axes[1, 0].set_title('Validation AUC')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # AUPR curves
        axes[1, 1].plot(self.val_auprs, label='Val AUPR')
        axes[1, 1].set_title('Validation AUPR')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUPR')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.show()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Drug-ncRNA Interaction Prediction Training')
    parser.add_argument('--dataset_type', type=str, default='miRNA', choices=['miRNA', 'lncRNA'], 
                       help='Dataset type: miRNA or lncRNA')
    parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN', 'GAT', 'RNA_Net'], 
                       help='Model type')
    parser.add_argument('--data_dir', type=str, default='data/mirna1', 
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, 
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device to use')
    parser.add_argument('--save_model', action='store_true', 
                       help='Save the best model')
    
    args = parser.parse_args()
    
    # Adjust data directory based on dataset type
    if args.dataset_type == 'lncRNA':
        args.data_dir = 'data/lnc1'
    
    print(f"Training {args.model_type} model on {args.dataset_type} dataset")
    print(f"Data directory: {args.data_dir}")
    
    # Load data
    drug_dict, rna_dict, train_samples, val_samples, test_samples = load_processed_data(args.data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        train_samples, val_samples, test_samples, 
        batch_size=args.batch_size
    )
    
    # Create model
    if args.model_type == 'GCN':
        model = GCNNet_Simple()
    elif args.model_type == 'GAT':
        model = GATNet_Simple()
    elif args.model_type == 'RNA_Net':
        model = DrugRNANet_Simple()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        lr=args.lr,
        epochs=args.epochs,
        dataset_type=args.dataset_type
    )
    
    # Training
    trainer.train()
    
    # Testing
    test_results = trainer.test()
    
    # Plot training history
    trainer.plot_training_history(f'training_history_{args.dataset_type}_{args.model_type}.png')
    
    # Save model
    if args.save_model:
        model_path = f'best_{args.dataset_type}_{args.model_type}_model.pth'
        torch.save(trainer.best_model_state, model_path)
        print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    main()
