#!/usr/bin/env python3
"""
miRNA dataset training main program - optimized version 2
Implements staged training strategy:
- First 3 epochs: random negative sampling for initial convergence
- From epoch 4: hard negative mining based on model predictions
"""

import sys
import os
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import json
from datetime import datetime

from util import *
from NDSGCL import NDSGCL
from interaction_table import InteractionTable
from negative_manager import NegativeSampleManager

def get_training_strategy(epoch, hard_negative_start_epoch=4, model=None):
    """
    Determine training strategy based on current epoch
    
    Args:
        epoch: current epoch (starting from 0)
        hard_negative_start_epoch: epoch to start using hard negative mining
        model: model object to check if training is complete
        
    Returns:
        tuple: (strategy_name, negative_ratio)
    """
    if epoch < hard_negative_start_epoch - 1:  # First 3 epochs (0,1,2)
        return 'random', 1  # Random sampling, 1:1 ratio
    else:
        # Check if model is trained (miRNA_emb attribute created during training)
        if model is not None and hasattr(model, 'miRNA_emb'):
            return 'hard_negative', 2  # Hard negative mining, 1:2 ratio
        else:
            print(f"Model not fully trained, epoch {epoch+1} continues with random sampling")
            return 'random', 1  # Continue with random sampling

def train_one_epoch_with_strategy(model, training_data, strategy, negative_ratio, 
                                interaction_table, negative_manager, all_drugs):
    """
    Train one epoch with specified strategy
    
    Args:
        model: model object
        training_data: training data
        strategy: training strategy ('random' or 'hard_negative')
        negative_ratio: negative sample ratio
        interaction_table: interaction table
        negative_manager: negative sample manager
        all_drugs: list of all drugs
        
    Returns:
        float: average loss
    """
    print(f"Using strategy: {strategy}, negative ratio: {negative_ratio}")
    
    if strategy == 'random':
        # Random negative sampling strategy
        return model.train_one_epoch(training_data, negative_ratio, interaction_table)
    elif strategy == 'hard_negative':
        # Hard negative mining strategy
        return model.train_one_epoch_hard_negative(training_data, negative_ratio, 
                                                interaction_table, negative_manager, all_drugs)
    else:
        raise ValueError(f"Unknown training strategy: {strategy}")

def main():
    """Main training function"""
    print("=" * 60)
    print("NDSGCL miRNA dataset training program")
    print("=" * 60)
    
    # miRNA version main function logic (main code)
    # Load configuration
    config = load_config('config.conf')
    print(f"Configuration loaded: {config}")
    
    # Load data
    print("Loading miRNA data...")
    data = load_data(config)
    print(f"Data loading completed: {len(data.miRNA)} miRNAs, {len(data.drug)} drugs")
    
    # Create model
    model = NDSGCL(config, data)
    print("Model created")
    
    # Create interaction table
    interaction_table = InteractionTable(data)
    print("Interaction table created")
    
    # Create negative sample manager
    negative_manager = NegativeSampleManager(data)
    print("Negative sample manager created")
    
    # Prepare training data
    training_data = prepare_training_data(data)
    print(f"Training data prepared: {len(training_data)} samples")
    
    # Training parameters
    epochs = config.get('epochs', 50)
    hard_negative_start_epoch = config.get('hard_negative_start_epoch', 4)
    
    # Training loop
    print(f"Starting training, {epochs} epochs total...")
    print(f"First {hard_negative_start_epoch-1} epochs use random negative sampling")
    print(f"From epoch {hard_negative_start_epoch} start using hard negative mining")
    
    best_auc = 0
    best_aupr = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")
        
        # Determine training strategy
        strategy, negative_ratio = get_training_strategy(epoch, hard_negative_start_epoch, model)
        
        # Train one epoch
        start_time = time.time()
        avg_loss = train_one_epoch_with_strategy(
            model, training_data, strategy, negative_ratio, 
            interaction_table, negative_manager, list(data.drug.keys())
        )
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1} completed - Average loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("Evaluating model performance...")
            auc, aupr = evaluate_model(model, data)
            print(f"Current performance - AUC: {auc:.4f}, AUPR: {aupr:.4f}")
            
            if auc > best_auc:
                best_auc = auc
                best_aupr = aupr
                best_epoch = epoch + 1
                print(f"New best performance! AUC: {best_auc:.4f}, AUPR: {best_aupr:.4f}")
                
                # Save best model
                model.save_model(f'best_model_epoch_{epoch+1}.pth')
                print(f"Model saved: best_model_epoch_{epoch+1}.pth")
    
    print(f"\nTraining completed!")
    print(f"Best performance - Epoch {best_epoch}: AUC: {best_auc:.4f}, AUPR: {best_aupr:.4f}")
    
    # lncRNA version main function logic (commented for future activation)
    # def main_lnc():
    #     """lncRNA version main function"""
    #     print("=" * 60)
    #     print("NDSGCL lncRNA dataset training program")
    #     print("=" * 60)
    #     
    #     config = load_config('config_lnc.conf')
    #     print(f"Configuration loaded: {config}")
    #     
    #     print("Loading lncRNA data...")
    #     data = load_data_lnc(config)
    #     print(f"Data loading completed: {len(data.lncRNA)} lncRNAs, {len(data.drug)} drugs")
    #     
    #     model = NDSGCL_lnc(config, data)
    #     print("Model created")
    #     
    #     training_data = prepare_training_data_lnc(data)
    #     print(f"Training data prepared: {len(training_data)} samples")
    #     
    #     epochs = config.get('epochs', 50)
    #     
    #     print(f"Starting training, {epochs} epochs total...")
    #     
    #     best_auc = 0
    #     best_aupr = 0
    #     best_epoch = 0
    #     
    #     for epoch in range(epochs):
    #         print(f"\n--- Epoch {epoch+1}/{epochs} ---")
    #         
    #         start_time = time.time()
    #         avg_loss = model.train_one_epoch(training_data, 1, interaction_table)
    #         epoch_time = time.time() - start_time
    #         
    #         print(f"Epoch {epoch+1} completed - Average loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
    #         
    #         if (epoch + 1) % 5 == 0:
    #             print("Evaluating model performance...")
    #             auc, aupr = evaluate_model_safe(model, data)
    #             print(f"Current performance - AUC: {auc:.4f}, AUPR: {aupr:.4f}")
    #             
    #             if auc > best_auc:
    #                 best_auc = auc
    #                 best_aupr = aupr
    #                 best_epoch = epoch + 1
    #                 print(f"New best performance! AUC: {best_auc:.4f}, AUPR: {best_aupr:.4f}")
    #                 
    #                 model.save_model(f'best_lnc_model_epoch_{epoch+1}.pth')
    #                 print(f"Model saved: best_lnc_model_epoch_{epoch+1}.pth")
    #     
    #     print(f"\nlncRNA training completed!")
    #     print(f"Best performance - Epoch {best_epoch}: AUC: {best_auc:.4f}, AUPR: {best_aupr:.4f}")

def evaluate_model(model, data):
    """Evaluate model performance"""
    # miRNA version model evaluation logic (main code)
    print("Evaluating miRNA model performance...")
    
    # Get test data
    test_data = get_test_data(data)
    print(f"Test data: {len(test_data)} samples")
    
    # Generate negative samples
    all_drugs = list(data.drug.keys())
    negative_pairs = []
    
    for pair in test_data:
        lncRNA_id, drug_id, _ = pair
        # Randomly select different drug as negative sample
        neg_drug_id = np.random.choice(all_drugs)
        while neg_drug_id == drug_id:
            neg_drug_id = np.random.choice(all_drugs)
        negative_pairs.append((lncRNA_id, neg_drug_id))
    
    # Combine positive and negative samples
    all_pairs = test_data + negative_pairs
    all_labels = [1] * len(test_data) + [0] * len(negative_pairs)
    
    # Predict
    predictions = []
    for pair in all_pairs:
        lncRNA_id, drug_id = pair
        pred = model.predict(lncRNA_id, drug_id)
        predictions.append(pred)
    
    # Calculate metrics
    auc = roc_auc_score(all_labels, predictions)
    aupr = average_precision_score(all_labels, predictions)
    
    return auc, aupr
    
    # lncRNA version model evaluation logic (commented for future activation)
    # def evaluate_model_lnc(model, data):
    #     """lncRNA version model evaluation"""
    #     print("Evaluating lncRNA model performance...")
    #     
    #     test_data = get_test_data_lnc(data)
    #     print(f"Test data: {len(test_data)} samples")
    #     
    #     all_drugs = list(data.drug.keys())
    #     negative_pairs = []
    #     
    #     for pair in test_data:
    #         lncRNA_id, drug_id = _ = pair
    #         neg_drug_id = np.random.choice(all_drugs)
    #         while neg_drug_id == drug_id:
    #             neg_drug_id = np.random.choice(all_drugs)
    #         negative_pairs.append((lncRNA_id, neg_drug_id))
    #     
    #     all_pairs = test_data + negative_pairs
    #     all_labels = [1] * len(test_data) + [0] * len(negative_pairs)
    #     
    #     predictions = []
    #     for pair in all_pairs:
    #         lncRNA_id, drug_id = pair
    #         pred = model.predict(lncRNA_id, drug_id)
    #         predictions.append(pred)
    #     
    #     auc = roc_auc_score(all_labels, predictions)
    #     aupr = average_precision_score(all_labels, predictions)
    #     
    #     return auc, aupr

if __name__ == '__main__':
    main()