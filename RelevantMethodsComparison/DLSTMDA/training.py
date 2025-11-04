import sys
import torch
import torch.nn as nn
import numpy as np
from numpy import interp
from cnn_gcnmulti import GCNNetmuti
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from torch_geometric.data import DataLoader
import os


def train(model, device, train_loader, optimizer, epoch, loss_fn, LOG_INTERVAL):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = (output >= 0.5).float()
        total += data.y.size(0)
        correct += pred.eq(data.y.view(-1, 1)).sum().item()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.4f}'.format(
                epoch,
                batch_idx * train_loader.batch_size,
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item(),
                100. * correct / total))
    
    return total_loss / len(train_loader), 100. * correct / total


def predicting(model, device, loader):
    model.eval()
    total_probs = []
    total_preds = []
    total_labels = []
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            probs = torch.sigmoid(output)
            preds = (probs >= 0.5).float()
            
            total_probs.extend(probs.cpu().numpy().flatten())
            total_preds.extend(preds.cpu().numpy().flatten())
            total_labels.extend(data.y.cpu().numpy().flatten())
    
    return total_labels, total_preds, total_probs


def main():
    """Main training function for miRNA dataset"""
    print('Starting miRNA training...')
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    train_loader, test_loader = load_data()
    print(f'Loaded {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples')
    
    # Create model
    model = GCNNetmuti().to(device)
    print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Training parameters
    epochs = 100
    LOG_INTERVAL = 10
    best_auc = 0
    
    # Training loop
    for epoch in range(epochs):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, loss_fn, LOG_INTERVAL)
        
        if epoch % 10 == 0:
            y_true, y_pred, y_probs = predicting(model, device, test_loader)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            auc_score = roc_auc_score(y_true, y_probs)
            
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Test - Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}')
            
            if auc_score > best_auc:
                best_auc = auc_score
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'New best model saved with AUC: {best_auc:.4f}')
    
    print(f'Training completed. Best AUC: {best_auc:.4f}')
    
    # lncRNA version main function (commented for future activation)
    # def main_lnc():
    #     """Main training function for lncRNA dataset"""
    #     print('Starting lncRNA training...')
    #     
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     print(f'Using device: {device}')
    #     
    #     train_loader, test_loader = load_data_lnc()
    #     print(f'Loaded {len(train_loader.dataset)} training samples, {len(test_loader.dataset)} test samples')
    #     
    #     model = GCNNetmuti1().to(device)
    #     print(f'Model created with {sum(p.numel() for p in model.parameters())} parameters')
    #     
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    #     loss_fn = nn.BCEWithLogitsLoss()
    #     
    #     epochs = 100
    #     LOG_INTERVAL = 10
    #     best_auc = 0
    #     
    #     for epoch in range(epochs):
    #         train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, loss_fn, LOG_INTERVAL)
    #         
    #         if epoch % 10 == 0:
    #             y_true, y_pred, y_probs = predicting(model, device, test_loader)
    #             
    #             accuracy = accuracy_score(y_true, y_pred)
    #             precision = precision_score(y_true, y_pred)
    #             recall = recall_score(y_true, y_pred)
    #             f1 = f1_score(y_true, y_pred)
    #             auc_score = roc_auc_score(y_true, y_probs)
    #             
    #             print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    #             print(f'Test - Acc: {accuracy:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}, AUC: {auc_score:.4f}')
    #             
    #             if auc_score > best_auc:
    #                 best_auc = auc_score
    #                 torch.save(model.state_dict(), 'best_lnc_model.pth')
    #                 print(f'New best model saved with AUC: {best_auc:.4f}')
    #     
    #     print(f'lncRNA training completed. Best AUC: {best_auc:.4f}')


if __name__ == '__main__':
    main()