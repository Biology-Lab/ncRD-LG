#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GCN model for Drug-miRNA interaction prediction
GCN-based drug-miRNA interaction prediction model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

class GCNNet(nn.Module):
    """
    GCN network for drug-miRNA interaction prediction
    Drug part: GCN processes molecular graphs
    miRNA part: CNN processes sequences
    """
    
    def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=4, num_features_xt=5, output_dim=128, dropout=0.2):
        super(GCNNet, self).__init__()
        
        # Drug graph branch (GCN)
        self.n_output = n_output
        self.conv1 = GCNConv(num_features_xd, num_features_xd)
        self.conv2 = GCNConv(num_features_xd, num_features_xd * 2)
        self.conv3 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.fc_g1 = nn.Linear(num_features_xd * 4, 1024)
        self.fc_g2 = nn.Linear(1024, output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # miRNA sequence branch (CNN)
        self.conv_xt_1 = nn.Conv1d(in_channels=num_features_xt, out_channels=n_filters, kernel_size=8)
        self.conv_xt_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.conv_xt_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.pool_xt = nn.MaxPool1d(3)
        self.fc_xt_1 = nn.Linear(n_filters * 4, output_dim)
        
        # Fusion layer
        self.fc1 = nn.Linear(2 * output_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, self.n_output)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, data):
        """
        Forward pass
        Args:
            data: Data containing drug graphs and miRNA sequences
        """
        # Drug graph processing
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN layers
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        
        # Global pooling
        x = global_max_pool(x, batch)
        
        # Fully connected layers
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_g2(x)
        x = self.dropout(x)
        
        # miRNA sequence processing
        target = data.target  # miRNA one-hot encoding [batch_size, seq_len, 5]
        target = target.transpose(1, 2)  # [batch_size, 5, seq_len]
        
        # CNN layers
        conv_xt = self.conv_xt_1(target)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.pool_xt(conv_xt)
        
        conv_xt = self.conv_xt_2(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.pool_xt(conv_xt)
        
        conv_xt = self.conv_xt_3(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.pool_xt(conv_xt)
        
        # Global average pooling
        conv_xt = torch.mean(conv_xt, dim=2)
        
        # Fully connected layers
        conv_xt = self.fc_xt_1(conv_xt)
        conv_xt = self.relu(conv_xt)
        conv_xt = self.dropout(conv_xt)
        
        # Feature fusion
        xc = torch.cat((x, conv_xt), 1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        
        return out

class GCNNet_Simple(nn.Module):
    """
    Simplified GCN network for quick testing
    """
    
    def __init__(self, n_output=2, num_features_xd=4, num_features_xt=5, hidden_dim=64, dropout=0.2):
        super(GCNNet_Simple, self).__init__()
        
        # Drug graph branch
        self.conv1 = GCNConv(num_features_xd, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.fc_g = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # miRNA sequence branch
        self.conv_xt = nn.Conv1d(num_features_xt, hidden_dim, kernel_size=3)
        self.pool_xt = nn.AdaptiveMaxPool1d(1)
        self.fc_xt = nn.Linear(hidden_dim, hidden_dim)
        
        # Fusion layer
        self.fc_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_output)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, data):
        # Drug graph processing
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = global_max_pool(x, batch)
        x = self.fc_g(x)
        x = self.dropout(x)
        
        # miRNA sequence processing
        target = data.target.transpose(1, 2)
        target = self.conv_xt(target)
        target = self.relu(target)
        target = self.pool_xt(target).squeeze(-1)
        target = self.fc_xt(target)
        target = self.dropout(target)
        
        # Feature fusion
        xc = torch.cat((x, target), 1)
        xc = self.fc_fusion(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        
        return out

def test_gcn1():
    """Test GCN model"""
    print("Testing GCN1 model...")
    
    # Create test data
    batch_size = 2
    num_atoms = 10
    seq_len = 50
    
    # Drug graph data
    x = torch.randn(num_atoms, 4)  # Atom features
    edge_index = torch.randint(0, num_atoms, (2, num_atoms * 2))  # Edge indices
    batch = torch.zeros(num_atoms, dtype=torch.long)  # Batch indices
    
    # miRNA sequence data
    target = torch.randn(batch_size, seq_len, 5)  # One-hot encoding
    
    # Create data objects
    from torch_geometric.data import Data, Batch
    data_list = []
    for i in range(batch_size):
        data = Data(x=x, edge_index=edge_index, target=target[i])
        data_list.append(data)
    
    batch_data = Batch.from_data_list(data_list)
    
    # Test model
    model = GCNNet1_Simple()
    output = model(batch_data)
    
    print(f"Input shape: {batch_data.x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model, output

if __name__ == "__main__":
    test_gcn1()

