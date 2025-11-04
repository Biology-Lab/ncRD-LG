#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GAT model for Drug-miRNA interaction prediction
GAT-based drug-miRNA interaction prediction model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool, global_mean_pool
from torch.nn import BatchNorm1d

class GATNet(nn.Module):
    """
    GAT network for drug-miRNA interaction prediction
    Drug part: GAT processes molecular graphs
    miRNA part: CNN processes sequences
    """
    
    def __init__(self, n_output=2, n_filters=32, embed_dim=128, num_features_xd=4, num_features_xt=5, output_dim=128, dropout=0.2):
        super(GATNet, self).__init__()
        
        # Drug graph branch (GAT)
        self.n_output = n_output
        self.gat1 = GATConv(num_features_xd, num_features_xd, heads=8, dropout=dropout)
        self.gat2 = GATConv(num_features_xd * 8, num_features_xd * 4, heads=4, dropout=dropout)
        self.gat3 = GATConv(num_features_xd * 4 * 4, output_dim, heads=1, dropout=dropout)
        
        self.fc_g1 = nn.Linear(output_dim, output_dim)
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
        self.elu = nn.ELU()
        
    def forward(self, data):
        """
        Forward pass
        Args:
            data: Data containing drug graphs and miRNA sequences
        """
        # Drug graph processing
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GAT layers
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.gat2(x, edge_index)
        x = self.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.gat3(x, edge_index)
        x = self.relu(x)
        
        # Global pooling
        x = global_max_pool(x, batch)
        
        # Fully connected layers
        x = self.fc_g1(x)
        x = self.relu(x)
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

class GATNet_Simple(nn.Module):
    """
    Simplified GAT network for quick testing
    """
    
    def __init__(self, n_output=2, num_features_xd=4, num_features_xt=5, hidden_dim=64, dropout=0.2):
        super(GATNet_Simple, self).__init__()
        
        # Drug graph branch
        self.gat1 = GATConv(num_features_xd, hidden_dim, heads=4, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * 4, hidden_dim * 2, heads=2, dropout=dropout)
        self.gat3 = GATConv(hidden_dim * 2 * 2, hidden_dim, heads=1, dropout=dropout)
        self.fc_g = nn.Linear(hidden_dim, hidden_dim)
        
        # miRNA sequence branch
        self.conv_xt = nn.Conv1d(num_features_xt, hidden_dim, kernel_size=3, padding=1)
        self.pool_xt = nn.AdaptiveMaxPool1d(1)
        self.fc_xt = nn.Linear(hidden_dim, hidden_dim)
        
        # Fusion layer
        self.fc_fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_output)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        
    def forward(self, data):
        # Drug graph processing
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.gat2(x, edge_index)
        x = self.elu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.gat3(x, edge_index)
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

def test_gat1():
    """Test GAT model"""
    print("Testing GAT1 model...")
    
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
    model = GATNet1_Simple()
    output = model(batch_data)
    
    print(f"Input shape: {batch_data.x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return model, output

if __name__ == "__main__":
    test_gat1()

