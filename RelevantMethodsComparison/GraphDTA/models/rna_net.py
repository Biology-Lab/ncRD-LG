#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Specialized model for Drug-miRNA interaction prediction
Models specifically designed for drug-miRNA interaction prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_max_pool, global_mean_pool
from torch.nn import BatchNorm1d

class DrugGNN(nn.Module):
    """Drug graph neural network module - optimized version"""
    
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=256, dropout=0.2):
        super(DrugGNN, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv4 = GCNConv(hidden_dim * 2, output_dim)
        
        self.bn1 = BatchNorm1d(hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim * 2)
        self.bn4 = BatchNorm1d(output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.residual = nn.Linear(input_dim, output_dim)
        
    def forward(self, x, edge_index, batch):
        x_input = x
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        x = self.relu(x)
        
        # Global pooling
        x = global_max_pool(x, batch)
        
        residual = global_max_pool(x_input, batch)
        residual = self.residual(residual)
        x = x + residual
        
        return x

class RNACNN(nn.Module):
    """miRNA convolutional neural network module - optimized version"""
    
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=256, seq_len=50, dropout=0.2):
        super(RNACNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        
        self.fusion_conv = nn.Conv1d(hidden_dim * 3, hidden_dim * 2, kernel_size=1)
        self.conv4 = nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1)
        
        self.attention = nn.MultiheadAttention(hidden_dim * 4, num_heads=8, dropout=dropout)
        
        self.pool = nn.MaxPool1d(2)
        self.adaptive_pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc1 = nn.Linear(hidden_dim * 4, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim * 4)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        
        x1 = self.conv1(x)
        x1 = self.relu(x1)
        
        x2 = self.conv2(x)
        x2 = self.relu(x2)
        
        x3 = self.conv3(x)
        x3 = self.relu(x3)
        
        x = torch.cat([x1, x2, x3], dim=1)  # [batch_size, hidden_dim * 3, seq_len]
        x = self.fusion_conv(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv4(x)
        x = self.relu(x)
        
        batch_size, channels, seq_len = x.shape
        x_att = x.transpose(1, 2)  # [batch_size, seq_len, channels]
        x_att, _ = self.attention(x_att, x_att, x_att)
        x_att = x_att.transpose(1, 2)  # [batch_size, channels, seq_len]
        
        x = x + x_att
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)
        
        # Global pooling
        x = self.adaptive_pool(x).squeeze(-1)  # [batch_size, hidden_dim * 4]
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return x

class AttentionFusion(nn.Module):
    """Attention fusion module - optimized version"""
    
    def __init__(self, drug_dim=256, mirna_dim=256, hidden_dim=128, dropout=0.2):
        super(AttentionFusion, self).__init__()
        
        self.drug_proj = nn.Linear(drug_dim, hidden_dim)
        self.mirna_proj = nn.Linear(mirna_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, drug_feat, mirna_feat):
        drug_proj = self.drug_proj(drug_feat)  # [batch_size, hidden_dim]
        mirna_proj = self.mirna_proj(mirna_feat)  # [batch_size, hidden_dim]
        
        drug_attended, _ = self.attention(
            drug_proj.unsqueeze(1),  # [batch_size, 1, hidden_dim]
            drug_proj.unsqueeze(1),  # [batch_size, 1, hidden_dim]
            drug_proj.unsqueeze(1)   # [batch_size, 1, hidden_dim]
        )
        drug_attended = drug_attended.squeeze(1)  # [batch_size, hidden_dim]
        
        drug_attended = self.layer_norm1(drug_attended + drug_proj)
        
        drug_ff = self.feed_forward(drug_attended)
        drug_attended = self.layer_norm2(drug_attended + drug_ff)
        
        cross_attended, _ = self.attention(
            drug_attended.unsqueeze(1),  # [batch_size, 1, hidden_dim]
            mirna_proj.unsqueeze(1),     # [batch_size, 1, hidden_dim]
            mirna_proj.unsqueeze(1)      # [batch_size, 1, hidden_dim]
        )
        cross_attended = cross_attended.squeeze(1)  # [batch_size, hidden_dim]
        
        fused = torch.cat([cross_attended, mirna_proj], dim=1)  # [batch_size, hidden_dim * 2]
        fused = self.fusion(fused)
        
        return fused

class DrugRNANet(nn.Module):
    """Drug-miRNA interaction prediction network - simplified optimized version"""
    
    def __init__(self, n_output=2, drug_input_dim=4, mirna_input_dim=5, 
                 hidden_dim=128, dropout=0.2):
        super(DrugRNANet, self).__init__()
        
        self.drug_conv1 = GCNConv(drug_input_dim, hidden_dim)
        self.drug_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.drug_conv3 = GCNConv(hidden_dim, hidden_dim)
        self.drug_conv4 = GCNConv(hidden_dim, hidden_dim)
        
        self.drug_residual = nn.Linear(drug_input_dim, hidden_dim)
        
        self.drug_fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.mirna_conv1 = nn.Conv1d(mirna_input_dim, hidden_dim, kernel_size=3, padding=1)
        self.mirna_conv2 = nn.Conv1d(mirna_input_dim, hidden_dim, kernel_size=5, padding=2)
        self.mirna_conv3 = nn.Conv1d(mirna_input_dim, hidden_dim, kernel_size=7, padding=3)
        
        self.mirna_residual = nn.Conv1d(mirna_input_dim, hidden_dim, kernel_size=1)
        
        self.mirna_fusion = nn.Conv1d(hidden_dim * 3, hidden_dim, kernel_size=1)
        self.mirna_pool = nn.AdaptiveMaxPool1d(1)
        self.mirna_fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.cross_attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_output)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.drug_conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.drug_conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.drug_conv3(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.drug_conv4(x, edge_index)
        x = self.relu(x)
        
        # Global pooling
        x = global_max_pool(x, batch)
        residual = global_max_pool(x_input, batch)
        residual = self.drug_residual(residual)
        x = self.drug_fc(x)
        x = self.dropout(x)
        
        target = data.target.transpose(1, 2)
        
        target1 = self.mirna_conv1(target)
        target1 = self.relu(target1)
        
        target2 = self.mirna_conv2(target)
        target2 = self.relu(target2)
        
        target3 = self.mirna_conv3(target)
        target3 = self.relu(target3)
        
        target = torch.cat([target1, target2, target3], dim=1)
        target = self.mirna_fusion(target)
        target = self.relu(target)
        
        # Global pooling
        target = self.mirna_pool(target).squeeze(-1)
        
        target_residual = self.mirna_residual(data.target.transpose(1, 2))
        target_residual = self.mirna_pool(target_residual).squeeze(-1)
        target = target + target_residual
        
        target = self.mirna_fc(target)
        target = self.dropout(target)
        
        x_att = x.unsqueeze(1)
        target_att = target.unsqueeze(1)
        attended, _ = self.cross_attention(x_att, target_att, target_att)
        attended = attended.squeeze(1)
        
        attended = self.layer_norm(attended)
        
        fused = torch.cat([attended, target], dim=1)
        fused = self.fusion(fused)
        
        output = self.classifier(fused)
        
        return output

class DrugRNANet_Simple(nn.Module):
    """Simplified drug-miRNA network - optimized version"""
    
    def __init__(self, n_output=2, drug_input_dim=4, mirna_input_dim=5, 
                 hidden_dim=128, dropout=0.2):
        super(DrugRNANet_Simple, self).__init__()
        
        self.drug_conv1 = GCNConv(drug_input_dim, hidden_dim)
        self.drug_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.drug_conv3 = GCNConv(hidden_dim, hidden_dim)
        self.drug_fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.mirna_conv1 = nn.Conv1d(mirna_input_dim, hidden_dim, kernel_size=3, padding=1)
        self.mirna_conv2 = nn.Conv1d(mirna_input_dim, hidden_dim, kernel_size=5, padding=2)
        self.mirna_fusion = nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.mirna_pool = nn.AdaptiveMaxPool1d(1)
        self.mirna_fc = nn.Linear(hidden_dim, hidden_dim)
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_output)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.drug_conv1(x, edge_index)
        x = self.relu(x)
        x = self.drug_conv2(x, edge_index)
        x = self.relu(x)
        x = self.drug_conv3(x, edge_index)
        x = self.relu(x)
        x = global_max_pool(x, batch)
        x = self.drug_fc(x)
        x = self.dropout(x)
        
        target = data.target.transpose(1, 2)
        target1 = self.mirna_conv1(target)
        target1 = self.relu(target1)
        
        target2 = self.mirna_conv2(target)
        target2 = self.relu(target2)
        
        target = torch.cat([target1, target2], dim=1)
        target = self.mirna_fusion(target)
        target = self.relu(target)
        target = self.mirna_pool(target).squeeze(-1)
        target = self.mirna_fc(target)
        target = self.dropout(target)
        
        x_att = x.unsqueeze(1)
        target_att = target.unsqueeze(1)
        attended, _ = self.attention(x_att, target_att, target_att)
        attended = attended.squeeze(1)
        
        fused = torch.cat([attended, target], dim=1)
        fused = self.fusion(fused)
        
        output = self.classifier(fused)
        
        return output

class DrugRNANet_Ensemble(nn.Module):
    """Ensemble drug-miRNA network"""
    
    def __init__(self, n_output=2, drug_input_dim=4, mirna_input_dim=5, 
                 hidden_dim=64, dropout=0.2):
        super(DrugRNANet_Ensemble, self).__init__()
        
        self.model1 = DrugRNANet_Simple(n_output, drug_input_dim, mirna_input_dim, hidden_dim, dropout)
        self.model2 = DrugRNANet_Simple(n_output, drug_input_dim, mirna_input_dim, hidden_dim, dropout)
        self.model3 = DrugRNANet_Simple(n_output, drug_input_dim, mirna_input_dim, hidden_dim, dropout)
        
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        
    def forward(self, data):
        out1 = self.model1(data)
        out2 = self.model2(data)
        out3 = self.model3(data)
        
        weights = F.softmax(self.ensemble_weights, dim=0)
        output = weights[0] * out1 + weights[1] * out2 + weights[2] * out3
        
        return output

def test_mirna_net1():
    """Test miRNA network model"""
    print("Testing miRNA Net1 model...")
    
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
    
    # Test different models
    models = {
        'Simple': DrugmiRNANet1_Simple(),
        'Full': DrugmiRNANet1(),
        'Ensemble': DrugmiRNANet1_Ensemble()
    }
    
    for name, model in models.items():
        output = model(batch_data)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name} model - Output shape: {output.shape}, Parameters: {params}")
    
    return models

if __name__ == "__main__":
    test_mirna_net1()



