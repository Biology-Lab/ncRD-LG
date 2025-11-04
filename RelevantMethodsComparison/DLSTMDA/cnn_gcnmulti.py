import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp

# GCN based model for miRNA dataset
class GCNNetmuti(torch.nn.Module):
    def __init__(self, n_output=1, n_filters=32, embed_dim=64, num_features_xd=78, num_features_smile=66, num_features_xt=25, output_dim=128, dropout=0.2):
        super(GCNNetmuti, self).__init__()

        # SMILES character CNN processing
        self.smile_embed = nn.Embedding(num_features_smile + 1, embed_dim)
        self.conv_xd_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xd_12 = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=3, padding=1)
        self.conv_xd_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xd_22 = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=2, padding=1)
        self.conv_xd_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=1, padding=1)
        self.conv_xd_32 = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=1, padding=1)
        self.fc_smiles = torch.nn.Linear(n_filters * 2, output_dim)

        # SMILES graph branch
        self.n_output = n_output
        self.gcnv1 = GCNConv(num_features_xd, num_features_xd*2)
        self.gcnv2 = GCNConv(num_features_xd*2, num_features_xd * 4)
        self.fc_g1 = torch.nn.Linear(num_features_xd*4, 1024)
        self.fc_g2 = torch.nn.Linear(1024, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Convolution layers for reducing dimensions after concatenation
        self.conv_reduce_smiles = nn.Conv1d(in_channels=output_dim*3, out_channels=output_dim, kernel_size=1)
        self.conv_reduce_xt = nn.Conv1d(in_channels=192, out_channels=output_dim, kernel_size=1)

        self.conv_transform = nn.Conv1d(in_channels=32, out_channels=128, kernel_size=1)

        # miRNA sequence branch (1D CNN) - main code
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_11 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=4, padding=2)
        self.conv_xt_12 = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=4, padding=2)

        self.conv_xt_21 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        self.conv_xt_22 = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=3, padding=1)

        self.conv_xt_31 = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        self.conv_xt_32 = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=2, padding=1)

        self.fc1_xt = nn.Linear(n_filters * 2, output_dim)

        # Combined layers
        self.fc1 = nn.Linear(output_dim * 2, 256)
        self.fc2 = nn.Linear(256, 128)
        self.out = nn.Linear(128, n_output)

        # lncRNA sequence branch (1D CNN) - commented for future activation
        # self.embedding_xt_lnc = nn.Embedding(num_features_xt + 1, embed_dim)
        # self.conv_xt_11_lnc = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=4, padding=2)
        # self.conv_xt_12_lnc = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=4, padding=2)
        # 
        # self.conv_xt_21_lnc = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=3, padding=1)
        # self.conv_xt_22_lnc = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=3, padding=1)
        # 
        # self.conv_xt_31_lnc = nn.Conv1d(embed_dim, out_channels=n_filters, kernel_size=2, padding=1)
        # self.conv_xt_32_lnc = nn.Conv1d(n_filters, out_channels=n_filters*2, kernel_size=2, padding=1)
        # 
        # self.fc1_xt_lnc = nn.Linear(n_filters * 2, output_dim)

    def forward(self, data):
        # miRNA version forward pass logic (main code)
        # SMILES character processing
        smile_embed = self.smile_embed(data.smile)
        smile_embed = smile_embed.permute(0, 2, 1)
        
        # Multi-scale CNN for SMILES
        conv_xd_1 = self.relu(self.conv_xd_11(smile_embed))
        conv_xd_1 = self.relu(self.conv_xd_12(conv_xd_1))
        conv_xd_1 = F.max_pool1d(conv_xd_1, kernel_size=2)
        
        conv_xd_2 = self.relu(self.conv_xd_21(smile_embed))
        conv_xd_2 = self.relu(self.conv_xd_22(conv_xd_2))
        conv_xd_2 = F.max_pool1d(conv_xd_2, kernel_size=2)
        
        conv_xd_3 = self.relu(self.conv_xd_31(smile_embed))
        conv_xd_3 = self.relu(self.conv_xd_32(conv_xd_3))
        conv_xd_3 = F.max_pool1d(conv_xd_3, kernel_size=2)
        
        # Concatenate and reduce dimensions
        conv_xd = torch.cat([conv_xd_1, conv_xd_2, conv_xd_3], dim=1)
        conv_xd = self.conv_reduce_smiles(conv_xd)
        conv_xd = F.max_pool1d(conv_xd, kernel_size=conv_xd.size(2))
        conv_xd = conv_xd.squeeze(2)
        conv_xd = self.fc_smiles(conv_xd)

        # SMILES graph processing
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gcnv1(x, edge_index)
        x = self.relu(x)
        x = self.gcnv2(x, edge_index)
        x = self.relu(x)
        x = gmp(x, batch)
        x = self.fc_g1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_g2(x)
        
        # miRNA sequence processing
        xt_embed = self.embedding_xt(data.target)
        xt_embed = xt_embed.permute(0, 2, 1)
        
        # Multi-scale CNN for miRNA
        conv_xt_1 = self.relu(self.conv_xt_11(xt_embed))
        conv_xt_1 = self.relu(self.conv_xt_12(conv_xt_1))
        conv_xt_1 = F.max_pool1d(conv_xt_1, kernel_size=2)
        
        conv_xt_2 = self.relu(self.conv_xt_21(xt_embed))
        conv_xt_2 = self.relu(self.conv_xt_22(conv_xt_2))
        conv_xt_2 = F.max_pool1d(conv_xt_2, kernel_size=2)
        
        conv_xt_3 = self.relu(self.conv_xt_31(xt_embed))
        conv_xt_3 = self.relu(self.conv_xt_32(conv_xt_3))
        conv_xt_3 = F.max_pool1d(conv_xt_3, kernel_size=2)
        
        # Concatenate and reduce dimensions
        conv_xt = torch.cat([conv_xt_1, conv_xt_2, conv_xt_3], dim=1)
        conv_xt = self.conv_reduce_xt(conv_xt)
        conv_xt = F.max_pool1d(conv_xt, kernel_size=conv_xt.size(2))
        conv_xt = conv_xt.squeeze(2)
        conv_xt = self.fc1_xt(conv_xt)
        
        # Combine features
        xc = torch.cat([x, conv_xt], dim=1)
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        
        return out

        # lncRNA version forward pass logic (commented for future activation)
        # def forward_lnc(self, data):
        #     """lncRNA version forward pass"""
        #     # SMILES character processing
        #     smile_embed = self.smile_embed(data.smile)
        #     smile_embed = smile_embed.permute(0, 2, 1)
        #     
        #     # Multi-scale CNN for SMILES
        #     conv_xd_1 = self.relu(self.conv_xd_11(smile_embed))
        #     conv_xd_1 = self.relu(self.conv_xd_12(conv_xd_1))
        #     conv_xd_1 = F.max_pool1d(conv_xd_1, kernel_size=2)
        #     
        #     conv_xd_2 = self.relu(self.conv_xd_21(smile_embed))
        #     conv_xd_2 = self.relu(self.conv_xd_22(conv_xd_2))
        #     conv_xd_2 = F.max_pool1d(conv_xd_2, kernel_size=2)
        #     
        #     conv_xd_3 = self.relu(self.conv_xd_31(smile_embed))
        #     conv_xd_3 = self.relu(self.conv_xd_32(conv_xd_3))
        #     conv_xd_3 = F.max_pool1d(conv_xd_3, kernel_size=2)
        #     
        #     # Concatenate and reduce dimensions
        #     conv_xd = torch.cat([conv_xd_1, conv_xd_2, conv_xd_3], dim=1)
        #     conv_xd = self.conv_reduce_smiles(conv_xd)
        #     conv_xd = F.max_pool1d(conv_xd, kernel_size=conv_xd.size(2))
        #     conv_xd = conv_xd.squeeze(2)
        #     conv_xd = self.fc_smiles(conv_xd)
        #     
        #     # SMILES graph processing
        #     x, edge_index, batch = data.x, data.edge_index, data.batch
        #     x = self.gcnv1(x, edge_index)
        #     x = self.relu(x)
        #     x = self.gcnv2(x, edge_index)
        #     x = self.relu(x)
        #     x = gmp(x, batch)
        #     x = self.fc_g1(x)
        #     x = self.relu(x)
        #     x = self.dropout(x)
        #     x = self.fc_g2(x)
        #     
        #     # lncRNA sequence processing
        #     xt_embed = self.embedding_xt_lnc(data.target)
        #     xt_embed = xt_embed.permute(0, 2, 1)
        #     
        #     # Multi-scale CNN for lncRNA
        #     conv_xt_1 = self.relu(self.conv_xt_11_lnc(xt_embed))
        #     conv_xt_1 = self.relu(self.conv_xt_12_lnc(conv_xt_1))
        #     conv_xt_1 = F.max_pool1d(conv_xt_1, kernel_size=2)
        #     
        #     conv_xt_2 = self.relu(self.conv_xt_21_lnc(xt_embed))
        #     conv_xt_2 = self.relu(self.conv_xt_22_lnc(conv_xt_2))
        #     conv_xt_2 = F.max_pool1d(conv_xt_2, kernel_size=2)
        #     
        #     conv_xt_3 = self.relu(self.conv_xt_31_lnc(xt_embed))
        #     conv_xt_3 = self.relu(self.conv_xt_32_lnc(conv_xt_3))
        #     conv_xt_3 = F.max_pool1d(conv_xt_3, kernel_size=2)
        #     
        #     # Concatenate and reduce dimensions
        #     conv_xt = torch.cat([conv_xt_1, conv_xt_2, conv_xt_3], dim=1)
        #     conv_xt = self.conv_reduce_xt(conv_xt)
        #     conv_xt = F.max_pool1d(conv_xt, kernel_size=conv_xt.size(2))
        #     conv_xt = conv_xt.squeeze(2)
        #     conv_xt = self.fc1_xt_lnc(conv_xt)
        #     
        #     # Combine features
        #     xc = torch.cat([x, conv_xt], dim=1)
        #     xc = self.fc1(xc)
        #     xc = self.relu(xc)
        #     xc = self.dropout(xc)
        #     xc = self.fc2(xc)
        #     xc = self.relu(xc)
        #     xc = self.dropout(xc)
        #     out = self.out(xc)
        #     
        #     return out