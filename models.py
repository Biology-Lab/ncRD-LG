import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Conv1d
from torch_geometric.nn import GCNConv, RGCNConv, global_sort_pool, global_add_pool
from torch_geometric.utils import dropout_edge
from torch_geometric.nn import dense, norm
from util_functions import *
import pdb
import time


class LocalGL(torch.nn.Module):
    # The GNN model of local context graph learning. 
    # Use GCN convolution + center-nodes readout.
    def __init__(self, dataset, gconv=GCNConv, latent_dim=[128,64,32,1],
                 num_relations=2, num_bases=2, regression=True, adj_dropout=0, 
                 force_undirected=False, side_features=False, n_side_features=0, 
                 multiply_by=1):

        super(LocalGL, self).__init__()

        self.regression = regression
        self.adj_dropout = adj_dropout
        self.force_undirected = force_undirected
        self.multiply_by = multiply_by
        self.convs = torch.nn.ModuleList()
        self.convs.append(gconv(-1, latent_dim[0], num_relations, num_bases))
        self.BN = norm.BatchNorm(latent_dim[0])

        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, num_bases))

        self.side_features = side_features
        if side_features:
            self.lin1 = Linear(2 * sum(latent_dim) + n_side_features, 128)
        else:
            self.lin1 = Linear(2 * sum(latent_dim), 128)

        if self.regression:
            self.lin2 = Linear(128, 1)
        else:
            self.lin2 = Linear(128, 1)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, data):
        start = time.time()
        x, edge_index, edge_type, batch = data.x, data.edge_index, data.edge_type, data.batch

        if self.adj_dropout > 0:

            edge_index,edge_type = dropout_edge(
                    edge_index,p=self.adj_dropout,
                    force_undirected=self.force_undirected,training=self.training)

        concat_states = []
        for conv in self.convs:
            x = torch.tanh(conv(x, edge_index))
            concat_states.append(x)
        concat_states = torch.cat(concat_states, 1)

        users = data.x[:, 0] == 1
        items = data.x[:, 1] == 1

        
        x = torch.cat([concat_states[users], concat_states[items]], 1)
        if self.side_features:
            x = torch.cat([x, data.u_feature, data.v_feature], 1)

        x_lin1=self.lin1(x)
        x = F.relu(x_lin1)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0] * self.multiply_by,x_lin1
        else:
            print('output:',F.log_softmax(x, dim=-1))
            return F.log_softmax(x, dim=-1)

