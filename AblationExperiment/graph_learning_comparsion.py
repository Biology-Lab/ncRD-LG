import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch_geometric.nn as pyg_nn
import random
import pandas as pd

# %%
# Set the random seed to ensure the repeatability of the results
SEED = 666

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Check if GPU is available, if so use GPU for calculation, otherwise use CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Define the GNN decoder class
class GNNDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):

        super(GNNDecoder, self).__init__()
        self.gcn1 = pyg_nn.GCNConv(in_channels, hidden_channels)
        self.gcn2 = pyg_nn.GCNConv(hidden_channels, out_channels)
        # The following are other optional convolutional layers that can be replaced as needed
        # self.gcn1 = pyg_nn.GATConv(in_channels, hidden_channels,heads=2)
        # self.gcn2 = pyg_nn.GATConv(2*hidden_channels, out_channels,heads=2)
        # self.gcn1 = pyg_nn.SAGEConv(in_channels, hidden_channels)
        # self.gcn2 = pyg_nn.SAGEConv(hidden_channels, out_channels)
        # self.gcn1 = pyg_nn.LEConv(in_channels, hidden_channels)
        # self.gcn2 = pyg_nn.LEConv(hidden_channels, out_channels)
        # self.gcn1 = pyg_nn.GeneralConv(in_channels, hidden_channels)
        # self.gcn2 = pyg_nn.GeneralConv(hidden_channels, out_channels)
        # self.gcn1 = pyg_nn.GraphConv(in_channels, hidden_channels)
        # self.gcn2 = pyg_nn.GraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):

        x = self.gcn1(x, edge_index)
        x = torch.relu(x)
        x = self.gcn2(x, edge_index)

        # num_RNA = 1167   ##lncRNA
        num_RNA = 1805
        RNA_emb = x[:num_RNA]
        drug_emb = x[num_RNA:]

        # Compute the matrix product of RNA embedding and drug embedding
        out = torch.matmul(RNA_emb, drug_emb.T)
        return out


# Define training function
def train(model, data, optimizer, train_edge_label_index, train_edge_label):

    model.train()

    optimizer.zero_grad()
    # Forward propagation
    out = model(data.x, data.edge_index)

    # Extract the prediction results of the training edge
    out = out[train_edge_label_index[0], train_edge_label_index[1]]

    # True labels for training edges
    labels = train_edge_label

    # Calculate loss, binary cross entropy loss function
    loss = nn.BCEWithLogitsLoss()(out, labels)
    # Backpropagation
    loss.backward()
    # Update model parameters
    optimizer.step()
    return loss.item()


# Define the evaluation function
def evaluate(model, data, test_edge_label_index, test_edge_label, method_name):

    model.eval()
    with torch.no_grad():
        # Forward propagation
        out = model(data.x, data.edge_index)

        out = out[test_edge_label_index[0], test_edge_label_index[1]]

        labels = test_edge_label
        y_true = labels.cpu().numpy()
        y_scores = out.cpu().numpy()

        df = pd.DataFrame({
            'y_scores': y_scores,
            'y_true': y_true
        })
        # Save the results to a CSV file
        res_name = f'{method_name}_MiDrug_score.csv'
        # res_name = f'{method_name}_LncDrug_score.csv'
        df.to_csv(res_name, index=False)

        auc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
    return auc, aupr


# Convert edge indices to adjacency matrix
def edge_index_to_matrix(edge_index, num_code_1, num_code_2):
    # Initialize the adjacency matrix
    matrix = torch.zeros((num_code_1, num_code_2), dtype=torch.float32).to(device)
    row, col = edge_index
    # Set the corresponding position value of the adjacency matrix to 1
    matrix[row, col] = 1
    matrix[col, row] = 1
    return matrix



learn_rate = 0.001

edge_type = 'MiDrug'
node_type = 'miRNA'
sim_type = 'MiMi'
# edge_type = 'LncDrug'
# node_type = 'lncRNA'
# sim_type = 'LncLnc'
method_name = 'GCNConv'

# Read the dataset
x_mi_RNA = torch.load('../Dataset/MiRNA.pt').to(device)
x_mi_drug = torch.load('../Dataset/MiDrug.pt').to(device)
# x_lnc_RNA = torch.load('../Dataset/LncRNA.pt').to(device)
# x_lnc_drug = torch.load('../Dataset/LncDrug.pt').to(device)
train_edge_index = torch.load('../Dataset/MiTrain_edge_index.pt').to(device)
test_edge_index = torch.load('../Dataset/MiTest_edge_index.pt').to(device)
train_edge_label = torch.load('../Dataset/MiTrain_edge_label.pt').to(device)
test_edge_label = torch.load('../Dataset/MiTest_edge_label.pt').to(device)
# train_edge_index = torch.load('../Dataset/LncTrain_edge_index.pt').to(device)
# test_edge_index = torch.load('../Dataset/LncTest_edge_index.pt').to(device)
# train_edge_label = torch.load('../Dataset/LncTrain_edge_label.pt').to(device)
# test_edge_label = torch.load('../Dataset/LncTest_edge_label.pt').to(device)

# RNA, the number of drug nodes
num_RNA = x_mi_RNA.shape[0]
num_drug = x_mi_drug.shape[0]
# num_RNA = x_lnc_RNA.shape[0]
# num_drug = x_lnc_drug.shape[0]
num_nodes = num_RNA + num_drug

# Build a node feature matrix and combine the features of RNA and drugs
node_features = torch.cat([x_mi_RNA, x_mi_drug], dim=0).to(device)
print('node_features:', node_features.device)

# Generate RNA similarity edges (edge indices corresponding to the diagonal matrix)
rna_sim_edge_index = torch.stack([torch.arange(num_RNA), torch.arange(num_RNA)], dim=0).to(device)
# Generate drug similarity edges (edge indices corresponding to the diagonal matrix), note that the drug index should be offset
drug_sim_edge_index = torch.stack(
    [torch.arange(num_RNA, num_RNA + num_drug), torch.arange(num_RNA, num_RNA + num_drug)], dim=0).to(device)

# Undirected graph, need to add reverse edge
reverse_rna_sim_edge_index = rna_sim_edge_index.flip(0)
reverse_drug_sim_edge_index = drug_sim_edge_index.flip(0)

# Merge forward and reverse edges
rna_sim_edge_index_combined = torch.cat([rna_sim_edge_index, reverse_rna_sim_edge_index], dim=1)
drug_sim_edge_index_combined = torch.cat([drug_sim_edge_index, reverse_drug_sim_edge_index], dim=1)

# Build all edges in the graph
# The original indexes of RNA and drugs both start at 0. Consider them as the same type and adjust the index of drugs.
edge_index_train = train_edge_index.clone()
edge_index_train[1] += num_RNA
edge_index_test = test_edge_index.clone()
edge_index_test[1] += num_RNA

# Add reverse edges to undirected graph
reverse_edge_index_train = edge_index_train.flip(0)
reverse_edge_index_test = edge_index_test.flip(0)

# Merge all edges
edge_index_combined_train = torch.cat(
    [edge_index_train, reverse_edge_index_train, rna_sim_edge_index_combined, drug_sim_edge_index_combined], dim=1)
edge_index_combined_test = torch.cat(
    [edge_index_test, reverse_edge_index_test, rna_sim_edge_index_combined, drug_sim_edge_index_combined], dim=1)

# Construct positive and negative sample index and label in training set and test set
train_edge_label_index = train_edge_index
train_edge_label = train_edge_label

test_edge_label_index = test_edge_index
test_edge_label = test_edge_label

# Build graph data object
data_train = Data(x=node_features, edge_index=edge_index_combined_train).to(device)
data_test = Data(x=node_features, edge_index=edge_index_combined_test).to(device)

# Define model and optimizer
model = GNNDecoder(in_channels=node_features.shape[1], hidden_channels=128, out_channels=128).to(device)
optimizer = optim.RMSprop(model.parameters(), lr=learn_rate, alpha=0.99)

#Train model
num_epochs = 200
for epoch in range(num_epochs):
    loss = train(model, data_train, optimizer, train_edge_label_index, train_edge_label)
    if epoch % 20 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Test model performance
auc, aupr = evaluate(model, data_test, test_edge_label_index, test_edge_label, method_name)
print(f'Test AUC: {auc}, AUPR: {aupr}')

# # Use random walk model to find the optimal parameters
# # Define hyperparameter search space
# hidden_channels_list = [64, 128, 256]
# learn_rate_list = [0.0001, 0.001, 0.01]
# num_epochs_list = [100, 200, 300]
#
# # Number of random search trials
# num_trials = 20
# # initialization
# best_auc = 0
# best_aupr = 0
# best_hidden_channels = None
# best_learn_rate = None
# best_num_epochs = None
#
# # Perform a specified number of random search trials
# for _ in range(num_trials):
#
#     hidden_channels = random.choice(hidden_channels_list)
#
#     learn_rate = random.choice(learn_rate_list)
#
#     num_epochs = random.choice(num_epochs_list)
#
#     # Initialize the model based on the randomly selected number of hidden layer channels
#     model = GNNDecoder(in_channels=node_features.shape[1], hidden_channels=hidden_channels, out_channels=128).to(device)
#     # Initialize the optimizer based on a randomly selected learning rate and use the RMSprop optimization algorithm
#     optimizer = optim.RMSprop(model.parameters(), lr=learn_rate, alpha=0.99)
#
#     # Start training the model with a randomly selected number of num_epochs
#     for epoch in range(num_epochs):
#
#         loss = train(model, data_train, optimizer, train_edge_label_index, train_edge_label)
#
#         if epoch % 20 == 0:
#             print(f'Epoch {epoch}, Loss: {loss}, hidden_channels: {hidden_channels}, learn_rate: {learn_rate}, num_epochs: {num_epochs}')
#
#     # After training is completed, call the evaluate function to evaluate the model performance on the test set and get the AUC and AUPR values
#     auc, aupr = evaluate(model, data_test, test_edge_label_index, test_edge_label, method_name='GeneralConv')
#     print(f'hidden_channels: {hidden_channels}, learn_rate: {learn_rate}, num_epochs: {num_epochs}, AUC: {auc}, AUPR: {aupr}')
#
#     # If the AUC and AUPR values obtained in the current experiment are better than the best values recorded previously
#     if auc > best_auc and aupr > best_aupr:
#
#         best_auc = auc
#         best_aupr = aupr
#         best_hidden_channels = hidden_channels
#         best_learn_rate = learn_rate
#         best_num_epochs = num_epochs
#
#
# print(f'Best hidden_channels: {best_hidden_channels}, Best learn_rate: {best_learn_rate}, Best num_epochs: {best_num_epochs}')
#
# print(f'Best AUC: {best_auc}, Best AUPR: {best_aupr}')