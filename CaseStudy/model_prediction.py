import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from models import LocalGL
from util_functions import MyDynamicDataset

# Set up device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Load data
def import_data():
    x_mi_RNA = torch.load(f'../Dataset/MiRNA.pt')
    x_mi_drug = torch.load(f'../Dataset/MiDrug.pt')
    pair = np.load(f'../Dataset/MiPair.npy')
    pair_t_v = np.load(f'../Dataset/MiPair_new.npy')
    return x_mi_RNA, x_mi_drug, pair, pair_t_v

# Load test data
miRNA_feature, drug_feature, pair, pair_t_v = import_data()
print(miRNA_feature.shape)
print(drug_feature.shape)

# Extract test set index
test_index = np.where((pair_t_v == 2) | (pair_t_v == -30))
adj_test = np.zeros(pair.shape)
adj_test[test_index] = pair[test_index]
test_label = np.array(adj_test[test_index])

# Build test set graph data
adj_test = csr_matrix(adj_test)
test_graphs = MyDynamicDataset(
    root='data/test',
    A=adj_test,
    links=test_index,
    labels=test_label,
    h=2,
    sample_ratio=1.0,
    max_nodes_per_hop=100,
    u_features=miRNA_feature,
    v_features=drug_feature,
    class_values=[0, 1],
    max_num=None
)
data = test_graphs.get(0)
print("Subgraph node feature dimension:", data.x.shape)

# Load model
model = LocalGL(
    dataset=test_graphs,
    latent_dim=[128, 64, 32, 1],
    num_relations=2,
    num_bases=0,
    regression=True,
    adj_dropout=0,
    force_undirected=False,
    side_features=True,
    n_side_features=miRNA_feature.shape[1] + drug_feature.shape[1],
    multiply_by=1
)
model.to(device)

# Load model weights
checkpoint_path = './mi_model_checkpoint20.pth'
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# Test set prediction
predictions = []
true_labels = []

with torch.no_grad():
    # Iterate through test set
    for i in range(len(test_graphs)):
        data = test_graphs.get(i).to(device)
        pred, _ = model(data)
        predictions.append(pred.cpu().numpy())
        true_labels.append(data.y.cpu().numpy())

# Process results
predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)

# Get RNA and Drug indices for test set
rna_indices = test_index[0]
drug_indices = test_index[1]

# Confirm data length consistency
assert len(predictions) == len(rna_indices) == len(drug_indices), "Data length inconsistent, please check input data"

# Set classification threshold
threshold = 0.9
positive_mask = predictions >= threshold
negative_mask = predictions < threshold

# Positive and negative sample dataframes
positive_samples = pd.DataFrame({
    "RNA": rna_indices[positive_mask],
    "Drug": drug_indices[positive_mask]
})

negative_samples = pd.DataFrame({
    "RNA": rna_indices[negative_mask],
    "Drug": drug_indices[negative_mask]
})

# Save as CSV files
positive_samples.to_csv(f"./positive_samples_0.9.csv", index=False)
negative_samples.to_csv(f"./negative_samples_0.9.csv", index=False)

print("Positive samples saved to positive_samples_0.9.csv")
print("Negative samples saved to negative_samples_0.9.csv")