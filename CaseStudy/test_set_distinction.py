import torch
import pandas as pd

# 1. Load necessary data
edge_labels = torch.load("../Dataset/MiTest_edge_label.pt")  # Edge labels
edge_indices = torch.load("../Dataset/MiTest_edge_index.pt")  # Edge indices

# 2. Extract RNA and Drug indices
rna_indices = edge_indices[0]  # RNA indices
drug_indices = edge_indices[1]  # Drug indices

# 3. Get positive and negative sample indices
positive_indices = (edge_labels == 1).nonzero(as_tuple=True)[0]  # Positive sample indices
negative_indices = (edge_labels == 0).nonzero(as_tuple=True)[0]  # Negative sample indices

# 4. Extract RNA and Drug indices based on indices
positive_rna = rna_indices[positive_indices]
positive_drug = drug_indices[positive_indices]

negative_rna = rna_indices[negative_indices]
negative_drug = drug_indices[negative_indices]

# 5. Convert to DataFrame format for subsequent operations
positive_pairs = pd.DataFrame({"RNA": positive_rna.tolist(), "Drug": positive_drug.tolist()})
negative_pairs = pd.DataFrame({"RNA": negative_rna.tolist(), "Drug": negative_drug.tolist()})

# Save as .csv files
positive_pairs.to_csv("./positive_pairs.csv", index=False)
negative_pairs.to_csv("./negative_pairs.csv", index=False)