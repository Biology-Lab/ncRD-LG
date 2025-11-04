import torch
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

#Load data
x_lnc_RNA = torch.load(f'../Dataset/LncRNA.pt').numpy()
x_lnc_drug = torch.load(f'../Dataset/LncDrug.pt').numpy()
# x_mi_RNA = torch.load(f'../Dataset/MiRNA.pt').numpy()
# x_mi_drug = torch.load(f'../Dataset/MiDrug.pt').numpy()
train_edge_index = torch.load(f'../Dataset/LncTrain_edge_index.pt').numpy().T
test_edge_index = torch.load(f'../Dataset/LncTest_edge_index.pt').numpy().T
train_edge_label = torch.load(f'../Dataset/LncTrain_edge_label.pt').numpy()
test_edge_label = torch.load(f'../Dataset/LncTest_edge_label.pt').numpy()
# train_edge_index = torch.load(f'../Dataset/MiTrain_edge_index.pt').numpy().T
# test_edge_index = torch.load(f'../Dataset/MiTest_edge_index.pt').numpy().T
# train_edge_label = torch.load(f'../Dataset/MiTrain_edge_label.pt').numpy()
# test_edge_label = torch.load(f'../Dataset/MiTest_edge_label.pt').numpy()

x = np.vstack([x_lnc_RNA, x_lnc_drug])

# Extract features for training, validation, and test sets
def extract_features(edge_index, x):
    features = []
    for i, j in edge_index:
        feature = np.hstack([x[i], x[j]])
        features.append(feature)
    return np.array(features)

train_features = extract_features(train_edge_index, x)
test_features = extract_features(test_edge_index, x)

# Define model
models = {
    'MLP': MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500),
    'SVM': SVC(probability=True),
    'RF': RandomForestClassifier(n_estimators=100),
    'DT': DecisionTreeClassifier()
}

# Train the model and calculate AUC and AUPR
for name, model in models.items():
    model.fit(train_features, train_edge_label)

    # Predict probability
    y_pred_proba = model.predict_proba(test_features)[:, 1]

    # Calculate AUC and AUPR
    auc_score = roc_auc_score(test_edge_label, y_pred_proba)
    aupr_score = average_precision_score(test_edge_label, y_pred_proba)

    print(f'{name} - AUC: {auc_score:.4f}, AUPR: {aupr_score:.4f}')