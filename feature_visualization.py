import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
import torch
import seaborn as sns


file_path = os.path.dirname(__file__)
dataset_path = os.path.abspath(os.path.join(file_path, 'Dataset'))
LncRNA = torch.load(os.path.join(dataset_path, 'LncRNA.pt'))
LncDrug = torch.load(os.path.join(dataset_path, 'LncDrug.pt'))
latent_features = np.load(os.path.join(dataset_path, 'Lnc_latent_feature.npy'))
LncPair_new = np.load(os.path.join(dataset_path, 'LncPair_new.npy'))
# MiRNA = torch.load(os.path.join(dataset_path, 'MiRNA.pt'))
# MiDrug = torch.load(os.path.join(dataset_path, 'MiDrug.pt'))
# latent_features = np.load(os.path.join(dataset_path, 'Mi_latent_feature.npy'))
# MiPair_new = np.load(os.path.join(dataset_path, 'MiPair_new.npy'))

test_samples = []

for i in range(LncPair_new.shape[0]):
    for j in range(LncPair_new.shape[1]):
        value = LncPair_new[i, j]
        if value == 2:
            test_samples.append((i, j, 1))
        elif value == -30:
            test_samples.append((i, j, 0))

# for i in range(MiPair_new.shape[0]):
#     for j in range(MiPair_new.shape[1]):
#         value = MiPair_new[i, j]
#         if value == 2:
#             test_samples.append((i, j, 1))
#         elif value == -30:
#             test_samples.append((i, j, 0))


test_samples = np.array(test_samples)
labels_list = []

features_list = []


for i, j, label in test_samples:
    LncRNA_feat = LncRNA[i].numpy()
    LncDrug_feat = LncDrug[j].numpy()
    combined_feat = np.hstack([LncRNA_feat, LncDrug_feat])
    # MiRNA_feat = MiRNA[i].numpy()
    # MiDrug_feat = MiDrug[j].numpy()
    # combined_feat = np.hstack([MiRNA_feat, MiDrug_feat])
    features_list.append(combined_feat)
    labels_list.append(label)


features = np.array(features_list)
labels = np.array(labels_list)
pos_indices = np.where(labels == 1)[0]
neg_indices = np.where(labels == 0)[0]

tsne_2d = TSNE(n_components=2, random_state=42, init='pca')


# raw node features
features_2d = tsne_2d.fit_transform(features)
pos_2d = features_2d[pos_indices]
neg_2d = features_2d[neg_indices]

plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")
pos = plt.scatter(pos_2d[:, 0], pos_2d[:, 1], c='darkorange', label='Positive', alpha=0.6, edgecolor='k', s=50)
neg = plt.scatter(neg_2d[:, 0], neg_2d[:, 1], c='blue', label='Negative', alpha=0.6, edgecolor='k', s=50)
plt.title('t-SNE Visualization of drug-lncRNA\n(raw node features)', fontsize=28,fontweight='bold')
# plt.title('t-SNE Visualization of drug-miRNA\n(raw node features)', fontsize=28,fontweight='bold')
plt.legend(fontsize=20, loc='lower left',markerscale=2)
plt.tight_layout()
plt.show()


# latent features
latent_features_2d = tsne_2d.fit_transform(latent_features)
latent_pos_2d = latent_features_2d[pos_indices]
latent_neg_2d = latent_features_2d[neg_indices]

plt.figure(figsize=(10, 10))
sns.set(style="whitegrid")
pos = plt.scatter(latent_pos_2d[:, 0], latent_pos_2d[:, 1], c='darkorange', label='Positive', alpha=0.6, edgecolor='k', s=50)
neg = plt.scatter(latent_neg_2d[:, 0], latent_neg_2d[:, 1], c='blue', label='Negative', alpha=0.6, edgecolor='k', s=50)
plt.title('t-SNE Visualization of drug-lncRNA\n(latent features)', fontsize=28,fontweight='bold')
# plt.title('t-SNE Visualization of drug-miRNA\n(latent features)', fontsize=28,fontweight='bold')
plt.legend(fontsize=20, loc='lower left',markerscale=2)
plt.tight_layout()
plt.show()

