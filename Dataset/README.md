# 数据集说明

----- lncRNA_with_sequence：List of lncRNA names and their molecular sequences obtained after data preprocessing

----- lnc_drug_with_SMILES： List of drugs corresponding to each lncRNA and their SMILES, obtained after data preprocessing

----- LncDrug_features：List of lncRNA-related drug names and their 768-dimensional features extracted by a pretrained large predictive model

----- LncDrug.pt：768-dimensional features of drugs in the lncRNA dataset extracted by a pretrained large language model

----- LncRNA.pt：768-dimensional features of RNAs in the lncRNA dataset extracted by a pretrained large language model

----- LncTrain_edge_index.pt：Edge index of the lncRNA training set

----- LncVal_edge_index.pt：Edge index of the lncRNA validation set

----- LncTest_edge_index.pt：Edge index of the lncRNA test set

----- LncTrain_edge_label.pt：Edge labels of the lncRNA training set

----- LncVal_edge_label.pt：Edge labels of the lncRNA validation set

----- LncTest_edge_label.pt：Edge labels of the lncRNA test set

----- LncPair.pt：lncRNA-drug association matrix, where a(i,j) denotes the association between the i-th lncRNA and the j-th drug

----- LncPair_new.pt：lncRNA-drug association matrix filled with dataset positive/negative labels; label 1 and -20 denote positive/negative samples in the training set, label -1 and -10 denote positive/negative samples in the validation set, and label 2 and -30 denote positive/negative samples in the test set

----- Lnc_latent_feature.npy：Features obtained after subgraph extraction for lncRNA

----- miRNA_with_sequence：List of miRNA names and their molecular sequences obtained after data preprocessing

----- mi_drug_with_SMILES：List of drugs corresponding to each miRNA and their SMILES, obtained after data preprocessing

----- MiDrug_features：List of miRNA-related drug names and their 768-dimensional features extracted by a pretrained large predictive model

----- MiDrug.pt：768-dimensional features of drugs in the miRNA dataset extracted by a pretrained large language model

----- MiRNA.pt：768-dimensional features of RNAs in the miRNA dataset extracted by a pretrained large language model

----- MiTrain_edge_index.pt：Edge index of the miRNA training set

----- MiVal_edge_index.pt：Edge index of the miRNA validation set

----- MiTest_edge_index.pt：Edge index of the miRNA test set

----- MiTrain_edge_label.pt：Edge labels of the miRNA training set

----- MiVal_edge_label.pt：Edge labels of the miRNA validation set

----- MiTest_edge_label.pt：Edge labels of the miRNA test set

----- MiPair.pt：miRNA-drug association matrix, where a(i,j) denotes the association between the i-th miRNA and the j-th drug

----- MiPair_new.pt：miRNA-drug association matrix filled with dataset positive/negative labels; label 1 and -20 denote positive/negative samples in the training set, label -1 and -10 denote positive/negative samples in the validation set, and label 2 and -30 denote positive/negative samples in the test set

----- MiEvaluation.npy：Test results predicted by the model for miRNA

----- Mi_latent_feature.npy：Features obtained after subgraph extraction for miRNA
