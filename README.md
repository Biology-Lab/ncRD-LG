# ncRD-LG

This is a Pytorch implementation of our paper: "A Unified Framework Integrating Molecular Language Models and Subgraph Learning for Drug-ncRNA Target Prediction". It includes the implementation of the proposed predictor, along with datasets and tutorials.

# Environment Setup

----- OS: Ubuntu 20.04.5
----- Python 3.8
----- CUDA 11.3
----- PyTorch 1.12.1

# Introduction

Non-coding RNAs (ncRNAs), important molecules involved in the regulation of gene expression and protein function, are transcribed from genome regions that do not encode proteins. we propose a novel prediction framework named ncRD-LG, which relies solely on 1D sequences and subgraph-based learning for drug–ncRNA target prediction. The core idea of our approach is as follows: we first utilize pretrained language models to enrich the semantic features of drug molecules and ncRNAs. Then, based on the global drug–ncRNA interaction graph, we extract h-hop enclosing subgraphs centered on the target drug–ncRNA pair. A graph convolutional network (GCN) is applied to the local subgraph to learn context-specific representations for the central pair. Finally, the embeddings of the drug and ncRNA are concatenated and input into a multilayer perceptron (MLP) to obtain the association probability.

# Main Scripts

----- drug_language_feature.py + ChemBERTa-zinc-base-v1: Extraction of drug features using ChemBERT.

----- ncRNA_language_feature.py + BiRNA-BERT + BiRNA-Tokenizer: Extraction of ncRNA features using BiRNA-BERT.

----- main.py + train_eval.py : Training code for local subgraph learning

----- models.py: Records the graph neural network architecture in ncRD-LG used for local context subgraph learning

----- util_function.py: Contains basic utility functions for constructing local context subgraphs for target drug–ncRNA pairs

----- evaluation_fun.py: Contains basic evaluation metric functions

----- find_best_params.py: Used to analyze parameter sensitivity and find optimal hyperparameters on the training set

----- feature_comparison.py: Ablation experiment to validate the role of features extracted by large language models

----- machine_learning_comparison.py: Ablation experiment to validate the role of graph structure

----- graph_learning_comparison.py: Ablation experiment to validate the role of local subgraph learning

----- feature_visualization.py: Feature visualization using t-Distributed Stochastic Neighbor Embedding (t-SNE)

----- model_prediction.py + test_set_distinction.py: Extract the positive samples from the model’s predictions on the test set, as well as the positive and negative samples from the original test set.