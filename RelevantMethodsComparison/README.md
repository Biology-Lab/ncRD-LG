# Comparison of Relevant Methods for ncRNA-Drug Association Prediction

## Overview

This repository contains implementations of four state-of-the-art deep learning models for predicting ncRNA-drug associations. All models have been unified to support both miRNA and lncRNA datasets with consistent naming conventions and English documentation.

## Models Included

### 1. [AGCLNDA](AGCLNDA/) - Adaptive Graph Contrastive Learning

- **Focus**: Adaptive graph contrastive learning with multi-generator architecture
- **Key Features**: VGAE generator, DenoisingNet, information bottleneck
- **Best For**: Complex molecular interactions with adaptive learning

### 2. [DLSTMDA](DLSTMDA/) - Deep Learning with GCN and CNN

- **Focus**: Multi-modal learning combining graph and sequence information
- **Key Features**: GCN for molecular graphs, 1D CNN for sequences
- **Best For**: Multi-scale feature extraction from different data types

### 3. [NDSGCL](NDSGCL/) - Neural Drug-ncRNA Subgraph Contrastive Learning

- **Focus**: Subgraph contrastive learning with staged training strategy
- **Key Features**: Hard negative mining, prototype learning, staged training
- **Best For**: Learning from difficult negative samples

### 4. [GraphDTA](GraphDTA/) - Graph-based Drug-Target Affinity

- **Focus**: Graph neural networks for drug-target affinity prediction
- **Key Features**: GCN/GAT architectures, ensemble models, attention mechanisms
- **Best For**: Molecular graph representation and affinity prediction

## Unified Features

All models have been standardized with:

- **Unified Naming**: Removed numerical suffixes, consistent naming conventions
- **Dual Dataset Support**: Both miRNA and lncRNA support (lncRNA code commented for future activation)
- **English Documentation**: All comments and documentation in English
- **Consistent Structure**: Similar file organization and parameter naming
- **Unified Data Paths**: All models read from `../../Dataset/` directory

## Quick Start

### Prerequisites

```bash
pip install torch torch-geometric
pip install rdkit-pypi
pip install scikit-learn pandas numpy
pip install scipy faiss-cpu
```

### Data Preparation

Place your data files in the `../../Dataset/` directory:

#### miRNA Dataset

- `MiPair.csv` - miRNA-drug interaction pairs
- `MiRNA_features.csv` - miRNA features
- `MiDrug_features.csv` - Drug features
- `mi_drug_with_SMILES.csv` - Drug SMILES data
- `miRNA_with_sequence.xlsx` - miRNA sequences

#### lncRNA Dataset

- `LncPair.csv` - lncRNA-drug interaction pairs
- `lncRNA_with_sequence.xlsx` - lncRNA sequences
- `lnc_drug_with_SMILES.xlsx` - Drug SMILES data

### Running Models

Each model can be run independently:

```bash
# AGCLNDA
cd AGCLNDA
python Main.py

# DLSTMDA
cd DLSTMDA
python process_data.py  # Preprocess data first
python training.py

# NDSGCL
cd NDSGCL
python main.py

# GraphDTA
cd GraphDTA
python create_data.py  # Preprocess data first
python training.py
```

## Model Comparison

| Model        | Architecture               | Key Features                            | Best Use Case                           |
| ------------ | -------------------------- | --------------------------------------- | --------------------------------------- |
| **AGCLNDA**  | GCN + Contrastive Learning | Multi-generator, Information bottleneck | Complex interactions, Adaptive learning |
| **DLSTMDA**  | GCN + CNN                  | Multi-modal, Multi-scale                | Graph + sequence data                   |
| **NDSGCL**   | GCN + Contrastive Learning | Hard negative mining, Staged training   | Difficult negative samples              |
| **GraphDTA** | GCN/GAT + CNN              | Attention, Ensemble                     | Molecular affinity prediction           |

## Performance Metrics

All models evaluate performance using:

- **AUC**: Area Under the ROC Curve
- **AUPR**: Area Under the Precision-Recall Curve
- **Accuracy**: Classification accuracy
- **Precision/Recall/F1**: Additional classification metrics

## Dataset Support

### miRNA Dataset (Primary)

All models are optimized for miRNA datasets with:

- Shorter sequence lengths
- miRNA-specific feature extraction
- Optimized hyperparameters

### lncRNA Dataset (Future Activation)

lncRNA support is available but commented out:

- Longer sequence handling
- lncRNA-specific processing
- Can be activated by uncommenting relevant code sections

## File Structure

```
Comparison of Relevant Methods/
├── AGCLNDA/
│   ├── Main.py              # Main training script
│   ├── Model.py             # Model architecture
│   ├── DataHandler.py       # Data loading
│   ├── Params.py            # Configuration
│   └── README.md           # Model documentation
├── DLSTMDA/
│   ├── training.py          # Main training script
│   ├── cnn_gcnmulti.py     # Model architecture
│   ├── utils.py            # Utility functions
│   ├── process_data.py     # Data preprocessing
│   └── README.md           # Model documentation
├── NDSGCL/
│   ├── main.py             # Main training script
│   ├── NDSGCL.py           # Model architecture
│   ├── util.py             # Utility functions
│   ├── config.conf         # Configuration
│   └── README.md           # Model documentation
├── GraphDTA/
│   ├── training.py         # Main training script
│   ├── create_data.py      # Data preprocessing
│   ├── utils.py            # Utility functions
│   ├── models/             # Model architectures
│   └── README.md           # Model documentation
└── README.md              # This file
```

## Configuration

Each model has its own configuration system:

- **AGCLNDA**: `Params.py` - Command line arguments
- **DLSTMDA**: Hardcoded parameters in `training.py`
- **NDSGCL**: `config.conf` - Configuration file
- **GraphDTA**: Hardcoded parameters in `training.py`

## Common Parameters

| Parameter       | Description         | Typical Range |
| --------------- | ------------------- | ------------- |
| `learning_rate` | Learning rate       | 0.001 - 0.01  |
| `epochs`        | Number of epochs    | 50 - 200      |
| `batch_size`    | Batch size          | 32 - 256      |
| `embedding_dim` | Embedding dimension | 64 - 128      |
| `dropout`       | Dropout rate        | 0.1 - 0.3     |
