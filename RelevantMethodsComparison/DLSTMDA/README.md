# DLSTMDA: Deep Learning for ncRNA-Drug Association Prediction

## Overview

DLSTMDA is a deep learning model that combines Graph Convolutional Networks (GCN) with Convolutional Neural Networks (CNN) for predicting ncRNA-drug associations. The model processes molecular graphs and sequence data to learn effective representations.

## Features

- **Multi-modal Learning**: Combines graph and sequence information
- **Graph Convolutional Networks**: Processes molecular graphs from SMILES
- **1D CNN**: Handles ncRNA sequence data
- **Multi-scale Feature Extraction**: Uses different kernel sizes for comprehensive feature learning
- **Dual Dataset Support**: Supports both miRNA and lncRNA datasets

## Model Architecture

```
Input: ncRNA Sequence + Drug SMILES
    ↓
Sequence Embedding + Graph Construction
    ↓
Multi-scale CNN + GCN Processing
    ↓
Feature Fusion
    ↓
Final Prediction
```

## Key Components

### 1. Graph Processing (`GCNNetmuti`)

- **SMILES Graph Branch**: Converts SMILES to molecular graphs
- **GCN Layers**: Graph convolutional operations
- **Multi-scale CNN**: Different kernel sizes (1, 2, 3) for SMILES

### 2. Sequence Processing

- **ncRNA Sequence Branch**: 1D CNN for sequence processing
- **Multi-scale CNN**: Different kernel sizes (2, 3, 4) for sequences
- **Embedding Layer**: Converts sequences to embeddings

### 3. Feature Fusion

- **Concatenation**: Combines graph and sequence features
- **Fully Connected Layers**: Final prediction layers

## File Structure

```
DLSTMDA/
├── training.py          # Main training script
├── cnn_gcnmulti.py     # Model architecture
├── utils.py            # Utility functions
├── process_data.py     # Data preprocessing
└── README.md          # This file
```

## Usage

### Prerequisites

```bash
pip install torch torch-geometric
pip install rdkit-pypi
pip install scikit-learn pandas numpy
pip install torch-geometric
```

### Data Preparation

1. **Preprocess Data**:

```bash
python process_data.py
```

2. **Training**:

```bash
python training.py
```

### Data Format

The model expects the following data structure:

#### miRNA Dataset

- `mi_drug_with_SMILES.csv` - Drug SMILES data
- `miRNA_with_sequence.xlsx` - miRNA sequences
- `MiPair.csv` - miRNA-drug interaction pairs

#### lncRNA Dataset (Commented for Future Activation)

- `lnc_drug_with_SMILES.xlsx` - Drug SMILES data
- `lncRNA_with_sequence.xlsx` - lncRNA sequences
- `LncPair.csv` - lncRNA-drug interaction pairs

## Model Parameters

| Parameter            | Description            | Default Value |
| -------------------- | ---------------------- | ------------- |
| `n_filters`          | Number of CNN filters  | 32            |
| `embed_dim`          | Embedding dimension    | 64            |
| `num_features_xd`    | Drug feature dimension | 78            |
| `num_features_smile` | SMILES vocabulary size | 66            |
| `num_features_xt`    | ncRNA vocabulary size  | 25            |
| `output_dim`         | Output dimension       | 128           |
| `dropout`            | Dropout rate           | 0.2           |

## Training Configuration

```python
# Training parameters
epochs = 100
batch_size = 32
learning_rate = 0.001
weight_decay = 1e-5

# Model parameters
n_filters = 32
embed_dim = 64
output_dim = 128
```

## Data Processing Pipeline

### 1. SMILES Processing

- Convert SMILES to molecular graphs using RDKit
- Extract atom features and bond information
- Create adjacency matrices

### 2. Sequence Processing

- Convert ncRNA sequences to one-hot encoding
- Apply multi-scale CNN with different kernel sizes
- Extract sequence features

### 3. Graph Construction

- Build molecular graphs from SMILES
- Create edge indices and node features
- Prepare for GCN processing

## Model Training

The training process includes:

1. **Data Loading**: Load preprocessed data
2. **Model Initialization**: Create GCNNetmuti model
3. **Training Loop**: 
   - Forward pass
   - Loss computation
   - Backward pass
   - Parameter updates
4. **Evaluation**: Regular performance evaluation
5. **Model Saving**: Save best model checkpoints

## Performance Metrics

The model evaluates performance using:

- **Accuracy**: Classification accuracy
- **Precision**: Precision score
- **Recall**: Recall score
- **F1-Score**: F1 score
- **AUC**: Area Under the ROC Curve

## Output Files

- `best_model.pth` - Best model checkpoint
- `processed/` - Preprocessed data directory
- Training logs and metrics

## Dataset Support

### miRNA Dataset (Main)

- Processes miRNA sequences
- Uses miRNA-drug interaction data
- Optimized for shorter sequences

### lncRNA Dataset (Commented)

- Designed for longer lncRNA sequences
- Handles variable sequence lengths
- Can be activated by uncommenting relevant code

## Usage Example

```python
from cnn_gcnmulti import GCNNetmuti
from utils import load_data

# Load data
train_loader, test_loader = load_data()

# Create model
model = GCNNetmuti()

# Training
for epoch in range(epochs):
    train_loss, train_acc = train(model, device, train_loader, optimizer, epoch, loss_fn, LOG_INTERVAL)
    
    if epoch % 10 == 0:
        y_true, y_pred, y_probs = predicting(model, device, test_loader)
        # Evaluate performance
```

## Citation

If you use this model in your research, please cite:

```bibtex
@article{dlstmda2024,
  title={DLSTMDA: Deep Learning for ncRNA-Drug Association Prediction},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions and support, please contact [your-email@domain.com].
