# GraphDTA: Graph-based Drug-Target Affinity Prediction

## Overview

GraphDTA is a graph neural network model for predicting drug-target affinity. The model combines molecular graph representation with sequence-based features to predict binding affinity between drugs and targets (ncRNAs in this context).

## Features

- **Graph Neural Networks**: Uses GCN and GAT for molecular graph processing
- **Multi-architecture Support**: GCN, GAT, and ensemble models
- **Sequence Processing**: Handles ncRNA sequences with CNN
- **SMILES Processing**: Converts SMILES to molecular graphs
- **Dual Dataset Support**: Supports both miRNA and lncRNA datasets

## Model Architecture

```
Input: Drug SMILES + ncRNA Sequence
    ↓
Molecular Graph + Sequence Embedding
    ↓
GCN/GAT Processing + CNN Processing
    ↓
Feature Fusion
    ↓
Final Affinity Prediction
```

## Key Components

### 1. Graph Models

- **GCN**: Graph Convolutional Network
- **GAT**: Graph Attention Network
- **Ensemble**: Combined model architecture

### 2. Sequence Processing

- **CNN Layers**: 1D CNN for sequence processing
- **Multi-scale Processing**: Different kernel sizes
- **Embedding Layers**: Sequence to vector conversion

### 3. Feature Fusion

- **Attention Mechanism**: Weighted feature combination
- **Concatenation**: Feature merging
- **Fully Connected Layers**: Final prediction

## File Structure

```
GraphDTA/
├── training.py          # Main training script
├── create_data.py      # Data preprocessing
├── utils.py            # Utility functions
├── models/
│   ├── gcn.py         # GCN model
│   ├── gat.py         # GAT model
│   └── rna_net.py     # ncRNA network
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
python create_data.py
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

## Model Architectures

### 1. GCN Model (`GCNNet`)

- Graph Convolutional Network
- Multi-layer GCN processing
- Global pooling for graph-level features

### 2. GAT Model (`GATNet`)

- Graph Attention Network
- Attention mechanism for node importance
- Multi-head attention

### 3. Ensemble Model (`DrugRNANet_Ensemble`)

- Combines multiple architectures
- Weighted ensemble prediction
- Improved robustness

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

# Model selection
model_type = 'GCN'  # or 'GAT' or 'Ensemble'
```

## Data Processing Pipeline

### 1. SMILES Processing

- Convert SMILES to molecular graphs using RDKit
- Extract atom features and bond information
- Create adjacency matrices and edge indices

### 2. Sequence Processing

- Convert ncRNA sequences to one-hot encoding
- Apply multi-scale CNN with different kernel sizes
- Extract sequence features

### 3. Graph Construction

- Build molecular graphs from SMILES
- Create node features and edge indices
- Prepare for GNN processing

## Model Training

The training process includes:

1. **Data Loading**: Load preprocessed data
2. **Model Initialization**: Create selected model architecture
3. **Training Loop**: 
   - Forward pass
   - Loss computation
   - Backward pass
   - Parameter updates
4. **Evaluation**: Regular performance evaluation
5. **Model Saving**: Save best model checkpoints

## Performance Metrics

The model evaluates performance using:

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Pearson**: Pearson correlation coefficient

## Output Files

- `best_model.pth` - Best model checkpoint
- `data/` - Preprocessed data directory
- Training logs and metrics

## Dataset Support

### miRNA Dataset (Main)

- Processes miRNA sequences
- Uses miRNA-drug interaction data
- Optimized for shorter sequences

### lncRNA Dataset (Commented for Future Activation)

- Designed for longer lncRNA sequences
- Handles variable sequence lengths
- Can be activated by uncommenting relevant code

## Usage Example

```python
from models.gcn import GCNNet
from utils import load_processed_data, get_data_loaders

# Load data
drug_dict, rna_dict, train_samples, val_samples, test_samples, stats = load_processed_data('data/mirna')

# Create data loaders
train_loader, val_loader, test_loader = get_data_loaders(train_samples, val_samples, test_samples)

# Create model
model = GCNNet()

# Training
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    
    if epoch % 10 == 0:
        val_loss = validate_epoch(model, val_loader, device)
        print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
```

## Advanced Features

### 1. Multi-scale Processing

- Different kernel sizes for comprehensive feature extraction
- Captures both local and global patterns

### 2. Attention Mechanism

- Weighted feature combination
- Focus on important molecular regions

### 3. Ensemble Learning

- Combines multiple model architectures
- Improved prediction robustness

## Citation

If you use this model in your research, please cite:

```bibtex
@article{graphdta2024,
  title={GraphDTA: Graph-based Drug-Target Affinity Prediction},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions and support, please contact [your-email@domain.com].
