# NDSGCL: Neural Drug-ncRNA Subgraph Contrastive Learning

## Overview

NDSGCL is a neural network model for drug-ncRNA association prediction using subgraph contrastive learning. The model implements a staged training strategy with random negative sampling followed by hard negative mining for improved performance.

## Features

- **Staged Training Strategy**: Random sampling → Hard negative mining
- **Subgraph Contrastive Learning**: Learns from local subgraph structures
- **Prototype-based Learning**: Uses clustering for better representation learning
- **Hard Negative Mining**: Dynamically selects difficult negative samples
- **Dual Dataset Support**: Supports both miRNA and lncRNA datasets

## Model Architecture

```
Input: ncRNA-Drug Pairs
    ↓
Graph Construction
    ↓
GCN Encoder
    ↓
Contrastive Learning
    ↓
Prototype Learning
    ↓
Hard Negative Mining
    ↓
Final Prediction
```

## Key Components

### 1. Main Model (`NDSGCL`)

- Graph convolutional network encoder
- Contrastive learning module
- Prototype-based clustering

### 2. Training Strategies

- **Random Sampling**: Initial epochs with random negative samples
- **Hard Negative Mining**: Later epochs with difficult negative samples
- **Prototype Learning**: K-means clustering for representation learning

### 3. Loss Functions

- **Main Loss**: Binary cross-entropy for association prediction
- **Contrastive Loss**: Contrastive learning for representation quality
- **Prototype Loss**: Prototype-based contrastive learning
- **Hard Negative Loss**: Margin loss for hard negative mining

## File Structure

```
NDSGCL/
├── main.py              # Main training script
├── NDSGCL.py           # Model architecture
├── util.py             # Utility functions
├── config.conf         # Configuration file
├── interaction_table.py # Interaction table management
├── negative_manager.py  # Negative sample management
└── README.md          # This file
```

## Usage

### Prerequisites

```bash
pip install torch torch-geometric
pip install faiss-cpu
pip install scikit-learn pandas numpy
pip install scipy
```

### Configuration

Edit `config.conf` to adjust model parameters:

```ini
# Model parameters
embbedding.size=64
gnn_layer=3
learnRate=0.001
num.max.epoch=100
batch_size=256

# Contrastive learning parameters
NCL=on -n_layer 3 -tau 0.1 -ssl_reg 0.1 -hyper_layers 2 -alpha 0.1 -proto_reg 0.01 -num_clusters 10

# Data paths
data_path=../../Dataset/
```

### Data Preparation

1. Place your data files in the `../../Dataset/` directory:
   - `MiPair.csv` - miRNA-drug interaction pairs
   - `miRNA_with_sequence.xlsx` - miRNA sequences
   - `mi_drug_with_SMILES.csv` - Drug SMILES

2. For lncRNA data:
   - `LncPair.csv` - lncRNA-drug interaction pairs
   - `lncRNA_with_sequence.xlsx` - lncRNA sequences
   - `lnc_drug_with_SMILES.xlsx` - Drug SMILES

### Training

```bash
python main.py
```

## Model Parameters

| Parameter         | Description                         | Default Value |
| ----------------- | ----------------------------------- | ------------- |
| `embbedding.size` | Embedding dimension                 | 64            |
| `gnn_layer`       | Number of GNN layers                | 3             |
| `learnRate`       | Learning rate                       | 0.001         |
| `num.max.epoch`   | Number of epochs                    | 100           |
| `batch_size`      | Batch size                          | 256           |
| `tau`             | Temperature parameter               | 0.1           |
| `ssl_reg`         | Contrastive learning regularization | 0.1           |
| `proto_reg`       | Prototype learning regularization   | 0.01          |
| `num_clusters`    | Number of clusters                  | 10            |

## Training Strategy

### Stage 1: Random Negative Sampling (Epochs 1-3)

- Uses random negative sampling
- 1:1 positive to negative ratio
- Allows model to converge initially

### Stage 2: Hard Negative Mining (Epochs 4+)

- Uses model predictions to select hard negatives
- 1:2 positive to negative ratio
- Improves learning from difficult samples

## Key Functions

### 1. `get_training_strategy()`

Determines which training strategy to use based on current epoch and model state.

### 2. `train_one_epoch_with_strategy()`

Executes one training epoch using the specified strategy.

### 3. `evaluate_model()`

Evaluates model performance on test data.

## Data Processing

### 1. Data Loading

- Loads interaction pairs from CSV files
- Creates ID mappings for ncRNAs and drugs
- Splits data into training and test sets

### 2. Graph Construction

- Builds interaction graphs
- Creates adjacency matrices
- Prepares data for GCN processing

### 3. Negative Sample Generation

- Random negative sampling for initial training
- Hard negative mining for later epochs
- Dynamic negative sample selection

## Performance Metrics

The model evaluates performance using:

- **AUC**: Area Under the ROC Curve
- **AUPR**: Area Under the Precision-Recall Curve
- **Training Loss**: Average loss per epoch
- **Convergence**: Model convergence monitoring

## Output Files

- `best_model_epoch_*.pth` - Best model checkpoints
- Training logs with performance metrics
- Configuration files

## Dataset Support

### miRNA Dataset (Main)

- Uses `MiPair.csv` for interaction pairs
- Processes miRNA sequences
- Optimized for shorter sequences

### lncRNA Dataset (Commented for Future Activation)

- Uses `LncPair.csv` for interaction pairs
- Processes lncRNA sequences
- Handles longer sequences
- Can be activated by uncommenting relevant code

## Advanced Features

### 1. Hard Negative Mining

- Dynamically selects difficult negative samples
- Improves model learning from challenging cases
- Uses model predictions to identify hard negatives

### 2. Prototype Learning

- K-means clustering for representation learning
- Prototype-based contrastive learning
- Better representation quality

### 3. Staged Training

- Gradual transition from random to hard negative sampling
- Prevents early overfitting
- Improves final performance

## Usage Example

```python
from NDSGCL import NDSGCL
from util import load_config, load_data

# Load configuration and data
config = load_config('config.conf')
data = load_data(config)

# Create model
model = NDSGCL(config, data)

# Training with staged strategy
for epoch in range(epochs):
    strategy, negative_ratio = get_training_strategy(epoch, model)
    avg_loss = train_one_epoch_with_strategy(model, training_data, strategy, ...)
    
    if epoch % 5 == 0:
        auc, aupr = evaluate_model(model, data)
```

## Citation

If you use this model in your research, please cite:

```bibtex
@article{ndsgcl2024,
  title={NDSGCL: Neural Drug-ncRNA Subgraph Contrastive Learning},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License.

## Contact

For questions and support, please contact [your-email@domain.com].
