# AGCLNDA: Adaptive Graph Contrastive Learning for ncRNA-Drug Association Prediction

## Overview

AGCLNDA is a deep learning model for predicting ncRNA-drug associations using adaptive graph contrastive learning. The model combines graph neural networks with contrastive learning to learn effective representations of ncRNAs and drugs.

## Features

- **Adaptive Graph Learning**: Uses graph neural networks to capture structural information
- **Contrastive Learning**: Implements contrastive learning for better representation learning
- **Multi-generator Architecture**: Includes VGAE and DenoisingNet generators
- **Dual Dataset Support**: Supports both miRNA and lncRNA datasets
- **Information Bottleneck**: Implements information bottleneck loss for regularization

## Model Architecture

```
Input Data (ncRNA-Drug Pairs)
    ↓
Graph Neural Network Encoder
    ↓
Contrastive Learning Module
    ↓
Information Bottleneck
    ↓
Multi-generator Training
    ↓
Final Prediction
```

## Key Components

### 1. Main Model (`AGCLNDA`)

- Graph convolutional network for feature extraction
- Adaptive learning mechanism for different ncRNA types

### 2. Generators

- **VGAE Generator**: Variational Graph Autoencoder for data augmentation
- **DenoisingNet Generator**: Denoising network for robust learning

### 3. Loss Functions

- **Main Loss**: Binary cross-entropy for association prediction
- **Contrastive Loss**: Contrastive learning for representation quality
- **Information Bottleneck Loss**: Regularization for better generalization
- **Reconstruction Loss**: Generator-specific reconstruction losses

## File Structure

```
AGCLNDA/
├── Main.py              # Main training script
├── Model.py             # Model architecture definitions
├── DataHandler.py       # Data loading and preprocessing
├── Params.py            # Configuration parameters
└── README.md           # This file
```

## Usage

### Prerequisites

```bash
pip install torch torch-geometric
pip install scikit-learn pandas numpy
pip install scipy faiss-cpu
```

### Data Preparation

1. Place your data files in the `../../Dataset/` directory:
   - `MiPair.csv` - miRNA-drug interaction pairs
   - `MiRNA_features.csv` - miRNA features
   - `MiDrug_features.csv` - Drug features

2. For lncRNA data:
   - `LncPair.csv` - lncRNA-drug interaction pairs
   - `lncRNA_with_sequence.xlsx` - lncRNA sequences
   - `lnc_drug_with_SMILES.xlsx` - Drug SMILES

### Training

```bash
python Main.py
```

### Configuration

Edit `Params.py` to adjust model parameters:

```python
parser.add_argument('--nc', default=1000, type=int, help='number of ncRNAs')
parser.add_argument('--drug', default=1000, type=int, help='number of drugs')
parser.add_argument('--latdim', default=64, type=int, help='embedding dimension')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--epoch', default=100, type=int, help='number of epochs')
```

## Model Parameters

| Parameter   | Description                           | Default Value |
| ----------- | ------------------------------------- | ------------- |
| `nc`        | Number of ncRNAs                      | 1000          |
| `drug`      | Number of drugs                       | 1000          |
| `latdim`    | Embedding dimension                   | 64            |
| `gnn_layer` | Number of GNN layers                  | 3             |
| `lr`        | Learning rate                         | 0.001         |
| `epoch`     | Number of training epochs             | 100           |
| `batch`     | Batch size                            | 256           |
| `ssl_reg`   | Contrastive learning regularization   | 0.1           |
| `ib_reg`    | Information bottleneck regularization | 0.01          |

## Output

The model outputs:

- **AUC**: Area Under the ROC Curve
- **AUPR**: Area Under the Precision-Recall Curve
- **Model Checkpoints**: Saved in `best_model_epoch_*.pth`

## Dataset Support

### miRNA Dataset

- Uses `MiPair.csv` for interaction pairs
- Loads miRNA features from `MiRNA_features.csv`
- Loads drug features from `MiDrug_features.csv`

### lncRNA Dataset (Commented for Future Activation)

- Uses `LncPair.csv` for interaction pairs
- Loads lncRNA sequences from `lncRNA_with_sequence.xlsx`
- Loads drug SMILES from `lnc_drug_with_SMILES.xlsx`

## Performance

The model achieves competitive performance on ncRNA-drug association prediction tasks:

- **miRNA Dataset**: High accuracy and AUC scores
- **lncRNA Dataset**: Robust performance on longer sequences

## Citation

If you use this model in your research, please cite the original paper:

```bibtex
@article{agclnda2024,
  title={AGCLNDA: Adaptive Graph Contrastive Learning for ncRNA-Drug Association Prediction},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and support, please contact [your-email@domain.com].
