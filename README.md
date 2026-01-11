# Virtual Cell Challenge Solution

A machine learning solution for the Virtual Cell Challenge that predicts gene expression profiles from protein embeddings using ESM-2 (5120-dimensional) protein embeddings and a PyTorch multi-layer perceptron (MLP).

## Overview

This project implements a neural network-based approach to predict bulk gene expression profiles from protein embeddings. The solution leverages state-of-the-art protein language model embeddings (ESM-2) as input features and trains a deep learning model to predict expression across 18,080 genes.

## Methodology

The solution follows a structured pipeline consisting of four main phases:

### Phase 1: Data Loading and Exploration
- Load training data from `competition_train.h5` using scanpy
- Extract target genes and verify data structure
- Explore the dataset dimensions and gene annotations

### Phase 2: Pseudo-bulking
- Group single-cell data by target gene
- Calculate mean expression profiles for each target gene group
- Generate aggregated pseudo-bulk expression matrices (151 target genes × 18,080 genes)

### Phase 3: Embedding Verification
- Load ESM-2 protein embeddings (5120-dimensional vectors)
- Verify embedding structure and gene name mappings
- Handle special cases (e.g., TAZ → WWTR1 mapping, non-targeting genes)
- Validate that all required genes are present in the embeddings

### Phase 4: Model Training
- Design and train a PyTorch MLP with:
  - Input size: 5120 (ESM-2 embedding dimension)
  - Output size: 18,080 (number of genes)
  - Architecture: 2 hidden layers (256 neurons each) with ReLU activation and dropout (0.3)
  - Loss function: Mean Squared Error (MSE)
  - Optimizer: Adam (learning rate: 0.001)
- Training configuration:
  - Training set: 130 target genes
  - Test set: 21 target genes
  - Epochs: 50
  - Batch size: 32

## Results

The trained model achieved a **final training loss of 0.036** (MSE), demonstrating successful learning of the mapping from protein embeddings to gene expression profiles.

## Project Structure

```
.
├── competition_support_set/       # Competition data files
│   ├── competition_train.h5       # Training data
│   ├── competition_val_template.h5ad  # Validation template
│   ├── ESM2_pert_features.pt     # ESM-2 protein embeddings (5120-d)
│   └── gene_names.csv            # Gene name mappings
├── load_data.py                  # Phase 1: Data loading script
├── create_pseudobulk.py          # Phase 2: Pseudo-bulking script
├── check_embeddings.py           # Phase 3: Embedding verification script
├── train_model.py                # Phase 4: Model training script
├── generate_submission.py        # Submission generation script
├── vcc_predictor.pt              # Trained model weights
├── pseudobulk_train.csv          # Generated pseudo-bulk data
├── submission_final.h5ad         # Final submission file
└── training_loss_plot.png        # Training loss visualization
```

## Requirements

```bash
pip install torch
pip install scanpy
pip install pandas
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install scipy
pip install cell-eval  # For submission validation
```

## Usage

### 1. Data Loading and Exploration
```bash
python load_data.py
```
Loads and explores the training data structure, displays unique target genes, and verifies data integrity.

### 2. Pseudo-bulking
```bash
python create_pseudobulk.py
```
Generates pseudo-bulk expression profiles by averaging single-cell data grouped by target gene. Outputs `pseudobulk_train.csv`.

### 3. Embedding Verification
```bash
python check_embeddings.py
```
Verifies ESM-2 embeddings, checks gene name mappings, and validates that all training genes are present.

### 4. Model Training
```bash
python train_model.py
```
Trains the PyTorch MLP model on pseudo-bulk data. Generates `vcc_predictor.pt` and `training_loss_plot.png`.

### 5. Generate Submission
```bash
python generate_submission.py
```
Generates the final submission file (`submission_final.h5ad`) by:
- Loading the validation template
- Mapping target genes to ESM-2 embeddings
- Running predictions through the trained model
- Validating the output format using cell-eval

## Technical Details

### Model Architecture

- **Input Layer**: 5120 neurons (ESM-2 embedding dimension)
- **Hidden Layer 1**: 256 neurons + ReLU activation + Dropout (0.3)
- **Hidden Layer 2**: 256 neurons + ReLU activation + Dropout (0.3)
- **Output Layer**: 18,080 neurons (linear, no activation)

### Data Handling

- Special gene mappings:
  - `TAZ` → `WWTR1` (for embedding lookup)
  - `non-targeting` → zero vector (5120 zeros)
- Handles sparse matrices efficiently for memory optimization
- Categorical encoding/decoding for target gene annotations

## License

This project is part of a solution for the Virtual Cell Challenge competition.

## Author

Siwei Li
