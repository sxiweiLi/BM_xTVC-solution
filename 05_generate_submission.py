import torch  # Import PyTorch for neural network and model loading
import torch.nn as nn  # Import neural network modules
import scanpy as sc  # Import scanpy for reading/writing h5ad files
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for numerical operations
import scipy.sparse as sp  # Import scipy sparse for sparse matrix operations
import os  # Import os for file path operations
import subprocess  # Import subprocess for running cell-eval command

# Define file paths
template_file_path = "competition_support_set/competition_val_template.h5ad"  # Path to validation template
embeddings_file_path = "competition_support_set/ESM2_pert_features.pt"  # Path to embeddings file
model_file_path = "vcc_predictor.pt"  # Path to trained model
output_file_path = "submission_final.h5ad"  # Path for output submission file

# Check if files exist
if not os.path.exists(template_file_path):
    print(f"Error: {template_file_path} not found!")
    exit(1)

if not os.path.exists(embeddings_file_path):
    print(f"Error: {embeddings_file_path} not found!")
    exit(1)

if not os.path.exists(model_file_path):
    print(f"Error: {model_file_path} not found!")
    exit(1)

print("=" * 60)
print("Generating Submission for Virtual Cell Challenge")
print("=" * 60)

# Step 1: Define the Neural Network Model (must match training architecture)
print("\nStep 1: Defining model architecture...")
print("-" * 60)

class GenePredictor(nn.Module):  # Define neural network class (same as training)
    def __init__(self, input_size=5120, output_size=18080, hidden_size1=256, hidden_size2=256, dropout_rate=0.3):
        super(GenePredictor, self).__init__()  # Initialize parent class
        
        # First hidden layer: input_size -> hidden_size1
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First fully connected layer
        self.relu1 = nn.ReLU()  # ReLU activation function
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer
        
        # Second hidden layer: hidden_size1 -> hidden_size2
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Second fully connected layer
        self.relu2 = nn.ReLU()  # ReLU activation function
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer
        
        # Output layer: hidden_size2 -> output_size
        self.fc3 = nn.Linear(hidden_size2, output_size)  # Output layer
    
    def forward(self, x):  # Define forward pass
        # First layer
        x = self.fc1(x)  # Apply first linear transformation
        x = self.relu1(x)  # Apply ReLU activation
        x = self.dropout1(x)  # Apply dropout (disabled during inference)
        
        # Second layer
        x = self.fc2(x)  # Apply second linear transformation
        x = self.relu2(x)  # Apply ReLU activation
        x = self.dropout2(x)  # Apply dropout
        
        # Output layer
        x = self.fc3(x)  # Apply output linear transformation
        return x  # Return predictions

# Create model instance
model = GenePredictor(input_size=5120, output_size=18080)  # Create model with correct architecture

# Load trained model weights
model.load_state_dict(torch.load(model_file_path, map_location='cpu'))  # Load model weights from file
model.eval()  # Set model to evaluation mode (disables dropout)
print(f"✓ Loaded trained model from: {model_file_path}")  # Confirm model loaded

# Step 2: Load the template file
print("\nStep 2: Loading validation template...")
print("-" * 60)

adata = sc.read_h5ad(template_file_path)  # Load the validation template
print(f"Template shape: {adata.n_obs} cells × {adata.n_vars} genes")  # Print template dimensions
print(f"Template .X type: {type(adata.X)}")  # Print type of expression matrix

# Check if target_gene column exists
if 'target_gene' not in adata.obs.columns:
    print("Error: 'target_gene' column not found in template!")
    print(f"Available columns: {list(adata.obs.columns)}")
    exit(1)

# Get target_gene column and handle categorical encoding
target_gene_col = adata.obs['target_gene']  # Extract target_gene column

# Convert categorical to strings if needed
if pd.api.types.is_categorical_dtype(target_gene_col):  # Check if categorical
    target_genes = target_gene_col.astype('str')  # Convert categorical to strings
    print("Converted categorical target_gene to strings")
else:
    target_genes = target_gene_col.values  # Use as is
    print("target_gene is already in string format")

print(f"Found {len(target_genes)} target genes in template")  # Print number of target genes

# Step 3: Load embeddings
print("\nStep 3: Loading embeddings...")
print("-" * 60)

embeddings_data = torch.load(embeddings_file_path, map_location='cpu')  # Load embeddings dictionary
print(f"✓ Loaded {len(embeddings_data)} embeddings")  # Confirm embeddings loaded

# Step 4: Map embeddings to target genes
print("\nStep 4: Mapping embeddings to target genes...")
print("-" * 60)

embedding_list = []  # List to store embeddings for each cell
missing_genes = []  # Track genes not found in embeddings

for i, gene in enumerate(target_genes):  # Iterate through each target gene
    # Handle special cases for gene name mapping
    if gene == 'TAZ':  # TAZ maps to WWTR1 in embeddings
        embedding_key = 'WWTR1'  # Use WWTR1 for TAZ
    elif gene == 'non-targeting':  # non-targeting has no embedding
        embedding_key = None  # Mark as special case
    else:
        embedding_key = gene  # Use gene name directly
    
    # Get embedding for this gene
    if embedding_key is None:  # Handle non-targeting case
        # Create a vector of all zeros for non-targeting
        embedding = torch.zeros(5120, dtype=torch.float32)  # Create zero vector of size 5120
    elif embedding_key in embeddings_data:  # Check if gene exists in embeddings
        embedding = embeddings_data[embedding_key]  # Get embedding from dictionary
        
        # Convert to tensor if it's not already
        if torch.is_tensor(embedding):  # Check if already a tensor
            embedding = embedding.clone().detach().float()  # Clone, detach, and convert to float
        else:
            embedding = torch.tensor(embedding, dtype=torch.float32)  # Convert to tensor
        
        # Ensure embedding is 1D and has correct size
        if embedding.dim() > 1:  # Check if multi-dimensional
            embedding = embedding.flatten()  # Flatten to 1D
        
        # Ensure correct size (5120)
        if embedding.size(0) != 5120:  # Check size
            print(f"Warning: {gene} embedding has size {embedding.size(0)}, expected 5120. Using zeros.")
            embedding = torch.zeros(5120, dtype=torch.float32)  # Use zeros if size doesn't match
    else:
        print(f"Warning: {gene} (key: {embedding_key}) not found in embeddings. Using zeros.")
        embedding = torch.zeros(5120, dtype=torch.float32)  # Use zeros for missing genes
        missing_genes.append(gene)  # Track missing gene
    
    embedding_list.append(embedding)  # Add embedding to list

# Stack embeddings into a batch tensor
X_batch = torch.stack(embedding_list)  # Stack embeddings (cells × 5120)
print(f"✓ Prepared {X_batch.shape[0]} embeddings for prediction")  # Confirm embeddings prepared
if missing_genes:
    print(f"Warning: {len(missing_genes)} genes used zero vectors (not found in embeddings)")

# Step 5: Run predictions through the model
print("\nStep 5: Running predictions through the model...")
print("-" * 60)

with torch.no_grad():  # Disable gradient computation for inference
    predictions = model(X_batch)  # Run predictions through model
    predictions_np = predictions.numpy()  # Convert predictions to numpy array

print(f"✓ Generated predictions with shape: {predictions_np.shape}")  # Confirm predictions generated

# Step 6: Fill the .X layer with predictions
print("\nStep 6: Filling .X layer with predictions...")
print("-" * 60)

# Check if predictions shape matches template shape
if predictions_np.shape != (adata.n_obs, adata.n_vars):  # Check shape match
    print(f"Warning: Prediction shape {predictions_np.shape} doesn't match template shape ({adata.n_obs}, {adata.n_vars})")
    print("Attempting to reshape or adjust...")
    
    # If number of genes differs, we might need to subset or pad
    if predictions_np.shape[1] != adata.n_vars:  # Check if gene count differs
        print(f"Prediction has {predictions_np.shape[1]} genes, template expects {adata.n_vars} genes")
        if predictions_np.shape[1] > adata.n_vars:  # If predictions have more genes
            predictions_np = predictions_np[:, :adata.n_vars]  # Truncate to match
            print("Truncated predictions to match template")
        else:  # If predictions have fewer genes
            # Pad with zeros (this shouldn't happen normally)
            padding = np.zeros((predictions_np.shape[0], adata.n_vars - predictions_np.shape[1]))
            predictions_np = np.hstack([predictions_np, padding])  # Pad with zeros
            print("Padded predictions with zeros to match template")

# Convert to sparse matrix format if template uses sparse (for memory efficiency)
if sp.issparse(adata.X):  # Check if template uses sparse matrix
    adata.X = sp.csr_matrix(predictions_np)  # Convert predictions to sparse CSR matrix
    print("✓ Filled .X with sparse matrix predictions")  # Confirm sparse matrix set
else:
    adata.X = predictions_np  # Set predictions as dense matrix
    print("✓ Filled .X with dense matrix predictions")  # Confirm dense matrix set

# Step 7: Save the submission file
print("\nStep 7: Saving submission file...")
print("-" * 60)

adata.write_h5ad(output_file_path)  # Save AnnData object to h5ad file
print(f"✓ Saved submission to: {output_file_path}")  # Confirm file saved

# Step 8: Run cell-eval prep to validate format
print("\nStep 8: Running cell-eval prep to validate format...")
print("-" * 60)

try:
    # Run cell-eval prep command
    result = subprocess.run(
        ['cell-eval', 'prep', output_file_path],  # Run cell-eval prep command
        capture_output=True,  # Capture output
        text=True,  # Return as text
        check=False  # Don't raise exception on error (we'll handle it)
    )
    
    # Print output
    if result.stdout:  # If there's stdout
        print(result.stdout)  # Print standard output
    
    if result.stderr:  # If there's stderr
        print("Error output:")  # Print error label
        print(result.stderr)  # Print error output
    
    if result.returncode == 0:  # Check if command succeeded
        print(f"✓ cell-eval prep completed successfully!")  # Confirm success
    else:
        print(f"⚠ cell-eval prep exited with code {result.returncode}")  # Warn about exit code
        print("Please check the output above for details")  # Suggest checking output

except FileNotFoundError:  # Handle case where cell-eval is not installed
    print("⚠ cell-eval command not found. Please install cell-eval package:")
    print("  pip install cell-eval")
    print("Skipping validation step...")  # Skip validation if not installed
except Exception as e:  # Handle other exceptions
    print(f"⚠ Error running cell-eval prep: {e}")  # Print error
    print("Please run cell-eval prep manually to validate the submission")  # Suggest manual validation

print("\n" + "=" * 60)
print("Submission Generation Summary:")
print("=" * 60)
print(f"Template: {template_file_path}")  # Print template path
print(f"Model: {model_file_path}")  # Print model path
print(f"Output: {output_file_path}")  # Print output path
print(f"Shape: {adata.n_obs} cells × {adata.n_vars} genes")  # Print dimensions
if missing_genes:
    print(f"Warning: {len(missing_genes)} genes used zero vectors")  # Warn about missing genes
print("=" * 60)
