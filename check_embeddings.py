import torch  # Import torch library for loading .pt files
import scanpy as sc  # Import scanpy for loading training data
import pandas as pd  # Import pandas for data manipulation
import numpy as np  # Import numpy for array operations
import os  # Import os for file path operations

# Define the file paths
h5_file_path = "competition_support_set/competition_train.h5"  # Path to training data
embeddings_file_path = "competition_support_set/ESM2_pert_features.pt"  # Path to embeddings file

# Check if files exist
if not os.path.exists(h5_file_path):
    print(f"Error: {h5_file_path} not found!")
    exit(1)

if not os.path.exists(embeddings_file_path):
    print(f"Error: {embeddings_file_path} not found!")
    exit(1)

print("=" * 60)
print("Step 1: Loading training data to get gene list...")
print("=" * 60)

# Load the training data to extract the target genes
adata = sc.read_h5ad(h5_file_path)  # Load the training data file

# Extract unique target genes from the training data
if 'target_gene' in adata.obs.columns:
    target_gene_col = adata.obs['target_gene']  # Get the target_gene column
    
    # Handle categorical encoding if present
    if pd.api.types.is_categorical_dtype(target_gene_col):  # Check if it's categorical
        training_genes = target_gene_col.cat.categories.tolist()  # Get unique categories (gene names)
    else:
        training_genes = pd.unique(target_gene_col).tolist()  # Get unique gene names directly
    
    # Sort the genes for consistency
    training_genes = sorted(training_genes)  # Sort alphabetically
    
    print(f"Loaded {len(training_genes)} unique genes from training data")
else:
    print("Error: 'target_gene' column not found in training data!")
    exit(1)

print("\n" + "=" * 60)
print("Step 2: Loading ESM2 embeddings file...")
print("=" * 60)

# Load the embeddings file using torch.load()
embeddings_data = torch.load(embeddings_file_path, map_location='cpu')  # Load the .pt file (use CPU to avoid GPU issues)

print(f"Embeddings file loaded successfully!")
print(f"Type of loaded data: {type(embeddings_data)}")  # Print the type of the loaded data

print("\n" + "=" * 60)
print("Step 3: Exploring the structure of embeddings file...")
print("=" * 60)

# Check if it's a dictionary
if isinstance(embeddings_data, dict):  # Check if the loaded data is a dictionary
    print("✓ The embeddings file is a dictionary")
    print(f"  Number of keys (genes) in dictionary: {len(embeddings_data)}")  # Count the number of keys
    
    # Print first few keys to see the structure
    sample_keys = list(embeddings_data.keys())[:5]  # Get first 5 keys
    print(f"  Sample keys (first 5): {sample_keys}")  # Print sample keys
    
    # Check the type and shape of values
    first_key = list(embeddings_data.keys())[0]  # Get the first key
    first_value = embeddings_data[first_key]  # Get the value for the first key
    print(f"  Type of values: {type(first_value)}")  # Print the type of values
    
    # Check if values are tensors
    if torch.is_tensor(first_value):  # Check if it's a PyTorch tensor
        print(f"  Value is a PyTorch tensor with shape: {first_value.shape}")  # Print tensor shape
        print(f"  Tensor dtype: {first_value.dtype}")  # Print data type
    elif isinstance(first_value, (list, tuple)):  # Check if it's a list or tuple
        print(f"  Value is a {type(first_value).__name__} with length: {len(first_value)}")  # Print length
    else:
        print(f"  Value type: {type(first_value)}")  # Print the type
else:
    print("✗ The embeddings file is NOT a dictionary")
    print(f"  Actual type: {type(embeddings_data)}")  # Print the actual type

print("\n" + "=" * 60)
print("Step 4: Checking if training genes are present in embeddings...")
print("=" * 60)

# Check which training genes are present in the embeddings
if isinstance(embeddings_data, dict):  # Only proceed if it's a dictionary
    embedding_keys = set(embeddings_data.keys())  # Convert dictionary keys to a set for fast lookup
    training_genes_set = set(training_genes)  # Convert training genes to a set
    
    # Find genes that are present in both
    genes_found = training_genes_set.intersection(embedding_keys)  # Find intersection (genes in both sets)
    genes_missing = training_genes_set - embedding_keys  # Find difference (genes in training but not in embeddings)
    
    print(f"Training genes present in embeddings: {len(genes_found)}/{len(training_genes)}")  # Print count
    print(f"Training genes missing from embeddings: {len(genes_missing)}/{len(training_genes)}")  # Print count of missing
    
    # Report missing genes if any
    if genes_missing:
        print(f"\nMissing genes ({len(genes_missing)}):")
        for gene in sorted(genes_missing):  # Sort and print missing genes
            print(f"  - {gene}")
    else:
        print("\n✓ All training genes are present in the embeddings file!")

print("\n" + "=" * 60)
print("Step 5: Examining embedding for a specific gene (AKT2)...")
print("=" * 60)

# Try to get the embedding for 'AKT2'
test_gene = 'AKT2'  # The gene name to examine

if isinstance(embeddings_data, dict):  # Check if it's a dictionary
    if test_gene in embeddings_data:  # Check if the gene exists in embeddings
        gene_embedding = embeddings_data[test_gene]  # Get the embedding for the test gene
        
        # Convert to numpy array if it's a tensor
        if torch.is_tensor(gene_embedding):  # Check if it's a PyTorch tensor
            embedding_array = gene_embedding.numpy()  # Convert tensor to numpy array
        elif isinstance(gene_embedding, (list, tuple)):  # Check if it's a list or tuple
            embedding_array = np.array(gene_embedding)  # Convert to numpy array
        else:
            embedding_array = gene_embedding  # Use as is
        
        embedding_length = len(embedding_array)  # Get the length of the embedding
        print(f"Found gene '{test_gene}' in embeddings")  # Confirm gene was found
        print(f"Embedding length: {embedding_length}")  # Print the length
        
        # Check if length is 1280
        if embedding_length == 1280:  # Check if length matches expected value
            print("✓ Embedding length is 1280 (as expected)")  # Confirm correct length
        else:
            print(f"✗ Embedding length is {embedding_length}, expected 1280")  # Report mismatch
        
        # Print first 5 numbers
        print(f"\nFirst 5 numbers of embedding:")
        for i, value in enumerate(embedding_array[:5], 1):  # Get first 5 values
            print(f"  {i}. {value:.6f}")  # Print with 6 decimal places
    else:
        print(f"✗ Gene '{test_gene}' not found in embeddings")  # Report if gene is missing
        # Try to find a gene that exists
        if genes_found:
            alternative_gene = sorted(genes_found)[0]  # Get first available gene
            print(f"\nTrying alternative gene: '{alternative_gene}'")  # Print alternative
            gene_embedding = embeddings_data[alternative_gene]  # Get embedding
            
            # Convert to numpy array if it's a tensor
            if torch.is_tensor(gene_embedding):  # Check if it's a tensor
                embedding_array = gene_embedding.numpy()  # Convert to numpy
            elif isinstance(gene_embedding, (list, tuple)):  # Check if it's a list
                embedding_array = np.array(gene_embedding)  # Convert to array
            else:
                embedding_array = gene_embedding  # Use as is
            
            embedding_length = len(embedding_array)  # Get length
            print(f"Embedding length: {embedding_length}")  # Print length
            
            if embedding_length == 1280:  # Check length
                print("✓ Embedding length is 1280 (as expected)")  # Confirm
            else:
                print(f"✗ Embedding length is {embedding_length}, expected 1280")  # Report mismatch
            
            print(f"\nFirst 5 numbers of embedding:")
            for i, value in enumerate(embedding_array[:5], 1):  # Get first 5
                print(f"  {i}. {value:.6f}")  # Print values

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
if isinstance(embeddings_data, dict):
    print(f"Total genes in embeddings file: {len(embeddings_data)}")
    print(f"Training genes found: {len(genes_found)}/{len(training_genes)}")
    if genes_missing:
        print(f"Training genes missing: {len(genes_missing)}")
    else:
        print("All training genes are present!")
else:
    print("Could not analyze embeddings structure (not a dictionary)")

print("=" * 60)
