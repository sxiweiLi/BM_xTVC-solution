import scanpy as sc  # Import scanpy library for reading h5ad files
import pandas as pd  # Import pandas for DataFrames and CSV operations
import os  # Import os for file path operations
import numpy as np  # Import numpy for numerical operations

# Define the file paths
h5_file_path = "competition_support_set/competition_train.h5"  # Path to the training H5 file
output_csv_path = "pseudobulk_train.csv"  # Path for the output CSV file

# Check if input file exists
if not os.path.exists(h5_file_path):
    print(f"Error: {h5_file_path} not found!")
    exit(1)

print("=" * 60)
print("Phase 4: Pseudo-bulking")
print("=" * 60)

# Step 1: Load the training data using scanpy
print("\nStep 1: Loading competition_train.h5...")
print("-" * 60)
adata = sc.read_h5ad(h5_file_path)  # Load the h5ad-compatible H5 file
print(f"Loaded data: {adata.n_obs} cells, {adata.n_vars} genes")  # Print data dimensions

# Step 2: Get the target_gene column and handle categorical encoding
print("\nStep 2: Extracting target_gene column...")
print("-" * 60)

# Check if 'target_gene' column exists
if 'target_gene' not in adata.obs.columns:
    print("Error: 'target_gene' column not found in adata.obs!")
    print(f"Available columns: {list(adata.obs.columns)}")
    exit(1)

# Get the target_gene column
target_gene_col = adata.obs['target_gene']  # Extract the target_gene column

# Handle categorical encoding - convert to strings for grouping
if pd.api.types.is_categorical_dtype(target_gene_col):  # Check if it's categorical
    target_genes = target_gene_col.astype('str')  # Convert categorical to string (decodes codes to gene names)
    print(f"Converted categorical target_gene to strings")
else:
    target_genes = target_gene_col  # Use as is if already strings

# Get unique target genes for reporting
unique_target_genes = pd.unique(target_genes)  # Get unique target gene names
print(f"Found {len(unique_target_genes)} unique target genes")  # Print count of unique genes

# Step 3: Convert expression matrix to DataFrame (handles sparse matrices)
print("\nStep 3: Converting expression matrix to DataFrame...")
print("-" * 60)
print("Note: This may use significant memory for large datasets...")  # Memory warning

# Convert sparse matrix to dense DataFrame using scanpy's to_df method
# This handles sparse matrices efficiently
expression_df = adata.to_df()  # Convert AnnData expression matrix to pandas DataFrame (cells x genes)
print(f"Expression DataFrame shape: {expression_df.shape}")  # Print DataFrame dimensions

# Add target_gene column to the expression DataFrame for grouping
expression_df['target_gene'] = target_genes.values  # Add target_gene as a new column

# Step 4: Group by target_gene and calculate mean expression
print("\nStep 4: Grouping cells by target_gene and calculating mean expression...")
print("-" * 60)

# Group by target_gene and calculate mean for all gene expression columns
# This computes the average expression across all cells for each target_gene group
pseudobulk_df = expression_df.groupby('target_gene').mean()  # Group by target_gene and compute mean expression

print(f"Pseudo-bulked DataFrame shape: {pseudobulk_df.shape}")  # Print resulting shape (target_genes x genes)
print(f"  Rows (target genes): {pseudobulk_df.shape[0]}")  # Number of target genes
print(f"  Columns (genes): {pseudobulk_df.shape[1]}")  # Number of genes (should be 18,080)

# The DataFrame now has:
# - Index: target_gene names (rows)
# - Columns: gene names (columns)
# - Values: mean expression values

# Step 5: Save to CSV
print("\nStep 5: Saving pseudobulk data to CSV...")
print("-" * 60)

# Save the DataFrame to CSV
# index=True saves the target_gene names as the first column (row names)
pseudobulk_df.to_csv(output_csv_path, index=True)  # Save DataFrame to CSV file

print(f"✓ Saved pseudobulk data to: {output_csv_path}")  # Confirm save

# Print summary information
print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print(f"Input: {adata.n_obs} cells × {adata.n_vars} genes")  # Input dimensions
print(f"Output: {pseudobulk_df.shape[0]} target genes × {pseudobulk_df.shape[1]} genes")  # Output dimensions
print(f"Output file: {output_csv_path}")  # Output file path

# Show a preview of the pseudobulk data
print("\nPreview of pseudobulk data (first 5 rows, first 5 columns):")
print("-" * 60)
print(pseudobulk_df.iloc[:5, :5])  # Print first 5 rows and 5 columns as preview

print("\n" + "=" * 60)
print("Pseudo-bulking complete!")
print("=" * 60)
