import torch  # Import PyTorch for neural network and training
import torch.nn as nn  # Import neural network modules
import torch.optim as optim  # Import optimizers
import pandas as pd  # Import pandas for CSV handling
import numpy as np  # Import numpy for numerical operations
import matplotlib.pyplot as plt  # Import matplotlib for plotting
from sklearn.model_selection import train_test_split  # Import train_test_split for data splitting
from sklearn.preprocessing import StandardScaler  # Import StandardScaler for feature scaling
import os  # Import os for file path operations

# Set random seeds for reproducibility
torch.manual_seed(42)  # Set PyTorch random seed
np.random.seed(42)  # Set NumPy random seed

# Define file paths
pseudobulk_csv_path = "pseudobulk_train.csv"  # Path to pseudobulk CSV file
embeddings_file_path = "competition_support_set/ESM2_pert_features.pt"  # Path to embeddings file
output_model_path = "vcc_predictor.pt"  # Path for saving the trained model

# Check if files exist
if not os.path.exists(pseudobulk_csv_path):
    print(f"Error: {pseudobulk_csv_path} not found!")
    exit(1)

if not os.path.exists(embeddings_file_path):
    print(f"Error: {embeddings_file_path} not found!")
    exit(1)

print("=" * 60)
print("Phase 5: Training the Predictor")
print("=" * 60)

# Step 1: Load the data
print("\nStep 1: Loading data...")
print("-" * 60)

# Load pseudobulk CSV (target outputs)
pseudobulk_df = pd.read_csv(pseudobulk_csv_path, index_col=0)  # Load CSV with first column as index (target_gene names)
print(f"Loaded pseudobulk CSV: {pseudobulk_df.shape[0]} target genes × {pseudobulk_df.shape[1]} genes")

# Load embeddings (input vectors)
embeddings_data = torch.load(embeddings_file_path, map_location='cpu')  # Load embeddings dictionary
print(f"Loaded embeddings: {len(embeddings_data)} genes")

# Step 2: Data Preparation - Match gene names and prepare inputs/outputs
print("\nStep 2: Matching gene names and preparing data...")
print("-" * 60)

# Get target gene names from CSV index
target_genes = pseudobulk_df.index.tolist()  # Get list of target gene names from CSV

# Prepare input embeddings and output expressions
X_list = []  # List to store input embeddings
y_list = []  # List to store output expressions
matched_genes = []  # List to store successfully matched genes

for gene in target_genes:  # Iterate through each target gene
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
        embedding = torch.zeros(5120)  # Create zero vector of size 5120
    elif embedding_key in embeddings_data:  # Check if gene exists in embeddings
        embedding = embeddings_data[embedding_key]  # Get embedding from dictionary
        
        # Convert to tensor if it's not already
        if torch.is_tensor(embedding):  # Check if already a tensor
            embedding = embedding.clone().detach()  # Clone and detach to avoid gradients
        else:
            embedding = torch.tensor(embedding, dtype=torch.float32)  # Convert to tensor
        
        # Ensure embedding is 1D and has correct size
        if embedding.dim() > 1:  # Check if multi-dimensional
            embedding = embedding.flatten()  # Flatten to 1D
        
        # Ensure correct size (5120)
        if embedding.size(0) != 5120:  # Check size
            print(f"Warning: {gene} embedding has size {embedding.size(0)}, expected 5120")
            continue  # Skip this gene if size doesn't match
    else:
        print(f"Warning: {gene} (key: {embedding_key}) not found in embeddings, skipping")
        continue  # Skip genes not found in embeddings
    
    # Get output expression (pseudobulk row for this gene)
    expression_row = pseudobulk_df.loc[gene].values  # Get expression values for this gene
    expression_tensor = torch.tensor(expression_row, dtype=torch.float32)  # Convert to tensor
    
    # Store matched data
    X_list.append(embedding)  # Add embedding to input list
    y_list.append(expression_tensor)  # Add expression to output list
    matched_genes.append(gene)  # Track matched gene

print(f"Successfully matched {len(matched_genes)} genes")  # Print number of matched genes

# Convert lists to tensors
X = torch.stack(X_list)  # Stack embeddings into a tensor (genes × 5120)
y = torch.stack(y_list)  # Stack expressions into a tensor (genes × 18080)

print(f"Input tensor shape: {X.shape}")  # Print input shape
print(f"Output tensor shape: {y.shape}")  # Print output shape

# Step 3: Split data into training and testing sets (130 training, 21 testing)
print("\nStep 3: Splitting data into training and testing sets...")
print("-" * 60)

# Convert to numpy for sklearn's train_test_split
X_np = X.numpy()  # Convert to numpy
y_np = y.numpy()  # Convert to numpy

# Split: 130 training, 21 testing (approximately 86% train, 14% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=21, random_state=42, shuffle=True
)  # Split data with fixed test size

# Convert back to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)  # Convert training inputs to tensor
X_test = torch.tensor(X_test, dtype=torch.float32)  # Convert test inputs to tensor
y_train = torch.tensor(y_train, dtype=torch.float32)  # Convert training outputs to tensor
y_test = torch.tensor(y_test, dtype=torch.float32)  # Convert test outputs to tensor

print(f"Training set: {X_train.shape[0]} samples")  # Print training set size
print(f"Testing set: {X_test.shape[0]} samples")  # Print test set size

# Step 4: Define the Neural Network Model
print("\nStep 4: Defining the neural network model...")
print("-" * 60)

class GenePredictor(nn.Module):  # Define neural network class
    def __init__(self, input_size=5120, output_size=18080, hidden_size1=256, hidden_size2=256, dropout_rate=0.3):
        super(GenePredictor, self).__init__()  # Initialize parent class
        
        # First hidden layer: input_size -> hidden_size1
        self.fc1 = nn.Linear(input_size, hidden_size1)  # First fully connected layer
        self.relu1 = nn.ReLU()  # ReLU activation function
        self.dropout1 = nn.Dropout(dropout_rate)  # Dropout layer to prevent overfitting
        
        # Second hidden layer: hidden_size1 -> hidden_size2
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)  # Second fully connected layer
        self.relu2 = nn.ReLU()  # ReLU activation function
        self.dropout2 = nn.Dropout(dropout_rate)  # Dropout layer
        
        # Output layer: hidden_size2 -> output_size
        self.fc3 = nn.Linear(hidden_size2, output_size)  # Output layer (no activation, linear)
    
    def forward(self, x):  # Define forward pass
        # First layer
        x = self.fc1(x)  # Apply first linear transformation
        x = self.relu1(x)  # Apply ReLU activation
        x = self.dropout1(x)  # Apply dropout (only active during training)
        
        # Second layer
        x = self.fc2(x)  # Apply second linear transformation
        x = self.relu2(x)  # Apply ReLU activation
        x = self.dropout2(x)  # Apply dropout
        
        # Output layer
        x = self.fc3(x)  # Apply output linear transformation
        return x  # Return predictions

# Create model instance
model = GenePredictor(input_size=5120, output_size=18080)  # Create model with correct input/output sizes
print(f"Model created: {model}")  # Print model architecture

# Step 5: Set up training
print("\nStep 5: Setting up training...")
print("-" * 60)

# Loss function: Mean Squared Error (MSE) for regression
criterion = nn.MSELoss()  # Define loss function (Mean Squared Error)

# Optimizer: Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Create Adam optimizer with learning rate 0.001

# Training parameters
num_epochs = 50  # Number of training epochs
batch_size = 32  # Batch size for training

# Lists to store loss history for plotting
train_losses = []  # List to store training losses
test_losses = []  # List to store test losses

print(f"Loss function: MSE (Mean Squared Error)")  # Print loss function
print(f"Optimizer: Adam (learning rate: 0.001)")  # Print optimizer
print(f"Training for {num_epochs} epochs...")  # Print number of epochs

# Step 6: Training loop
print("\nStep 6: Training the model...")
print("-" * 60)

for epoch in range(num_epochs):  # Iterate through each epoch
    # Set model to training mode (enables dropout)
    model.train()  # Set model to training mode
    
    # Training phase
    train_loss = 0.0  # Initialize training loss for this epoch
    
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):  # Iterate through batches
        # Get batch
        batch_X = X_train[i:i+batch_size]  # Get batch of inputs
        batch_y = y_train[i:i+batch_size]  # Get batch of outputs
        
        # Forward pass
        optimizer.zero_grad()  # Zero gradients
        predictions = model(batch_X)  # Get predictions from model
        loss = criterion(predictions, batch_y)  # Calculate loss
        
        # Backward pass
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        
        train_loss += loss.item()  # Accumulate loss
    
    # Average training loss for this epoch
    avg_train_loss = train_loss / (len(X_train) / batch_size)  # Calculate average loss
    train_losses.append(avg_train_loss)  # Store training loss
    
    # Evaluation phase (on test set)
    model.eval()  # Set model to evaluation mode (disables dropout)
    with torch.no_grad():  # Disable gradient computation for evaluation
        test_predictions = model(X_test)  # Get predictions on test set
        test_loss = criterion(test_predictions, y_test).item()  # Calculate test loss
        test_losses.append(test_loss)  # Store test loss
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0 or epoch == 0:  # Print every 10 epochs or first epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Test Loss: {test_loss:.6f}")

print("\nTraining complete!")  # Confirm training finished

# Step 7: Save the trained model
print("\nStep 7: Saving the trained model...")
print("-" * 60)

torch.save(model.state_dict(), output_model_path)  # Save model weights to file
print(f"✓ Model saved to: {output_model_path}")  # Confirm save

# Step 8: Plot training and test loss over time
print("\nStep 8: Creating loss plot...")
print("-" * 60)

plt.figure(figsize=(10, 6))  # Create figure with specified size
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o')  # Plot training loss
plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', marker='s')  # Plot test loss
plt.xlabel('Epoch')  # Set x-axis label
plt.ylabel('Loss (MSE)')  # Set y-axis label
plt.title('Model Training: Loss Over Time')  # Set plot title
plt.legend()  # Show legend
plt.grid(True, alpha=0.3)  # Add grid for easier reading
plt.tight_layout()  # Adjust layout to prevent label cutoff

# Save the plot
plot_path = "training_loss_plot.png"  # Path for saving plot
plt.savefig(plot_path, dpi=150)  # Save plot as PNG file
print(f"✓ Loss plot saved to: {plot_path}")  # Confirm plot saved

# Show the plot (optional - comment out if running in headless environment)
# plt.show()  # Display the plot

print("\n" + "=" * 60)
print("Training Summary:")
print("=" * 60)
print(f"Final Training Loss: {train_losses[-1]:.6f}")  # Print final training loss
print(f"Final Test Loss: {test_losses[-1]:.6f}")  # Print final test loss
print(f"Model saved to: {output_model_path}")  # Print model path
print(f"Loss plot saved to: {plot_path}")  # Print plot path
print("=" * 60)
