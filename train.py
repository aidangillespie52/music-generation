# Model and Custom Dataset
from MusicGenerationModel import MusicLSTMModel
from MusicSequenceDataset import MusicSequenceDataset

# Pytorch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn

# NumPy
import numpy as np

# Quality of life
from tqdm import tqdm

# Load the saved data
X = np.load('data/notes.npy', allow_pickle=True)
print(X[:4])

# Instantiate the dataset and DataLoader
sequence_length = 16
dataset = MusicSequenceDataset(X, sequence_length=sequence_length)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
print(dataset[0])

# Assuming you have your DataLoader as `dataloader`
total_batches = len(dataloader)
print(f"Total number of batches: {total_batches}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model and move it to the correct device
model = MusicLSTMModel()
model = model.to(device)

epochs = 3
learning_rate = 0.001
criterion = nn.MSELoss()  # Mean Squared Error for regression tasks
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Example of using a DataLoader (ensure it's initialized correctly elsewhere)
# train_loader = DataLoader(...)

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    # Loop over batches in the DataLoader
    for batch_idx, (X_batch, y_batch) in tqdm(enumerate(dataloader), bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} < {remaining}, {rate_fmt}]"):
        optimizer.zero_grad()  # Zero out previous gradients

        # Move batch data to device (GPU or CPU)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        outputs = model(X_batch)

        # Compute loss
        loss = criterion(outputs, y_batch)  # Compare predicted outputs with actual targets

        # Backward pass and optimization
        loss.backward()  # Compute gradients
        optimizer.step()  # Update the model parameters

        # Accumulate the running loss
        running_loss += loss.item()

    # Print the loss for this epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")