import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Custom dataset to load images and corresponding white percentage values
class GrapeLeafDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)  # Load CSV file with image paths and white percentages
        self.image_folder = image_folder  # Image directory
        self.transform = transform  # Image transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])  # Get image path
        image = Image.open(img_name).convert("RGB")  # Load image
        label = float(self.data.iloc[idx, 1]) / 100.0  # Normalize percentage (0 to 1)
        
        if self.transform:
            image = self.transform(image)  # Apply transformations
        
        return image, torch.tensor(label, dtype=torch.float32)  # Return image and label

# Define a neural network model for predicting white percentage
class LeafWhiteSurfaceNN(nn.Module):
    def __init__(self):
        super(LeafWhiteSurfaceNN, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            nn.Linear(3*128*128, 512),  # Adjust for image size (128x128 RGB)
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Single output neuron
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )
    
    def forward(self, x):
        x = self.flatten(x)  # Flatten image
        return self.model(x).squeeze(1)  # Ensure output shape is correct

# Function to evaluate the model's performance
def Evaluate(model, dataset, device):
    model.eval()
    total_error = 0.0
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    with torch.no_grad():
        for image, true_label in test_loader:
            image, true_label = image.to(device), true_label.to(device)
            predicted_value = model(image).item() * 100  # Convert back to percentage
            true_label_value = true_label.item() * 100
            error = abs(predicted_value - true_label_value)
            total_error += error
            print(f"Predicted: {predicted_value:.2f}%, Actual: {true_label_value:.2f}%, Error: {error:.2f}%")
    
    print(f"\nAverage Absolute Error: {total_error / len(dataset):.2f}%")

# Main function for training and evaluating the model
def main():
    # Define image transformations (resize and convert to tensor)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Load datasets
    train_dataset = GrapeLeafDataset("train.csv", "train", transform)
    valid_dataset = GrapeLeafDataset("valid.csv", "valid", transform)
    test_dataset = GrapeLeafDataset("test.csv", "test", transform)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = LeafWhiteSurfaceNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    num_epochs = 10
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)
        
        val_loss = running_loss / len(valid_loader.dataset)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Evaluate on test data
    print("\nEvaluating on test data...")
    Evaluate(model, test_dataset, device)

if __name__ == "__main__":
    main()
