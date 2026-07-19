import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import pickle
import os

from data import BurmeseDigitDataset
from model import BurmeseDigitCNN_V2

def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing V2 Training Engine on: {device}")

    data_path = "data.pkl" 
    if not os.path.exists(data_path):
        print(f"Error: Could not find '{data_path}' in the root directory.")
        return

    print("Loading dataset...")
    with open(data_path, "rb") as file:
        data = pickle.load(file)
    
    train_data = data['trainDataset'] 

    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.7, 1.1)),
        transforms.ToTensor()
    ])

    train_dataset = BurmeseDigitDataset(train_data, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = BurmeseDigitCNN_V2().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            predictions = model(images)
            loss = criterion(predictions, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")

    print("V2 Training Complete! Saving model weights...")
    
    torch.save(model.state_dict(), "bhdr_v2_weights.pth")
    print("Weights successfully saved as 'bhdr_v2_weights.pth'")

if __name__ == "__main__":
    train()