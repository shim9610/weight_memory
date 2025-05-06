"""
Author: [SSY]
Created: [2025-05-06]
Purpose: Training and Validation of ViT-MAE based Classifier Model (Using CIFAR-10 Dataset)
Repository: https://github.com/shim9610/weight_memory
License: Apache License 2.0 (Based on Hugging Face Transformers)
"""
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, ViTMAEModel
from utility_my import ViTClassifier
from torch import nn
from torch.optim import AdamW
from tqdm import tqdm
import torch
# CUDA is available
# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = CIFAR10(root='./data', train=False, transform=transform, download=True)
# Create DataLoader
train_loader = DataLoader(
    train_dataset, batch_size=512, shuffle=True,
    num_workers=8, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=512, shuffle=False,
    num_workers=8, pin_memory=True
)

# Load the ViT-MAE model and processor
processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base')
encoder = ViTMAEModel.from_pretrained('facebook/vit-mae-base')
#freeze encoder parameters
for param in encoder.parameters():
    param.requires_grad = False
    
# define the ViTClassifier
model = ViTClassifier(encoder, num_classes=10).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-4)
model.train()

# Early stopping parameters
best_loss = float('inf')  # minimum loss for early stopping
eraly_stop_counter=0
limit=10
# Training loop
for epoch in range(100):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader):
        labels = labels.to(device)
        # convert images to PIL format
        images_list = [transforms.ToPILImage()(img) for img in images]
        # image processor
        inputs = processor(images=images_list, return_tensors="pt").to(device)
        pixel_values = inputs["pixel_values"]
        optimizer.zero_grad()
        logits = model(pixel_values)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    # Validation loop
    model.eval()  # Evaluation mode
    val_loss = 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            labels = labels.to(device)
            # convert images to PIL format
            images_list = [transforms.ToPILImage()(img) for img in images]
            # image processor
            inputs = processor(images=images_list, return_tensors="pt").to(device)
            pixel_values = inputs["pixel_values"]
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            val_loss += loss.item()
    val_loss_avg = val_loss / len(test_loader)
    print(f'Epoch {epoch+1},Train Loss: {avg_loss:.8f},Validation Loss: {val_loss_avg:.8f}')
    eraly_stop_counter+=1
    if val_loss_avg < best_loss:
        best_loss = val_loss_avg
        torch.save(model.state_dict(), 'best_vit_classifier_cifar10.pt')
        print(f"🚩 Best model saved (Epoch {epoch+1},Train Loss: {avg_loss:.8f},Validation Loss: {val_loss_avg:.8f})")
        eraly_stop_counter=0
    if eraly_stop_counter>limit:
        print(f"Early stopping at epoch {epoch+1} with validation loss: {val_loss_avg:.8f}")
        break
# final model saving
torch.save(model.state_dict(), 'vit_classifier_cifar10.pt')