"""
ViTClassifier: Utility Module for Vision Transformer-based Classification.
Author: [SSY]
Created: [2025-05-06]
Description:
    Defines a custom classifier model built upon a pre-trained Vision Transformer (ViT) encoder.
    Extracts the [CLS] token from the transformer encoder output and applies additional
    fully-connected layers to perform classification tasks.
Usage:
    from utility_my import ViTClassifier
"""
import torch
from torch.optim import AdamW
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
from transformers import AutoImageProcessor, ViTMAEModel
from torch.utils.data import DataLoader

from tqdm import tqdm
class TrainViTClassifier():
    def __init__(self, device, num_classes=10,size=224,dataset='cifar10',batch_size=512,num_workers=8):
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
        ])
        if dataset=='cifar10':
            # Load CIFAR-10 dataset
            self.train_dataset = CIFAR10(root='./data', train=True, transform=self.transform, download=True)
            self.test_dataset = CIFAR10(root='./data', train=False, transform=self.transform, download=True)
        else: 
            self.train_dataset = None
            self.test_dataset = None
        
        # Create DataLoader
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        # Load the ViT-MAE model and processor
        self.processor = AutoImageProcessor.from_pretrained('facebook/vit-mae-base')
        self.encoder = ViTMAEModel.from_pretrained('facebook/vit-mae-base')
        #freeze encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        # define the ViTClassifier
        self.model = ViTClassifier(self.encoder, num_classes=num_classes).to(device)
        
    def train(self,name='result',limit=10,max_epochs=100):
        device = self.device
        model = self.model
        processor = self.processor
        train_loader = self.train_loader
        test_loader = self.test_loader
        # Define loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=1e-4)
        model.train()
        # Early stopping parameters
        best_loss = float('inf')  # minimum loss for early stopping
        early_stop_counter=0
        # Training loop
        for epoch in range(max_epochs):
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
            early_stop_counter+=1
            if val_loss_avg < best_loss:
                best_loss = val_loss_avg
                torch.save(model.state_dict(), 'best_'+name+'.pt')
                print(f"🚩 Best model saved (Epoch {epoch+1},Train Loss: {avg_loss:.8f},Validation Loss: {val_loss_avg:.8f})")
                early_stop_counter=0
            if early_stop_counter>limit:
                print(f"Early stopping at epoch {epoch+1} with validation loss: {val_loss_avg:.8f}")
                break
        # final model saving
        torch.save(model.state_dict(), name+'.pt')
        
        
class ViTClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.config.hidden_size, 256), # Intermediate linear layer
            nn.GELU(),                                  # Non-linear activation
            nn.Dropout(0.2),                            # Dropout for regularization    
            nn.Linear(256, num_classes)                 # Final classification layer    
        )

    def forward(self, pixel_values):
        # Forward pass through encoder to obtain feature embeddings
        encoder_outputs = self.encoder(pixel_values)  
        # Use [CLS] token embedding (first token) for classification          
        pooled_output = encoder_outputs.last_hidden_state[:, 0]
        # Compute logits through classification head
        logits = self.classifier(pooled_output)
        
        return logits