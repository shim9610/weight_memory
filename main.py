"""
Author: [SSY]
Created: [2025-05-06]
Purpose: Training and Validation of ViT-MAE based Classifier Model (Using CIFAR-10 Dataset)
Repository: https://github.com/shim9610/weight_memory
License: Apache License 2.0 (Based on Hugging Face Transformers)
"""

from torchvision import transforms
from utility_my import TrainViTClassifier
import torch
# CUDA is available
# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
train=TrainViTClassifier(device)
train.train(max_epochs=5)