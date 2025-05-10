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
def check_device():
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    elif torch.xpu.is_available():
        print("Intel XPU is available. Using XPU.")
        return torch.device("xpu")
    else:
        print("CUDA and XPU is not available. Using CPU.")
        return  torch.device("cpu")

device = check_device()
print(f"Using device: {device}")
train=TrainViTClassifier(device,batch_size=32,num_workers=4)
train.train_xpu(max_epochs=100)