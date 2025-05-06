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
import torch.nn as nn

class ViTClassifier(nn.Module):
    def __init__(self, encoder, num_classes=10):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.config.hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        encoder_outputs = self.encoder(pixel_values)
        pooled_output = encoder_outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(pooled_output)
        return logits