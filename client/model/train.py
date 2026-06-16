#!/usr/bin/env python3
"""
Train a small VGG-like CNN for fruit/vegetable classification.

Architecture compatible with tbn-runtime operator set:
  Conv → ReLU → MaxPool → ... → GlobalAvgPool → Flatten → Linear

Usage:
    # Train on a dataset
    python train.py --data_dir ./fruits_dataset --epochs 30

    # Or with synthetic data for pipeline testing
    python train.py --synthetic

Output:
    model/fruits_model.onnx  — ONNX model for tbn-runtime
    model/labels.txt          — Class labels (generated from dataset)
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ── Model ────────────────────────────────────────────────────────────────────


class FruitCNN(nn.Module):
    """
    VGG-like CNN with only tbn-runtime-compatible operations:
    Conv2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Flatten, Linear.

    No BatchNorm, no Dropout (in eval mode), no fancy activations.
    """

    def __init__(self, num_classes: int = 15):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3x224x224 → 32x112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 32x112x112 → 64x56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 64x56x56 → 128x28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 128x28x28 → 256x14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 5: 256x14x14 → 512x14x14
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # → ONNX GlobalAveragePool
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, num_classes, bias=True),  # → ONNX Gemm
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out',
                                         nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


# ── Training ─────────────────────────────────────────────────────────────────


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += images.size(0)

    return total_loss / total, 100.0 * correct / total


def train_model(data_dir, epochs, batch_size, lr, device, save_dir):
    """Train the model on an ImageFolder dataset."""

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    # Save class labels
    labels_path = os.path.join(save_dir, 'labels.txt')
    with open(labels_path, 'w') as f:
        for cls_name in full_dataset.classes:
            f.write(f"{cls_name}\n")
    print(f"Labels saved to {labels_path}: {full_dataset.classes}")

    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images in {num_classes} classes")

    # Split
    val_size = max(int(0.2 * len(full_dataset)), num_classes)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Validation set uses val_transform
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = FruitCNN(num_classes=num_classes).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_acc = 0
    best_path = os.path.join(save_dir, 'best_model.pth')

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  → Best model saved ({best_acc:.1f}%)")

    # Load best for export
    model.load_state_dict(torch.load(best_path, weights_only=True))
    model.eval()

    # Export to ONNX
    onnx_path = os.path.join(save_dir, 'fruits_model.onnx')
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=13,
    )

    print(f"\nONNX model exported to: {onnx_path}")
    print(f"Best validation accuracy: {best_acc:.1f}%")

    return onnx_path, best_acc


# ── Synthetic data for pipeline testing ──────────────────────────────────────


def train_synthetic(save_dir, device):
    """Create a model with random weights for pipeline testing."""
    num_classes = 8
    model = FruitCNN(num_classes=num_classes).to(device)
    model.eval()

    onnx_path = os.path.join(save_dir, 'fruits_model.onnx')
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
        opset_version=13,
    )

    print(f"Synthetic (untrained) model exported to: {onnx_path}")
    print("This model has RANDOM weights — predictions will be meaningless.")
    print("Use --data_dir to train on real data.")
    return onnx_path, 0.0


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description='Train fruit classifier for Smart Store Scanner'
    )
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to ImageFolder dataset')
    parser.add_argument('--synthetic', action='store_true',
                        help='Generate untrained model for testing')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Output directory (default: model/)')

    args = parser.parse_args()

    # Save dir
    if args.save_dir:
        save_dir = args.save_dir
    else:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)

    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    if args.synthetic:
        train_synthetic(save_dir, device)
    elif args.data_dir:
        train_model(args.data_dir, args.epochs, args.batch_size,
                    args.lr, device, save_dir)
    else:
        print("ERROR: Specify --data_dir or --synthetic")
        print("\nDataset should be structured as:")
        print("  data_dir/")
        print("    apple/")
        print("      img001.jpg")
        print("      img002.jpg")
        print("    banana/")
        print("      ...")
        print("\nRecommended: Fruits-360 dataset from Kaggle")
        print("  https://www.kaggle.com/datasets/moltean/fruits")
        sys.exit(1)


if __name__ == '__main__':
    main()
