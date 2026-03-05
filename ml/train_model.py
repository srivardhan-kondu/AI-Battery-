"""
Model Training Script — Fine-tune MobileNetV2 on SLIBR Dataset (FR9)
Dataset: https://www.kaggle.com/datasets/thgere/spent-lithium-ion-battery-recyclingslibr-dataset/data

Usage:
    python ml/train_model.py --data_dir data/raw --epochs 25 --batch_size 32

Place the Kaggle dataset in data/raw/ with structure:
    data/raw/
    ├── battery/         ← images WITH batteries
    └── no_battery/      ← images WITHOUT batteries (non-battery objects)
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from datetime import datetime


def get_transforms(is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])


def build_model(num_classes=2, freeze_backbone=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
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
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train battery detector on SLIBR dataset")
    parser.add_argument("--data_dir", type=str, default="data/raw",
                        help="Path to dataset root (must have class sub-folders)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--no_freeze", action="store_true",
                        help="Unfreeze backbone from the start")
    args = parser.parse_args()

    # ── Setup ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    full_dataset = datasets.ImageFolder(args.data_dir, transform=get_transforms(True))
    print(f"Classes found: {full_dataset.classes}")
    print(f"Total samples: {len(full_dataset)}")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    # Override transform for validation
    val_ds.dataset.transform = get_transforms(False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────────
    num_classes = len(full_dataset.classes)
    model = build_model(num_classes=num_classes, freeze_backbone=not args.no_freeze)
    model = model.to(device)

    # Class weights for imbalanced data
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone after half the epochs
        if epoch == args.epochs // 2 and not args.no_freeze:
            print(f"\nEpoch {epoch}: Unfreezing backbone layers...")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=args.lr * 0.1,
                                    weight_decay=1e-4)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, "battery_detector.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved to {save_path} (Val Acc: {val_acc:.4f})")

    # ── Save training history ────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_path = os.path.join(args.output_dir, f"training_history_{ts}.json")
    history["class_names"] = full_dataset.classes
    history["best_val_acc"] = best_val_acc
    history["total_epochs"] = args.epochs
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n Training complete!")
    print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"   Model saved to: models/battery_detector.pth")
    print(f"   History saved to: {history_path}")


if __name__ == "__main__":
    main()
