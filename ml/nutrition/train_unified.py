"""
Train script for fine-tuning the EfficientNet-based food classifier
on the `nutrition/unified_nutrition_dataset.csv` dataset.

Features:
- Filter dataset to a configurable list of target classes (default: common Indian foods)
- Optional image download and local caching
- Data augmentation and balanced sampling
- Transfer learning (freeze backbone then unfreeze) with mixed precision
- Validation, classification report, confusion matrix, and checkpointing

Usage examples:
  # Dry run (no download, quick small experiment)
  python train_unified.py --epochs 5 --batch-size 32 --subset-size 2000

  # Full train (download remote images to cache and train)
  python train_unified.py --download-images --data-csv "../../nutrition/unified_nutrition_dataset.csv" \
    --output-dir ./models/unified_effnet --epochs 25 --batch-size 64

Notes:
- Achieving 95% accuracy will require sufficient, well-labeled images per class
- If your `image_path` values are URLs, use --download-images to cache them locally
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models

try:
    from efficientnet_pytorch import EfficientNet
except Exception:
    EfficientNet = None

import requests
from io import BytesIO
from sklearn.metrics import classification_report, confusion_matrix


DEFAULT_INDIAN_FOODS = [
    "poha", "upma", "idli", "dosa", "samosa", "biryani", "roti", "chapati",
    "paneer", "paratha", "kheer", "gulab_jamun", "dal", "chole", "rajma",
    "vada", "puri", "pakora", "pongal"
]


class UnifiedFoodDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: Path, labels: List[str], transforms=None):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.labels = labels
        self.label_to_idx = {l: i for i, l in enumerate(labels)}
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['local_image_path'] if 'local_image_path' in row and pd.notna(row['local_image_path']) else row['image_path']
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception:
            # fallback: try opening as URL
            try:
                resp = requests.get(row['image_path'], timeout=5)
                img = Image.open(BytesIO(resp.content)).convert('RGB')
            except Exception:
                # return a blank image if everything fails
                img = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transforms:
            img = self.transforms(img)

        label_name = row['label']
        label_idx = self.label_to_idx.get(label_name, 0)

        return img, label_idx


def download_image(url: str, dest: Path, timeout: int = 10) -> Optional[Path]:
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            return dest
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert('RGB')
        img.save(dest)
        return dest
    except Exception:
        return None


def build_dataframe_from_csv(csv_path: Path, target_labels: List[str], subset_size: Optional[int] = None) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Normalize food_items to lower-case single-word labels where possible
    def normalize_label(x: str) -> str:
        if pd.isna(x):
            return ''
        s = str(x).strip().lower()
        # keep only first token (most CSV entries have a single dish name)
        s = s.split(',')[0].split(' with ')[0]
        s = s.replace(' ', '_')
        return s

    df['label'] = df['food_items'].apply(normalize_label)

    # Filter to target labels
    df = df[df['label'].isin(target_labels)].copy()

    if subset_size and subset_size > 0 and subset_size < len(df):
        df = df.sample(subset_size, random_state=42).reset_index(drop=True)

    return df.reset_index(drop=True)


def create_transforms(train: bool = True):
    if train:
        return T.Compose([
            T.Resize((300, 300)),
            T.RandomResizedCrop(224, scale=(0.6, 1.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize((300, 300)),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def make_model(num_classes: int, device: str):
    # Prefer efficientnet_pytorch if available, otherwise fall back to torchvision models
    backbone = None
    in_features = None

    if EfficientNet is not None:
        try:
            backbone = EfficientNet.from_pretrained('efficientnet-b3')
            in_features = backbone._fc.in_features
            backbone._fc = nn.Identity()
        except Exception:
            backbone = None

    if backbone is None:
        # Try ResNet50 from torchvision as a robust fallback
        try:
            backbone = models.resnet50(pretrained=True)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        except Exception:
            # Final fallback: MobileNetV3 small
            backbone = models.mobilenet_v3_small(pretrained=True)
            # MobileNetV3 classifier is usually a Sequential ending with Linear
            if hasattr(backbone, 'classifier') and isinstance(backbone.classifier, nn.Sequential):
                last = backbone.classifier[-1]
                in_features = last.in_features if hasattr(last, 'in_features') else 576
                backbone.classifier = nn.Identity()
            else:
                # As a safe default
                in_features = 576

    # classification head
    classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )

    # Attach heads to model object for convenience
    class FullModel(nn.Module):
        def __init__(self, backbone, head):
            super().__init__()
            self.backbone = backbone
            self.head = head

        def forward(self, x):
            features = self.backbone(x)
            return self.head(features)

    full = FullModel(backbone, classifier)
    return full.to(device)


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device:', device)

    csv_path = Path(args.data_csv)
    image_root = Path(args.image_cache)
    image_root.mkdir(parents=True, exist_ok=True)

    target_labels = args.target_labels or DEFAULT_INDIAN_FOODS
    target_labels = [t.replace(' ', '_').lower() for t in target_labels]

    print(f'Building dataframe for target labels ({len(target_labels)}):', target_labels)
    df = build_dataframe_from_csv(csv_path, target_labels, subset_size=args.subset_size)

    if len(df) == 0:
        raise RuntimeError('No matching images found in CSV for target labels. Consider widening the labels or checking CSV paths.')

    # Optionally download images
    if args.download_images:
        print('Downloading images to cache (this may take a while)')
        local_paths = []
        for i, row in df.iterrows():
            url = row['image_path']
            fname = f"img_{i}_{row['label']}.jpg"
            dest = image_root / fname
            p = download_image(url, dest)
            local_paths.append(str(p) if p is not None else '')
            if i % 500 == 0 and i > 0:
                print(f'downloaded {i}/{len(df)}')
        df['local_image_path'] = local_paths

    # Split into train/val
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_frac = args.val_fraction
    val_n = int(len(df) * val_frac)
    val_df = df.iloc[:val_n].reset_index(drop=True)
    train_df = df.iloc[val_n:].reset_index(drop=True)

    print(f'Dataset sizes: train={len(train_df)}, val={len(val_df)}')

    train_transforms = create_transforms(True)
    val_transforms = create_transforms(False)

    train_dataset = UnifiedFoodDataset(train_df, image_root, target_labels, transforms=train_transforms)
    val_dataset = UnifiedFoodDataset(val_df, image_root, target_labels, transforms=val_transforms)

    # Balanced sampling
    label_counts = train_df['label'].value_counts().reindex(target_labels).fillna(0).values
    class_weights = 1.0 / (label_counts + 1e-6)
    sample_weights = train_df['label'].map(lambda l: class_weights[target_labels.index(l)]).values
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = make_model(num_classes=len(target_labels), device=device)

    # Freeze backbone initially
    for name, p in model.named_parameters():
        if 'head' not in name:
            p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    best_acc = 0.0
    patience = args.patience
    rounds_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, preds = outputs.max(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(labels.cpu().numpy().tolist())

        val_loss = val_loss / total if total else 0.0
        val_acc = correct / total if total else 0.0

        print(f'Epoch {epoch:02d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            rounds_no_improve = 0
            os.makedirs(args.output_dir, exist_ok=True)
            save_path = Path(args.output_dir) / f'best_effnet_epoch{epoch}_acc{val_acc:.4f}.pt'
            torch.save({'model_state_dict': model.state_dict(), 'labels': target_labels}, save_path)
            print('Saved best model to', save_path)
        else:
            rounds_no_improve += 1

        # Unfreeze after some epochs
        if epoch == args.unfreeze_epoch:
            print('Unfreezing backbone for fine-tuning')
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-5)

        if rounds_no_improve >= patience:
            print('Early stopping triggered')
            break

    # Final classification report
    try:
        target_names = target_labels
        report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0)
        print('\nClassification Report:\n', report)
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-csv', type=str, default='nutrition/unified_nutrition_dataset.csv')
    p.add_argument('--image-cache', type=str, default='datasets/images_cache')
    p.add_argument('--download-images', action='store_true')
    p.add_argument('--target-labels', nargs='*', default=None)
    p.add_argument('--subset-size', type=int, default=0, help='If >0, randomly sample this many rows from matching dataset')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--output-dir', type=str, default='ml_models/nutrition/models')
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--unfreeze-epoch', type=int, default=5)
    p.add_argument('--use-gpu', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # normalize target labels
    if args.target_labels:
        args.target_labels = [t.replace(' ', '_').lower() for t in args.target_labels]
    train(args)
