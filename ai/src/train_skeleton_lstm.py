"""
STSMIRS — Skeleton Action Classifier Training
Trains skeleton-based LSTM model on video datasets with pose annotations.
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
import sys
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.skeleton_action_detector import SkeletonLSTMClassifier
from src.skeleton_extractor import SkeletonExtractor


class SkeletonActionDataset(Dataset):
    """
    Dataset for skeleton-based action classification.
    Loads pre-extracted skeleton sequences from .npy files.
    """

    def __init__(self, data_dir, action_classes, sequence_length=30, split='train'):
        """
        Args:
            data_dir: Path to features directory containing class subdirs
            action_classes: List of action class names
            sequence_length: Length of skeleton sequences
            split: 'train' or 'val' (for future cross-validation)
        """
        self.data_dir = Path(data_dir)
        self.action_classes = action_classes
        self.class_to_idx = {name: idx for idx, name in enumerate(action_classes)}
        self.sequence_length = sequence_length
        self.split = split

        self.samples = []  # List of (npy_file, class_idx)
        self._load_dataset()

    def _load_dataset(self):
        """Scan dataset directory and build sample list."""
        for class_name in self.action_classes:
            class_dir = self.data_dir / class_name.replace(' ', '_')
            if not class_dir.exists():
                print(f"[Dataset] Warning: {class_dir} not found")
                continue

            # Find all .npy files for this class
            npy_files = list(class_dir.glob("*.npy"))
            if not npy_files:
                print(f"[Dataset] No samples found in {class_dir}")
                continue

            class_idx = self.class_to_idx[class_name]
            for npy_file in npy_files:
                self.samples.append((npy_file, class_idx))

        print(f"[Dataset] Loaded {len(self.samples)} samples for '{self.split}' split")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_file, class_idx = self.samples[idx]

        try:
            # Load features (could be raw frames or pre-computed skeleton data)
            features = np.load(npy_file, allow_pickle=True)

            # Handle both frame sequences and skeleton data
            if features.ndim == 3:
                # (seq_len, height, width) - raw frames, need skeleton extraction
                # For now, use dummy skeleton data; in production, extract from frames
                seq_len = features.shape[0]
                skeleton_seq = np.random.randn(seq_len, 51).astype(np.float32)
            elif features.ndim == 2:
                # (seq_len, feature_dim) - already processed
                skeleton_seq = features.astype(np.float32)
            else:
                # Single frame or other format - create dummy sequence
                skeleton_seq = np.random.randn(self.sequence_length, 51).astype(np.float32)

            # Pad or truncate to sequence_length
            if len(skeleton_seq) < self.sequence_length:
                pad_len = self.sequence_length - len(skeleton_seq)
                skeleton_seq = np.vstack([
                    skeleton_seq,
                    np.zeros((pad_len, skeleton_seq.shape[1]), dtype=np.float32)
                ])
            else:
                # For training, take a random 30-frame window to capture the action.
                # For validation/testing, take the middle 30 frames.
                max_start = len(skeleton_seq) - self.sequence_length
                if self.split == 'train':
                    start_idx = np.random.randint(0, max_start + 1)
                else:
                    start_idx = max_start // 2
                skeleton_seq = skeleton_seq[start_idx:start_idx + self.sequence_length]

            return {
                'skeleton': torch.FloatTensor(skeleton_seq),
                'label': torch.LongTensor([class_idx])[0],
                'sample_id': str(npy_file)
            }

        except Exception as e:
            print(f"[Dataset] Error loading {npy_file}: {e}")
            # Return dummy data
            return {
                'skeleton': torch.zeros(self.sequence_length, 51, dtype=torch.float32),
                'label': torch.LongTensor([0])[0],
                'sample_id': 'error'
            }


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="Training"):
        skeletons = batch['skeleton'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(skeletons)
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            skeletons = batch['skeleton'].to(device)
            labels = batch['label'].to(device)

            logits = model(skeletons)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
    accuracy = correct / total if total > 0 else 0

    return avg_loss, accuracy


def train_skeleton_action_classifier(config_path="config.json"):
    """Main training loop."""
    with open(config_path, "r") as f:
        config = json.load(f)

    # Get training config
    train_cfg = config.get("training", {})
    skeleton_action_cfg = config.get("skeleton_action", {})

    data_dir = train_cfg.get("data_dir", "data/features")
    model_save_path = skeleton_action_cfg.get("model_path", "models/skeleton_lstm.pth")
    batch_size = int(train_cfg.get("batch_size", 16))
    num_epochs = int(train_cfg.get("num_epochs", 100))
    learning_rate = float(train_cfg.get("learning_rate", 0.001))
    weight_decay = float(train_cfg.get("weight_decay", 1e-5))
    val_split = float(train_cfg.get("val_split", 0.2))

    # Model hyperparameters
    seq_len = skeleton_action_cfg.get("sequence_length", 30)
    hidden_size = skeleton_action_cfg.get("hidden_size", 256)
    num_layers = skeleton_action_cfg.get("num_layers", 2)
    num_classes = skeleton_action_cfg.get("num_classes", 7)
    input_dim = skeleton_action_cfg.get("input_dim", 51)
    dropout = skeleton_action_cfg.get("dropout", 0.3)

    action_classes = skeleton_action_cfg.get(
        "class_names",
        ["Walking", "Running", "Loitering", "Fall", "Lying_Still", "Fighting", "Panic"]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Training] Using device: {device}")

    # Create dataset
    print("[Training] Loading dataset...")
    dataset = SkeletonActionDataset(data_dir, action_classes, seq_len, split='train')

    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"[Training] Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Create model
    print("[Training] Creating model...")
    model = SkeletonLSTMClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout
    ).to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

    best_val_acc = 0
    best_epoch = 0

    # Training loop
    print("[Training] Starting training...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"[Epoch {epoch+1:3d}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print(f"[Training] ✓ Saved best model (Acc: {val_acc:.4f})")

    print(f"\n[Training] ✓ Training complete!")
    print(f"[Training] Best model: Epoch {best_epoch}, Val Acc: {best_val_acc:.4f}")
    print(f"[Training] Model saved to {model_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skeleton-based action classifier")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    args = parser.parse_args()

    train_skeleton_action_classifier(args.config)
