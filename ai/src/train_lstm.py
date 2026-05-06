"""
STSMIRS — LSTM Training Script
Trains the PyTorch LSTM model on extracted feature sequences.
"""

import os
import glob
import numpy as np
import argparse
import sys
import random
import json
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from src.action_detector import LSTMClassifier
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. Cannot train.")

class FeatureDataset(Dataset):
    def __init__(self, data_dir, classes):
        self.sequences = []
        self.labels = []
        self.sources = []
        self.classes = {c: i for i, c in enumerate(classes)}
        self.class_counts = {c: 0 for c in classes}
        
        print(f"Scanning for extracted features in {data_dir}...")
        
        if not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
            print(f"Created {data_dir}. No training data found yet.")
            return

        files = glob.glob(os.path.join(data_dir, "*.npy"))
        
        for file in files:
            # Filename convention: ClassName_source.npy. Match longest first
            # so Lying_Still is not mistaken for Lying.
            basename = os.path.basename(file)
            class_name = next(
                (c for c in sorted(self.classes, key=len, reverse=True) if basename.startswith(f"{c}_")),
                None
            )
            
            if class_name:
                try:
                    seqs = np.load(file) # Shape expected: (N_samples, seq_len, features)
                    if len(seqs.shape) == 3:
                        source_name = f"{class_name}:{basename[len(class_name) + 1 : -4]}"
                        for seq in seqs:
                            self.sequences.append(seq)
                            self.labels.append(self.classes[class_name])
                            self.sources.append(source_name)
                            self.class_counts[class_name] += 1
                except Exception as e:
                    print(f"Error loading {file}: {e}")
            else:
                print(f"Skipping {basename}: class prefix not recognized")
                    
        print(f"Loaded {len(self.sequences)} total sequences across {len(self.classes)} classes.")
        print("Class counts:")
        for class_name, count in self.class_counts.items():
            print(f"  {class_name}: {count}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def split_indices_by_source(dataset, val_ratio=0.2, seed=42):
    rng = random.Random(seed)
    if len(dataset) <= 1:
        return list(range(len(dataset))), list(range(len(dataset)))

    class_source_to_indices = {}
    for idx, source_name in enumerate(dataset.sources):
        label = dataset.labels[idx]
        class_source_to_indices.setdefault(label, {}).setdefault(source_name, []).append(idx)

    train_indices = []
    val_indices = []

    for label, source_map in class_source_to_indices.items():
        groups = list(source_map.items())
        rng.shuffle(groups)
        class_total = sum(len(indices) for _, indices in groups)
        target_val_samples = max(1, int(class_total * val_ratio)) if len(groups) > 1 else 0
        class_val = []
        class_train = []
        val_samples = 0

        for _, indices in groups:
            if val_samples < target_val_samples:
                class_val.extend(indices)
                val_samples += len(indices)
            else:
                class_train.extend(indices)

        if not class_train and class_val:
            class_train, class_val = class_val, []
        if not class_val and len(class_train) > 1:
            class_val = [class_train.pop()]

        train_indices.extend(class_train)
        val_indices.extend(class_val)

    if not val_indices and len(train_indices) > 1:
        val_indices = [train_indices.pop()]

    return train_indices, val_indices

def build_random_split(dataset, val_ratio=0.2, seed=42):
    indices = list(range(len(dataset)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_size = max(1, int((1.0 - val_ratio) * len(indices)))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    if not val_indices:
        val_indices = train_indices[-1:]
        train_indices = train_indices[:-1] or train_indices
    return train_indices, val_indices


def train(epochs=50, batch_size=32, learning_rate=0.001, resume=False, seed=42, split_mode="grouped"):
    if not TORCH_AVAILABLE:
        return
        
    # Hyperparams (should match config.json)
    seq_len = 30
    feature_dim = 8
    hidden_dim = 256
    num_layers = 2
    with open("config.json", "r") as f:
        config = json.load(f)
    classes = config.get(
        "lstm",
        {},
    ).get(
        "class_names",
        ["Walking", "Running", "Loitering", "Fall", "Lying_Still", "Fighting", "Panic"],
    )
    
    data_dir = "data/features"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    dataset = FeatureDataset(data_dir, classes)
    
    if len(dataset) == 0:
        print("\nNo data found! Please run tools/extract_training_data.py first.")
        # Create a dummy model file anyway so inference doesn't crash
        model = LSTMClassifier(feature_dim, hidden_dim, num_layers, len(classes))
        torch.save(model.state_dict(), os.path.join(model_dir, "lstm_classifier.pth"))
        print("Created uninitialized dummy weight file at models/lstm_classifier.pth")
        return
        
    if split_mode == "random":
        train_indices, val_indices = build_random_split(dataset, val_ratio=0.2, seed=seed)
        print(
            f"Legacy random split: train={len(train_indices)} sequences, "
            f"val={len(val_indices)} sequences"
        )
    else:
        train_indices, val_indices = split_indices_by_source(dataset, val_ratio=0.2, seed=seed)
        print(
            f"Grouped split by source video: train={len(train_indices)} sequences, "
            f"val={len(val_indices)} sequences, unique_sources={len(set(dataset.sources))}"
        )
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices if val_indices else train_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    
    model = LSTMClassifier(feature_dim, hidden_dim, num_layers, len(classes)).to(device)
    model_path = os.path.join(model_dir, "lstm_classifier.pth")
    if resume and os.path.exists(model_path):
        saved_state = torch.load(model_path, map_location=device)
        try:
            model.load_state_dict(saved_state)
            print(f"Resumed training from {model_path}")
        except RuntimeError:
            current_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in saved_state.items()
                if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
            }
            current_state.update(compatible_state)
            model.load_state_dict(current_state)
            print(
                f"Resumed partial training from {model_path} "
                f"({len(compatible_state)}/{len(current_state)} tensors matched)"
            )
    elif resume:
        print(f"Resume requested, but no existing model found at {model_path}. Starting fresh.")

    train_label_counts = Counter(dataset.labels[idx] for idx in train_indices)
    class_weights = []
    for class_idx, class_name in enumerate(classes):
        count = train_label_counts.get(class_idx, 0)
        weight = len(train_indices) / max(count, 1)
        class_weights.append(weight)
        print(f"Train samples for {class_name}: {count} | weight={weight:.2f}")
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    best_loss = float('inf')
    best_acc = 0.0
    meta_path = os.path.join(model_dir, "lstm_classifier_meta.json")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total if total > 0 else 0
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | LR: {current_lr:.6f}"
        )
        
        if val_acc > best_acc or (val_acc >= best_acc and val_loss < best_loss):
            best_acc = val_acc
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
            with open(meta_path, "w") as f:
                json.dump(
                    {
                        "class_names": classes,
                        "class_counts": dataset.class_counts,
                        "train_indices": len(train_indices),
                        "val_indices": len(val_indices),
                        "split_mode": split_mode,
                        "best_val_acc": best_acc,
                        "best_val_loss": best_loss,
                    },
                    f,
                    indent=2,
                )
            print(f"  -> Saved new best model")

    print(f"\nTraining Complete! Best Val Acc: {best_acc:.4f} | Best Val Loss: {best_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the STSMIRS LSTM classifier")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--resume", action="store_true", help="Continue from models/lstm_classifier.pth if present")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["grouped", "random"], default="grouped")
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        resume=args.resume,
        seed=args.seed,
        split_mode=args.split_mode,
    )
