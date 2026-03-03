#!/usr/bin/env python3
"""
Training pipeline for a simple feed-forward neural network that takes
11 offensive player positions (x, y for each player -> 22 features total)
in normalized field coordinates and learns to predict a label
(e.g. formation, play type, etc.).

Usage (example):

    python train_offense_positions.py ^
        --csv-path cache/TestingFootage/offense_positions.csv ^
        --label-col label ^
        --output-dir CNN/models/offense_positions

Expected CSV format (default behavior, produced by build_offense_positions_dataset.py):
    - A string column 'clip_name'
    - 22 numeric feature columns for normalized positions:
        nx1, ny1, ..., nx11, ny11
    - Optional original pixel center columns:
        ox1, oy1, ..., ox11, oy11
    - 1 label column (e.g. "label")

By default the script:
    - Automatically picks the 22 normalized position columns as features
      by selecting numeric columns whose names contain 'n' (nx*/ny*)
      and taking the first 22 of those
    - Encodes non-numeric labels into integer class IDs
    - Infers number of output classes from the data
    - Trains a small MLP classifier
    - Saves model weights and metadata into output-dir
"""

import argparse
import json
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
from itertools import combinations

class PositionsDataset(Dataset):
    """
    Dataset that reads a CSV with 22 feature columns (11 (x, y) pairs)
    and a label column.
    """

    def __init__(
        self,
        csv_path: str,
        label_col: str = "label",
        feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if label_col not in df.columns:
            raise ValueError(f"Label column '{label_col}' not found in CSV.")

        self.label_col = label_col

        # Drop rows with missing or empty labels (otherwise NaN -> int64 gives invalid class IDs)
        raw_labels_series = df[label_col]
        valid = raw_labels_series.notna() & (raw_labels_series.astype(str).str.strip() != "")
        if not valid.all():
            n_dropped = (~valid).sum()
            df = df.loc[valid].reset_index(drop=True)
            if len(df) == 0:
                raise ValueError(
                    f"Label column '{label_col}' has no valid (non-empty) labels. "
                    f"Dropped {n_dropped} row(s) with missing/empty labels."
                )
            print(f"[WARNING] Dropped {n_dropped} row(s) with missing/empty '{label_col}'.")

        # Keep clip names for later reporting (fall back to row index string)
        if "clip_name" in df.columns:
            self.clip_names = df["clip_name"].astype(str).tolist()
        else:
            self.clip_names = [str(i) for i in range(len(df))]

        # Determine feature columns
        if feature_cols is None:
            # Select numeric columns whose names indicate normalized positions (nx*/ny*)
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            norm_cols = [col for col in numeric_cols if col.startswith("nx") or col.startswith("ny")]
            if label_col in norm_cols:
                norm_cols.remove(label_col)

            if len(norm_cols) < 22:
                raise ValueError(
                    f"Expected at least 22 normalized feature columns (nx1, ny1, ..., nx11, ny11), "
                    f"found {len(norm_cols)}."
                )

            # Use the first 22 normalized columns by default
            self.feature_cols = norm_cols[:22]
        
        else:
            for col in feature_cols:
                if col not in df.columns:
                    raise ValueError(f"Feature column '{col}' not found in CSV.")
            self.feature_cols = feature_cols

        # Extract features
        features = df[self.feature_cols].to_numpy(dtype=np.float32)
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0)
        if normalize:
            # Normalize features and extract geometric features
            features = self.extract_geometric_features(features)
            self.features = features
        else:
            self.features = torch.from_numpy(features)
        

        # Handle labels (support numeric or string labels)
        raw_labels = df[label_col]
        if raw_labels.dtype == object:
            # String / categorical labels -> map to integer class IDs (skip NaN/empty already dropped)
            categories = sorted(raw_labels.unique())
            self.label_to_index = {label: idx for idx, label in enumerate(categories)}
            self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}
            labels = raw_labels.map(self.label_to_index).to_numpy(dtype=np.int64)
        else:
            # Assume already integer class IDs
            labels = raw_labels.to_numpy(dtype=np.int64)
            classes = sorted(int(c) for c in np.unique(labels))
            self.label_to_index = {int(c): int(c) for c in classes}
            self.index_to_label = {int(c): int(c) for c in classes}

        self.labels = torch.from_numpy(labels)

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    @property
    def num_classes(self) -> int:
        # Use mapping size so we don't depend on labels.max() (which breaks when labels had NaN -> int64 min)
        n = len(self.index_to_label)
        if n < 1:
            raise ValueError("No valid label classes in dataset.")
        return n

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]

    def extract_geometric_features(self, features: torch.Tensor) -> torch.Tensor:
        # NORMALIZE FEATURES
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)

        features = features.float()
        coords = features.reshape(-1, 11, 2)

        # 1. Center (translation invariance)
        centroid = coords.mean(axis=1, keepdims=True)
        coords = coords - centroid
        
        B = coords.shape[0]

        # -------------------------------------------------
        # 1. Base Flattened Coordinates (11 * 2 = 22)
        # -------------------------------------------------
        base_features = coords.reshape(B, -1)

        # -------------------------------------------------
        # 2. Width & Depth (Span Features)
        # -------------------------------------------------
        x_max = torch.amax(coords[:, :, 0], dim=1)
        x_min = torch.amin(coords[:, :, 0], dim=1)
        x_span = x_max - x_min

        y_max = torch.amax(coords[:, :, 1], dim=1)
        y_min = torch.amin(coords[:, :, 1], dim=1)
        y_span = y_max - y_min

        span_features = torch.stack([x_span, y_span], dim=1)

        # -------------------------------------------------
        # 3. Mean Pairwise Distance
        # -------------------------------------------------
        pairwise_dists = []

        for i, j in combinations(range(11), 2):
            diff = coords[:, i] - coords[:, j]
            dist = torch.norm(diff, dim=1)
            pairwise_dists.append(dist)

        pairwise_dists = torch.stack(pairwise_dists, dim=1)

        mean_pairwise_distance = pairwise_dists.mean(dim=1, keepdim=True)

        # -------------------------------------------------
        # 4. Distance to Centroid Statistics
        # -------------------------------------------------
        d_to_centroid = torch.norm(coords - centroid, dim=2)

        centroid_mean = d_to_centroid.mean(dim=1, keepdim=True)
        centroid_std = d_to_centroid.std(dim=1, keepdim=True)

        centroid_features = torch.cat([centroid_mean, centroid_std], dim=1)

        # -------------------------------------------------
        # 5. PCA Eigenvalues (Top 2)
        # -------------------------------------------------
        # Compute covariance matrix per batch
        centered = coords - centroid
        cov = torch.bmm(centered.transpose(1, 2), centered) / 11.0 + 1e-6 * torch.eye(2).to(coords.device)

        eigvals = []

        eigvals_all = torch.linalg.eigvalsh(cov)
        eigvals_sorted = torch.sort(eigvals_all, dim=1).values
        eigvals = eigvals_sorted[:, -2:]

        # -------------------------------------------------
        # Concatenate Everything
        # -------------------------------------------------
        features = torch.cat(
            [
                base_features,
                span_features,
                mean_pairwise_distance,
                centroid_features,
                eigvals,
            ],
            dim=1,
        )

        return features


class PositionsNet(nn.Module):
    """
    Simple MLP classifier for 22-dim input.
    """

    def __init__(
        self,
        input_dim: int = 22,
        hidden_dims: Tuple[int, ...] = (64, 64),
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_lookup: Optional[dict[int, str]] = None,
    log_examples: int = 3,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    if dataloader is None:
        return 0.0, 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def save_artifacts(
    model: nn.Module,
    output_dir: str,
    dataset: PositionsDataset,
    args: argparse.Namespace,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "formModel.pt")
    torch.save(model.state_dict(), model_path)

    metadata = {
        "feature_cols": dataset.feature_cols,
        "label_col": dataset.label_col,
        "label_to_index": dataset.label_to_index,
        "index_to_label": dataset.index_to_label,
        "num_features": dataset.num_features,
        "num_classes": dataset.num_classes,
        "normalize": True if dataset.mean is not None else False,
        "mean": dataset.mean.tolist() if dataset.mean is not None else None,
        "std": dataset.std.tolist() if dataset.std is not None else None,
        "train_args": vars(args),
    }

    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[INFO] Saved model to: {model_path}")
    print(f"[INFO] Saved metadata to: {metadata_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a simple NN on 11 offensive player positions (22 features)."
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV file containing features and labels.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of the label column in the CSV.",
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        nargs="*",
        default=None,
        help="Optional explicit list of feature column names. "
        "If omitted, the first 22 numeric columns (excluding label) are used.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="*",
        default=[64, 32],
        help="Hidden layer sizes for the MLP.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability between hidden layers.",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data to use for validation (0-1).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/offense_positions",
        help="Directory to save model and metadata.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.no_cuda) else "cpu"
    )
    print(f"[INFO] Using device: {device}")

    dataset = PositionsDataset(
        csv_path=args.csv_path,
        label_col=args.label_col,
        feature_cols=args.feature_cols,
        normalize=True,
    )
    print(f"[INFO] Loaded dataset with {len(dataset)} samples.")
    print(f"[INFO] Num features: {dataset.num_features}, Num classes: {dataset.num_classes}")

    # Train/validation split
    if 0.0 < args.val_split < 1.0 and len(dataset) > 1:
        val_size = int(len(dataset) * args.val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        if val_dataset is not None
        else None
    )

    model = PositionsNet(
        input_dim=dataset.num_features,
        hidden_dims=tuple(args.hidden_dims),
        num_classes=dataset.num_classes,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            label_lookup=dataset.index_to_label,
            log_examples=3,
        )
        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        else:
            val_loss, val_acc = 0.0, 0.0

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} "
            f"- val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        if val_loader is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_artifacts(model, args.output_dir, dataset, args)

    # If no validation set, save final model
    if val_loader is None:
        save_artifacts(model, args.output_dir, dataset, args)

    # ----- Per-clip actual vs predicted: use validation set if present, else full set -----
    if val_loader is not None:
        eval_loader = val_loader
        val_indices = val_dataset.indices  # original dataset indices for validation samples
        clip_names = [dataset.clip_names[i] for i in val_indices]
    else:
        eval_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
        clip_names = getattr(dataset, "clip_names", [str(i) for i in range(len(dataset))])

    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    all_confidence: List[float] = []
    with torch.no_grad():
        for inputs, targets in eval_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            pred_ids = probs.argmax(dim=1)
            confidences = probs.gather(1, pred_ids.unsqueeze(1)).squeeze(1).cpu().numpy()
            all_pred.extend(int(p) for p in pred_ids.cpu().numpy())
            all_true.extend(int(t) for t in targets.cpu().numpy())
            all_confidence.extend(float(c) for c in confidences)

    actual_labels: List[str] = []
    predicted_labels: List[str] = []
    for t_idx, p_idx in zip(all_true, all_pred):
        actual_labels.append(dataset.index_to_label.get(int(t_idx), str(t_idx)))
        predicted_labels.append(dataset.index_to_label.get(int(p_idx), str(p_idx)))

    result_df = pd.DataFrame(
        {
            "actual_play": actual_labels,
            "predicted_play": predicted_labels,
            "confidence": all_confidence[: len(actual_labels)],
        },
        index=clip_names[: len(actual_labels)],
    )

    os.makedirs(args.output_dir, exist_ok=True)
    result_path = os.path.join(args.output_dir, "play_predictions.csv")
    result_df.to_csv(result_path)
    print(f"[INFO] Saved per-clip predictions to {result_path}")
    print(result_df)


if __name__ == "__main__":
    main()

