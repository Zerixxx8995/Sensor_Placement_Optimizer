"""
core/surrogate_model.py
-----------------------
PyTorch MLP Surrogate Model for predicting sensor deployment fitness scores.

Replaces expensive analytical fitness evaluation during early PSO iterations.
- Input: flattened sensor positions (padded up to MAX_NODES) + config params (Rs, Rc, w1, w2, w3, area_W, area_H)
- Architecture: FC(256) -> ReLU -> Dropout(0.2) -> FC(128) -> ReLU -> Dropout(0.2) -> FC(64) -> ReLU -> FC(1)
- Output: single scalar fitness score
"""

import os
import logging
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

MAX_NODES = 100
PARAM_DIM = 7  # Rs, Rc, w1, w2, w3, area_W, area_H
INPUT_DIM = MAX_NODES * 2 + PARAM_DIM


class FitnessSurrogateMLP(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def encode_features(positions: np.ndarray, config: dict, max_nodes: int = MAX_NODES) -> np.ndarray:
    """
    Encode sensor positions and config parameters into a fixed-length feature vector.

    Args:
        positions: (N, 2) array of sensor (x, y) coords.
        config: dict containing area (width/height), sensing_radius, comm_radius, weights (w1,w2,w3)
        max_nodes: max supported nodes for zero-padding.

    Returns:
        1D float32 numpy array of length MAX_NODES * 2 + PARAM_DIM.
    """
    pos_flat = np.zeros(max_nodes * 2, dtype=np.float32)
    n = min(len(positions), max_nodes)
    if n > 0:
        pos_flat[: n * 2] = positions[:n].flatten()

    area = config.get("area", {})
    area_w = float(area.get("width", config.get("area_W", 100.0)))
    area_h = float(area.get("height", config.get("area_H", 100.0)))
    rs = float(config.get("sensing_radius", config.get("Rs", 10.0)))
    rc = float(config.get("comm_radius", config.get("Rc", 20.0)))
    weights = config.get("weights", {})
    w1 = float(weights.get("w1", config.get("w1", 0.5)))
    w2 = float(weights.get("w2", config.get("w2", 0.3)))
    w3 = float(weights.get("w3", config.get("w3", 0.2)))

    params = np.array([rs, rc, w1, w2, w3, area_w, area_h], dtype=np.float32)
    return np.concatenate([pos_flat, params])


def train_surrogate(
    training_runs: list[dict],
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str | Path | None = None,
) -> tuple[FitnessSurrogateMLP, dict]:
    """
    Train the surrogate MLP on historical runs.

    Args:
        training_runs: list of dicts with keys 'config', 'best_positions', 'fitness_history' or 'coverage_ratio'
        epochs: number of training epochs
        batch_size: batch size
        lr: learning rate
        save_path: path to save checkpoint .pt file

    Returns:
        (trained_model, metrics_dict)
    """
    X_list = []
    y_list = []

    for run in training_runs:
        cfg = run.get("config", {})
        best_pos = np.array(run.get("best_positions", []))
        fitness_hist = run.get("fitness_history", [])

        if len(best_pos) > 0 and len(fitness_hist) > 0:
            final_fit = float(fitness_hist[-1])
            x_vec = encode_features(best_pos, cfg)
            X_list.append(x_vec)
            y_list.append(final_fit)

    if not X_list:
        raise ValueError("No valid training samples found in training_runs")

    X_train = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_train = torch.tensor(np.array(y_list), dtype=torch.float32)

    model = FitnessSurrogateMLP()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)
        final_loss = epoch_loss / len(dataset)

    model.eval()
    metrics = {
        "samples_count": len(X_list),
        "epochs": epochs,
        "final_mse_loss": float(final_loss),
    }

    if save_path:
        save_checkpoint(model, save_path, metrics)

    return model, metrics


def save_checkpoint(model: nn.Module, path: str | Path, metadata: dict | None = None) -> None:
    """Save PyTorch model state and metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    torch.save(payload, str(path))
    logger.info("Saved surrogate model checkpoint to %s", path)


def load_checkpoint(path: str | Path) -> tuple[FitnessSurrogateMLP, dict]:
    """Load PyTorch model from checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    checkpoint = torch.load(str(path), weights_only=False)
    model = FitnessSurrogateMLP()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint.get("metadata", {})


def predict_surrogate_batch(
    model: FitnessSurrogateMLP,
    positions_batch: np.ndarray,
    config: dict,
) -> np.ndarray:
    """
    Predict fitness scores for a batch of particles.

    Args:
        model: Trained FitnessSurrogateMLP in eval mode
        positions_batch: Array of shape (P, N, 2)
        config: Optimization config dict

    Returns:
        Array of shape (P,) with predicted fitness scores.
    """
    model.eval()
    P = len(positions_batch)
    X_vecs = [encode_features(positions_batch[i], config) for i in range(P)]
    X_tensor = torch.tensor(np.array(X_vecs), dtype=torch.float32)

    with torch.no_grad():
        preds = model(X_tensor).numpy()

    return preds
