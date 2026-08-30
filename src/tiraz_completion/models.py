"""Simple baselines and a trainable permutation-invariant category-set encoder."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from tiraz_completion.data import N_CLASSES
from tiraz_completion.protocol import validate_training_config


class CompletionNet(nn.Module):
    def __init__(self, embedding_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(N_CLASSES, embedding_dim) * 0.1)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, N_CLASSES)
        )

    def forward(self, x: Tensor) -> Tensor:
        count = x.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = x @ self.embedding / count
        features = torch.cat((pooled, count.log()), dim=1)
        logits: Tensor = self.decoder(features)
        return logits.masked_fill(x.bool(), -1e9)


def baseline_logits(
    train_x: NDArray[np.float32],
    train_y: NDArray[np.int64],
    x: NDArray[np.float32],
) -> dict[str, NDArray[np.float64]]:
    counts = np.bincount(train_y, minlength=N_CLASSES).astype(np.float64) + 1
    frequency = np.broadcast_to(np.log(counts / counts.sum()), x.shape).copy()
    complete = train_x.astype(np.float64).copy()
    complete[np.arange(len(train_y)), train_y] = 1
    pairs = complete.T @ complete
    conditional = (pairs + 1) / (complete.sum(axis=0)[None, :] + 2)
    cooccurrence = x @ np.log(conditional).T / np.maximum(x.sum(axis=1, keepdims=True), 1)
    for logits in (frequency, cooccurrence):
        logits[x.astype(bool)] = -1e9
    return {"frequency": frequency, "cooccurrence": cooccurrence}


def train_model(
    train_x: NDArray[np.float32],
    train_y: NDArray[np.int64],
    tune_x: NDArray[np.float32],
    tune_y: NDArray[np.int64],
    config: dict[str, Any],
    seed: int,
) -> tuple[CompletionNet, list[dict[str, float | int]]]:
    validate_training_config(config)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative integer")
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    model = CompletionNet(int(config["embedding_dim"]), int(config["hidden_dim"]))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    x, y = torch.from_numpy(train_x), torch.from_numpy(train_y)
    vx, vy = torch.from_numpy(tune_x), torch.from_numpy(tune_y)
    generator = torch.Generator().manual_seed(seed)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best = copy.deepcopy(model.state_dict())
    for epoch in range(int(config["epochs"])):
        model.train()
        ordering = torch.randperm(len(x), generator=generator)
        loss_sum = 0.0
        batch_size = int(config["batch_size"])
        for start in range(0, len(ordering), batch_size):
            ids = ordering[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x[ids]), y[ids])
            loss.backward()  # type: ignore[no-untyped-call]  # PyTorch lacks this stub.
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        model.eval()
        with torch.no_grad():
            tune_loss = float(nn.functional.cross_entropy(model(vx), vy))
        if not math.isfinite(loss_sum) or not math.isfinite(tune_loss):
            raise ValueError("non-finite training or tuning loss")
        if tune_loss < best_loss:
            best_loss, best = tune_loss, copy.deepcopy(model.state_dict())
        history.append(
            {"epoch": epoch + 1, "training_nll": loss_sum / len(x), "tuning_nll": tune_loss}
        )
    model.load_state_dict(best)
    model.eval()
    return model, history


def predict_logits(model: CompletionNet, x: NDArray[np.float32]) -> NDArray[np.float64]:
    with torch.no_grad():
        result = np.asarray(model(torch.from_numpy(x)).numpy(), dtype=np.float64)
    return result
