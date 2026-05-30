from __future__ import annotations

from pathlib import Path

import torch
from torch import nn


class CompatibleGCNConv(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        # Names are kept compatible with torch_geometric style checkpoints:
        # convX.lin.weight and convX.bias
        self.lin = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = self.lin(x)
        return torch.matmul(adjacency, support) + self.bias


class FacialGCN(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 32, out_dim: int = 2) -> None:
        super().__init__()
        self.conv1 = CompatibleGCNConv(in_dim, hidden_dim)
        self.conv2 = CompatibleGCNConv(hidden_dim, hidden_dim)
        self.conv3 = CompatibleGCNConv(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x, adjacency))
        x = torch.relu(self.conv2(x, adjacency))
        return self.conv3(x, adjacency)


def build_adjacency(num_landmarks: int = 468) -> torch.Tensor:
    adjacency = torch.eye(num_landmarks)
    for index in range(num_landmarks - 1):
        adjacency[index, index + 1] = 1.0
        adjacency[index + 1, index] = 1.0

    degree = adjacency.sum(dim=1, keepdim=True).clamp(min=1.0)
    return adjacency / degree


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        if isinstance(state, dict):
            return state
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Unsupported checkpoint format.")


def _infer_dims_from_state_dict(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    conv1_weight = state_dict.get("conv1.lin.weight")
    conv3_weight = state_dict.get("conv3.lin.weight")
    if conv1_weight is None or conv3_weight is None:
        return 2, 32, 2

    hidden_dim = int(conv1_weight.shape[0])
    in_dim = int(conv1_weight.shape[1])
    out_dim = int(conv3_weight.shape[0])
    return in_dim, hidden_dim, out_dim


def load_model(model_path: Path, device: torch.device | None = None) -> tuple[FacialGCN, bool]:
    device = device or torch.device("cpu")
    model = FacialGCN().to(device)

    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = _extract_state_dict(checkpoint)
            in_dim, hidden_dim, out_dim = _infer_dims_from_state_dict(state_dict)
            model = FacialGCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            print(f"SoftPredict model loaded successfully from {model_path.resolve()}")
            print(f"SoftPredict architecture: in_dim={in_dim}, hidden_dim={hidden_dim}, out_dim={out_dim}")
            return model, True
        except Exception:
            print(f"SoftPredict failed to load weights from {model_path.resolve()}")
            model = FacialGCN().to(device)

    print(f"SoftPredict weights file missing or unavailable at {model_path.resolve()}")
    model.eval()
    return model, False
