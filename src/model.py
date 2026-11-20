"""PyTorch CNN+RNN (ROLO-style) Stacked LSTM Object Tracker model."""

import ctypes
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


def _windows_avail_commit_gb() -> Optional[float]:
    """Return available Windows commit memory (RAM + pagefile) in GiB, or None."""
    if sys.platform != "win32":
        return None
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return stat.ullAvailPageFile / (1024.0 ** 3)
    except Exception:
        return None
    return None


_CUDA_DISABLED_REASON: Optional[str] = None
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _CUDA_DISABLED_REASON = "CPU mode requested via --cpu."
elif "torch" not in sys.modules:
    _avail_commit = _windows_avail_commit_gb()
    if _avail_commit is not None and _avail_commit < 3.0:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("MKL_NUM_THREADS", "2")
        _CUDA_DISABLED_REASON = (
            f"Windows has only {_avail_commit:.2f} GiB of free virtual/commit memory "
            "(CUDA+cuDNN needs ~3.0+ GiB). Using CPU to avoid out-of-memory errors. "
            "Tip: Enable a Windows pagefile or close RAM-heavy apps to enable GPU acceleration."
        )

import torch
import torch.nn as nn


def _cuda_is_healthy() -> bool:
    """Verify CUDA + cuBLAS + cuDNN can safely initialize."""
    if torch.version.cuda is None:
        return False
    if _CUDA_DISABLED_REASON:
        print(f"[Info] {_CUDA_DISABLED_REASON}")
        return False
    if not torch.cuda.is_available():
        return False

    try:
        x = torch.zeros(1, 2, 8, device="cuda")
        lin = nn.Linear(8, 8).to("cuda")
        lstm = nn.LSTM(8, 8, batch_first=True).to("cuda")
        out, _ = lstm(lin(x))
        _ = out.sum().item()
        del x, lin, lstm, out
        torch.cuda.empty_cache()
        return True
    except Exception as exc:
        print(
            f"[Warning] CUDA detected ({torch.cuda.get_device_name(0)}) but failed health check "
            f"({exc.__class__.__name__}: {exc}). Falling back to CPU."
        )
        return False


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Select CUDA GPU if available, healthy, and requested, otherwise CPU."""
    if prefer_cuda and _cuda_is_healthy():
        return torch.device("cuda")
    return torch.device("cpu")


class CNNRNNTracker(nn.Module):
    """Stacked LSTM object tracker taking CNN visual features + YOLO bbox coordinates.

    Architecture:
        1. Visual feature projection: 4096 -> feat_proj_dim with LayerNorm + ReLU
        2. Concatenate projected visual features with 6 YOLO location parameters
        3. Stacked LSTM (default 3 layers) with dropout
        4. Linear prediction head + Sigmoid -> normalized [cx, cy, w, h] in [0, 1]
    """

    def __init__(
        self,
        num_feat: int = 4096,
        num_predict: int = 6,
        feat_proj_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()
        self.num_feat = num_feat
        self.num_predict = num_predict
        self.num_input = num_feat + num_predict
        self.feat_proj_dim = feat_proj_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.visual_proj = nn.Sequential(
            nn.Linear(num_feat, feat_proj_dim),
            nn.LayerNorm(feat_proj_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.loc_proj = nn.Sequential(
            nn.Linear(num_predict, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
        )

        lstm_input_dim = feat_proj_dim + 32 + num_predict
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, num_steps, num_feat + num_predict).
            hidden: Optional initial (h_0, c_0) LSTM state tuple.

        Returns:
            pred_box: Tensor of shape (batch, 4) with normalized [cx, cy, w, h].
            hidden_out: Final (h_n, c_n) LSTM state tuple.
        """
        visual_feats = x[:, :, : self.num_feat].contiguous()
        yolo_locs = x[:, :, self.num_feat : self.num_feat + self.num_predict].contiguous()

        vis_emb = self.visual_proj(visual_feats)
        loc_emb = self.loc_proj(yolo_locs)
        fused = torch.cat([vis_emb, loc_emb, yolo_locs], dim=-1)

        lstm_out, hidden_out = self.lstm(fused, hidden)
        last_out = lstm_out[:, -1, :]
        pred_box = self.head(last_out)
        return pred_box, hidden_out

    def get_config(self) -> Dict[str, Any]:
        return {
            "num_feat": self.num_feat,
            "num_predict": self.num_predict,
            "feat_proj_dim": self.feat_proj_dim,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }


def save_checkpoint(
    model: CNNRNNTracker,
    path: str | Path,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Save model weights and architecture config to a .pt checkpoint."""
    ckpt_path = Path(path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "config": model.get_config(),
        "state_dict": model.state_dict(),
    }
    if extra:
        payload["extra"] = extra
    torch.save(payload, ckpt_path)
    return ckpt_path


def load_checkpoint(
    path: str | Path,
    device: Optional[torch.device] = None,
) -> Tuple[CNNRNNTracker, Dict[str, Any]]:
    """Load a CNNRNNTracker model and metadata from a .pt checkpoint."""
    if device is None:
        device = get_device()
    ckpt_path = Path(path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt_path}'. "
            "Run python train.py first to train and save a model."
        )
    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = payload["config"]
    model = CNNRNNTracker(**config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload.get("extra", {})
