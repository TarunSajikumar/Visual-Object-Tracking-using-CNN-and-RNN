"""Step 1: Train the CNN+RNN (ROLO-style) Stacked LSTM Object Tracker.

Usage:
    python train.py --sequences Car4 David3 --epochs 100
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model import CNNRNNTracker, get_device, load_checkpoint, save_checkpoint
from src import utils


class TrackingSequenceDataset(Dataset):
    """Memory-efficient sliding-window dataset: (num_steps, 4102) -> normalized [cx, cy, w, h]."""

    def __init__(
        self,
        data_dir: str | Path,
        sequences: List[str],
        num_steps: int = 6,
        feature_dropout: float = 0.0,
    ) -> None:
        self.num_steps = num_steps
        self.feature_dropout = feature_dropout
        self._seq_feats: List[np.ndarray] = []
        self._seq_gts: List[np.ndarray] = []
        self._index: List[Tuple[int, int]] = []

        for seq in sequences:
            seq_dir = Path(data_dir) / seq
            yolo_dir = seq_dir / "yolo_out"
            gt_path = seq_dir / "groundtruth_rect.txt"

            if not yolo_dir.is_dir() or not gt_path.is_file():
                print(f"[Skipping '{seq}': missing yolo_out/ or groundtruth_rect.txt]")
                continue

            w_img, h_img, _ = utils.get_sequence_metadata(data_dir, seq)
            yolo_files = utils.load_folder(yolo_dir, extensions={".npy"})
            gt_lines = utils.load_dataset_gt(gt_path)
            n_frames = min(len(yolo_files), len(gt_lines))

            if n_frames < num_steps:
                print(f"[Skipping '{seq}': only {n_frames} frames < num_steps={num_steps}]")
                continue

            feats = np.stack(
                [np.load(p).reshape(4102).astype(np.float32) for p in yolo_files[:n_frames]],
                axis=0,
            )
            gts_px = [utils.find_gt_location(gt_lines, i) for i in range(n_frames)]
            gts_norm = np.asarray(
                utils.locations_from_0_to_1(w_img, h_img, gts_px), dtype=np.float32
            )

            seq_idx = len(self._seq_feats)
            self._seq_feats.append(feats)
            self._seq_gts.append(gts_norm)
            for start_idx in range(0, n_frames - num_steps + 1):
                self._index.append((seq_idx, start_idx))

        if not self._index:
            raise RuntimeError(
                f"No valid training sequences found in '{data_dir}' for sequences={sequences}. "
                "Place dataset in data/<seq>/ (with img/ and groundtruth_rect.txt) "
                "and run `python extract.py --sequence <seq>` first."
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_idx, start_idx = self._index[idx]
        x = self._seq_feats[seq_idx][start_idx : start_idx + self.num_steps].copy()
        y = self._seq_gts[seq_idx][start_idx + self.num_steps - 1]
        if self.feature_dropout > 0.0:
            mask = (np.random.rand(x.shape[0], 4096) >= self.feature_dropout).astype(np.float32)
            x[:, :4096] *= mask
        return torch.from_numpy(x), torch.from_numpy(y)


def train(args: argparse.Namespace) -> None:
    device = get_device(prefer_cuda=not args.cpu)
    print(f"Training CNN+RNN Object Tracker on device: {device}")

    sequences = args.sequences or utils.discover_sequences(args.data_dir, require_gt=True)
    if not sequences:
        raise RuntimeError(
            f"No sequences with `yolo_out/` and `groundtruth_rect.txt` found in '{args.data_dir}'. "
            "Place sequences under data/<seq>/ and run `python extract.py --all` first."
        )
    print(f"Using training sequences ({len(sequences)}): {sequences}")

    dataset = TrackingSequenceDataset(
        data_dir=args.data_dir,
        sequences=sequences,
        num_steps=args.num_steps,
        feature_dropout=args.feature_dropout,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    print(f"Loaded {len(dataset)} sliding-window samples (num_steps={args.num_steps}).")

    if args.resume and Path(args.weights).is_file():
        model, _ = load_checkpoint(args.weights, device=device)
        model.train()
        print(f"Resumed weights from '{args.weights}'.")
    else:
        model = CNNRNNTracker(
            num_feat=4096,
            num_predict=6,
            feat_proj_dim=args.feat_proj_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "training-log.txt"

    best_loss = float("inf")
    with open(log_path, "a", encoding="utf-8") as log_file:
        for epoch in range(1, args.epochs + 1):
            model.train()
            running_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                pred_box, _ = model(batch_x)
                loss = criterion(pred_box, batch_y) * 100.0
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                running_loss += float(loss.item())
                n_batches += 1

            avg_loss = running_loss / max(1, n_batches)
            log_file.write(f"epoch={epoch} avg_loss={avg_loss:.6f}\n")
            log_file.flush()

            if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
                print(f"Epoch [{epoch:03d}/{args.epochs:03d}] - MSE*100 Loss: {avg_loss:.6f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    model,
                    args.weights,
                    extra={
                        "num_steps": args.num_steps,
                        "epoch": epoch,
                        "best_loss": best_loss,
                        "sequences": sequences,
                    },
                )

    print(f"Training complete! Best loss: {best_loss:.6f}")
    print(f"Model saved to: {args.weights}")
    print(f"Training log saved to: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN+RNN Stacked LSTM Object Tracker.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root dataset directory.")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Specific sequence names to train on (e.g. --sequences Car4 David3).",
    )
    parser.add_argument("--weights", type=str, default="weights/tracker_model.pt", help="Checkpoint output path.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for training logs.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument("--num-steps", type=int, default=6, help="Temporal sequence length (frames).")
    parser.add_argument("--feat-proj-dim", type=int, default=128, help="Projected visual feature dimension.")
    parser.add_argument("--hidden-size", type=int, default=128, help="LSTM hidden dimension.")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of stacked LSTM layers.")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout rate.")
    parser.add_argument("--feature-dropout", type=float, default=0.0, help="Input visual feature dropout.")
    parser.add_argument("--log-every", type=int, default=10, help="Print loss every N epochs.")
    parser.add_argument("--resume", action="store_true", help="Resume training from --weights if present.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
