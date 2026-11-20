"""Utility functions and helpers for CNN+RNN (ROLO) Object Tracking.

Cleaned, documented, and modernized for Python 3.
"""

from __future__ import annotations

import configparser
import math
import os
import pickle
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mp4", ".mov"}


def tryint(s: str):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s: str):
    """Turn a string into a list of string and number chunks for natural sorting."""
    return [tryint(c) for c in re.split(r"([0-9]+)", str(s))]


def sort_nicely(items: List[str]) -> List[str]:
    """Sort the given list in natural human numeric order."""
    items.sort(key=alphanum_key)
    return items


def createFolder(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def load_folder(path: str | Path, extensions: Optional[set[str]] = None) -> List[str]:
    """Return naturally sorted file paths inside a directory."""
    folder = Path(path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")
    paths = [
        str(p)
        for p in folder.iterdir()
        if p.is_file() and (extensions is None or p.suffix.lower() in extensions)
    ]
    return sort_nicely(paths)


def load_dataset_gt(gt_file: str | Path) -> List[str]:
    """Load ground-truth text lines."""
    with open(gt_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.read().splitlines() if line.strip()]
    return lines


def find_gt_location(lines: Sequence[str], idx: int) -> List[float]:
    """Parse ground-truth bounding box [x1, y1, w, h] at 0-based index `idx`."""
    line = lines[idx].strip()
    elems = re.split(r"[\t,\s]+", line)
    if len(elems) < 4:
        raise ValueError(f"Invalid ground-truth line at index {idx}: '{line}'")
    return [float(elems[0]), float(elems[1]), float(elems[2]), float(elems[3])]


def find_yolo_location(fold: str | Path, frame_id: int) -> np.ndarray:
    """Load YOLO box [cx, cy, w, h] (normalized) for 1-based `frame_id`."""
    paths = load_folder(fold, extensions={".npy"})
    path = paths[frame_id - 1]
    yolo_output = np.load(path).reshape(-1)
    return yolo_output[4097:4101].astype(np.float32).copy()


def find_rolo_location(fold: str | Path, frame_id: int) -> np.ndarray:
    """Load tracker predicted box [cx, cy, w, h] (normalized) for 1-based `frame_id`."""
    path = Path(fold) / f"{frame_id}.npy"
    return np.load(path).reshape(-1).astype(np.float32).copy()


def file_to_img(filepath: str | Path) -> Optional[np.ndarray]:
    return cv2.imread(str(filepath))


def locations_normal(wid: float, ht: float, locations: Sequence[float]) -> List[float]:
    """Scale normalized [cx, cy, w, h] in [0, 1] back to pixel coordinates."""
    wid = float(wid)
    ht = float(ht)
    return [
        float(locations[0]) * wid,
        float(locations[1]) * ht,
        float(locations[2]) * wid,
        float(locations[3]) * ht,
    ]


def locations_from_0_to_1(
    wid: float, ht: float, locations: Sequence[Sequence[float]]
) -> List[List[float]]:
    """Convert top-left pixel boxes [x1, y1, w, h] to normalized center boxes [cx, cy, w, h]."""
    wid = float(wid)
    ht = float(ht)
    normalized: List[List[float]] = []
    for box in locations:
        x1, y1, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        cx = (x1 + w / 2.0) / wid
        cy = (y1 + h / 2.0) / ht
        normalized.append([cx, cy, w / wid, h / ht])
    return normalized


def validate_box(box: Sequence[float]) -> List[float]:
    return [0.0 if math.isnan(float(v)) else float(v) for v in box]


def iou_tlwh(box1: Sequence[float], box2: Sequence[float]) -> float:
    """Exact Intersection over Union (IoU) between two top-left boxes [x1, y1, w, h]."""
    b1 = validate_box(box1)
    b2 = validate_box(box2)

    x1_l, y1_t, w1, h1 = b1[0], b1[1], max(0.0, b1[2]), max(0.0, b1[3])
    x2_l, y2_t, w2, h2 = b2[0], b2[1], max(0.0, b2[2]), max(0.0, b2[3])

    inter_l = max(x1_l, x2_l)
    inter_t = max(y1_t, y2_t)
    inter_r = min(x1_l + w1, x2_l + w2)
    inter_b = min(y1_t + h1, y2_t + h2)

    inter_w = max(0.0, inter_r - inter_l)
    inter_h = max(0.0, inter_b - inter_t)
    intersection = inter_w * inter_h

    union = w1 * h1 + w2 * h2 - intersection
    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def iou(box1: Sequence[float], box2: Sequence[float]) -> float:
    """IoU between two center-format boxes [cx, cy, w, h]."""
    b1 = validate_box(box1)
    b2 = validate_box(box2)
    tl1 = [b1[0] - b1[2] / 2.0, b1[1] - b1[3] / 2.0, b1[2], b1[3]]
    tl2 = [b2[0] - b2[2] / 2.0, b2[1] - b2[3] / 2.0, b2[2], b2[3]]
    return iou_tlwh(tl1, tl2)


def cal_rolo_IOU(location: Sequence[float], gt_location: Sequence[float]) -> float:
    """IoU between tracker pixel box [cx, cy, w, h] and GT pixel box [x1, y1, w, h]."""
    tl_loc = [
        float(location[0]) - float(location[2]) / 2.0,
        float(location[1]) - float(location[3]) / 2.0,
        float(location[2]),
        float(location[3]),
    ]
    return iou_tlwh(tl_loc, gt_location)


def cal_yolo_IOU(location: Sequence[float], gt_location: Sequence[float]) -> float:
    """IoU between YOLO pixel box [cx, cy, w, h] and GT pixel box [x1, y1, w, h]."""
    return cal_rolo_IOU(location, gt_location)


def save_rolo_output(
    out_fold: str | Path,
    rolo_output: np.ndarray,
    frame_id: int,
) -> None:
    """Save single predicted bounding box to `<out_fold>/<frame_id>.npy`."""
    createFolder(out_fold)
    path = Path(out_fold) / f"{frame_id}.npy"
    np.save(path, np.asarray(rolo_output, dtype=np.float32))


def debug_3_locations(
    img: np.ndarray,
    gt_location: Optional[Sequence[float]],
    yolo_location: Optional[Sequence[float]],
    rolo_location: Optional[Sequence[float]],
    show: bool = False,
    wait_ms: int = 30,
    frame_idx: Optional[int] = None,
    yolo_iou: Optional[float] = None,
    rolo_iou: Optional[float] = None,
) -> np.ndarray:
    """Draw GT (red), YOLO (blue), and CNN+RNN Tracker (green) boxes onto frame."""
    img_cp = img.copy()

    if gt_location is not None:
        x, y, w, h = [int(round(v)) for v in gt_location]
        cv2.rectangle(img_cp, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if yolo_location is not None and yolo_location[2] > 0 and yolo_location[3] > 0:
        cx, cy, w, h = [int(round(v)) for v in yolo_location]
        cv2.rectangle(
            img_cp, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (255, 0, 0), 2
        )

    if rolo_location is not None and rolo_location[2] > 0 and rolo_location[3] > 0:
        cx, cy, w, h = [int(round(v)) for v in rolo_location]
        cv2.rectangle(
            img_cp, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), (0, 255, 0), 2
        )

    overlay_lines = []
    if frame_idx is not None:
        overlay_lines.append((f"Frame: {frame_idx}", (255, 255, 255)))
    if gt_location is not None:
        overlay_lines.append(("GT", (0, 0, 255)))
    if yolo_location is not None:
        txt = f"YOLO (IoU {yolo_iou:.2f})" if yolo_iou is not None else "YOLO"
        overlay_lines.append((txt, (255, 160, 0)))
    if rolo_location is not None:
        txt = f"CNN+RNN (IoU {rolo_iou:.2f})" if rolo_iou is not None else "CNN+RNN"
        overlay_lines.append((txt, (0, 255, 0)))

    if overlay_lines:
        box_h = 18 * len(overlay_lines) + 8
        cv2.rectangle(img_cp, (4, 4), (190, 4 + box_h), (20, 20, 20), -1)
        for idx, (text, color) in enumerate(overlay_lines):
            cv2.putText(
                img_cp,
                text,
                (10, 20 + idx * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )

    if show:
        cv2.imshow("CNN+RNN Object Tracker", img_cp)
        cv2.waitKey(wait_ms)

    return img_cp


def get_sequence_metadata(
    data_dir: str | Path, sequence_name: str
) -> Tuple[int, int, int]:
    """Infer (width, height, num_frames) from a sequence directory on disk."""
    seq_dir = Path(data_dir) / sequence_name
    img_dir = seq_dir / "img"
    yolo_dir = seq_dir / "yolo_out"

    if img_dir.is_dir():
        img_paths = load_folder(img_dir, extensions=IMAGE_EXTENSIONS)
        if not img_paths:
            raise ValueError(f"No images found in '{img_dir}'.")
        sample = cv2.imread(img_paths[0])
        if sample is None:
            raise ValueError(f"Could not read image '{img_paths[0]}'.")
        h_img, w_img = sample.shape[:2]
        num_frames = len(img_paths)
        if yolo_dir.is_dir():
            yolo_paths = load_folder(yolo_dir, extensions={".npy"})
            if yolo_paths:
                num_frames = min(num_frames, len(yolo_paths))
        return w_img, h_img, num_frames

    raise FileNotFoundError(
        f"Expected image folder at '{img_dir}'. "
        "Place images in data/<sequence>/img/ and run python extract.py."
    )


def discover_sequences(data_dir: str | Path, require_gt: bool = False) -> List[str]:
    """List valid sequence names inside `data_dir` that contain `yolo_out/`."""
    root = Path(data_dir)
    if not root.is_dir():
        return []
    sequences = []
    for entry in sorted(root.iterdir(), key=lambda p: alphanum_key(p.name)):
        if not entry.is_dir():
            continue
        has_yolo = (entry / "yolo_out").is_dir() and any(
            (entry / "yolo_out").glob("*.npy")
        )
        has_gt = (entry / "groundtruth_rect.txt").is_file()
        if has_yolo and (has_gt or not require_gt):
            sequences.append(entry.name)
    return sequences
