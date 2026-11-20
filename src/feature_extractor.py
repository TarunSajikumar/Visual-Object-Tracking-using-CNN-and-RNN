"""CNN + YOLO feature extractor for OTB/MOT image sequences and videos.

Extracts per-frame 4102-dim `.npy` files (`4096` YOLOv8 CNN backbone visual features
+ `6` detection coordinates `[class_id, cx, cy, w, h, conf]`) into
`data/<sequence>/yolo_out/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.model import get_device
from src import utils


def _pool_to_4096(feat_tensor: torch.Tensor) -> np.ndarray:
    """Convert a 4D CNN feature map (1, C, H, W) to a 4096-dim vector."""
    pooled = F.adaptive_avg_pool2d(feat_tensor.float(), (4, 4)).flatten(1)
    vec = (
        F.interpolate(pooled.unsqueeze(1), size=4096, mode="linear", align_corners=False)
        .squeeze(0)
        .squeeze(0)
    )
    vec = F.normalize(vec, dim=0)
    return vec.detach().cpu().numpy().astype(np.float32)


class YOLOFeatureExtractor:
    """Wraps Ultralytics YOLOv8 to extract 4096 CNN backbone features + 6 bbox params."""

    def __init__(
        self,
        model_name: str | Path = "weights/yolov8n.pt",
        conf: float = 0.25,
        target_class: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        from ultralytics import YOLO

        self.device = device or get_device()
        self.conf = conf
        self.target_class = target_class

        # Check weights path fallback
        weights_p = Path(model_name)
        if not weights_p.is_file() and (Path("weights") / weights_p.name).is_file():
            weights_p = Path("weights") / weights_p.name

        self.yolo = YOLO(str(weights_p))
        self.yolo.to(str(self.device))

        self._captured_feat: Optional[torch.Tensor] = None
        backbone_layer = self.yolo.model.model[9]  # YOLOv8 SPPF backbone output
        self._hook = backbone_layer.register_forward_hook(self._save_feat_hook)

    def _save_feat_hook(self, _module, _inputs, output) -> None:
        if isinstance(output, (tuple, list)):
            self._captured_feat = output[0].detach()
        else:
            self._captured_feat = output.detach()

    def close(self) -> None:
        if hasattr(self, "_hook") and self._hook:
            self._hook.remove()

    def extract_frame(
        self,
        img_bgr: np.ndarray,
        gt_tlwh: Optional[List[float]] = None,
        prev_box_cxcywh: Optional[List[float]] = None,
    ) -> Tuple[np.ndarray, Optional[List[float]]]:
        """Run YOLOv8 on one BGR frame and return `(feature_4102, selected_box_norm)`."""
        h_img, w_img = img_bgr.shape[:2]
        self._captured_feat = None

        results = self.yolo.predict(
            source=img_bgr,
            conf=self.conf,
            verbose=False,
            device=str(self.device),
        )
        res = results[0]

        if self._captured_feat is not None:
            vis_feat = _pool_to_4096(self._captured_feat)
        else:
            vis_feat = np.zeros(4096, dtype=np.float32)

        best_loc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # cls, cx, cy, w, h, conf
        selected_cxcywh: Optional[List[float]] = None

        if res.boxes is not None and len(res.boxes) > 0:
            xywhn = res.boxes.xywhn.detach().cpu().numpy()
            confs = res.boxes.conf.detach().cpu().numpy()
            clss = res.boxes.cls.detach().cpu().numpy()

            candidates = []
            for box_n, score, cls_id in zip(xywhn, confs, clss):
                if self.target_class is not None and int(cls_id) != self.target_class:
                    continue
                candidates.append((float(cls_id), box_n.tolist(), float(score)))

            if candidates:
                if gt_tlwh is not None:
                    best_score = -1.0
                    for cls_id, box_n, score in candidates:
                        box_px = utils.locations_normal(w_img, h_img, box_n)
                        ov = utils.cal_yolo_IOU(box_px, gt_tlwh)
                        if ov > best_score:
                            best_score = ov
                            best_loc = [cls_id, *box_n, score]
                            selected_cxcywh = box_n
                elif prev_box_cxcywh is not None:
                    best_metric = -1e9
                    for cls_id, box_n, score in candidates:
                        ov = utils.iou(box_n, prev_box_cxcywh)
                        dist = (box_n[0] - prev_box_cxcywh[0]) ** 2 + (
                            box_n[1] - prev_box_cxcywh[1]
                        ) ** 2
                        metric = ov - 0.5 * dist + 0.1 * score
                        if metric > best_metric:
                            best_metric = metric
                            best_loc = [cls_id, *box_n, score]
                            selected_cxcywh = box_n
                else:
                    cls_id, box_n, score = max(candidates, key=lambda c: c[2])
                    best_loc = [cls_id, *box_n, score]
                    selected_cxcywh = box_n

        feat_4102 = np.concatenate(
            [vis_feat, np.asarray(best_loc, dtype=np.float32)], axis=0
        ).reshape(1, 4102)
        return feat_4102, selected_cxcywh


def video_to_frames(
    video_path: str | Path, out_img_dir: str | Path, max_frames: Optional[int] = None
) -> int:
    """Extract frames from a video file into `out_img_dir/0001.jpg`, etc."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    utils.createFolder(out_img_dir)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        cv2.imwrite(str(Path(out_img_dir) / f"{count:04d}.jpg"), frame)
        if max_frames is not None and count >= max_frames:
            break
    cap.release()
    return count


def extract_sequence_features(
    data_dir: str | Path,
    sequence_name: str,
    model_name: str | Path = "weights/yolov8n.pt",
    conf: float = 0.25,
    target_class: Optional[int] = None,
    prefer_cuda: bool = True,
) -> None:
    """Run YOLOv8 CNN feature extraction on `data/<sequence_name>/img/`."""
    seq_dir = Path(data_dir) / sequence_name
    img_dir = seq_dir / "img"
    yolo_dir = seq_dir / "yolo_out"
    gt_file = seq_dir / "groundtruth_rect.txt"

    img_paths = utils.load_folder(img_dir, extensions=utils.IMAGE_EXTENSIONS)
    if not img_paths:
        raise ValueError(f"No images found in '{img_dir}'.")

    gt_lines = utils.load_dataset_gt(gt_file) if gt_file.is_file() else None
    utils.createFolder(yolo_dir)

    extractor = YOLOFeatureExtractor(
        model_name=model_name,
        conf=conf,
        target_class=target_class,
        device=get_device(prefer_cuda=prefer_cuda),
    )
    prev_box: Optional[List[float]] = None

    try:
        print(
            f"Extracting CNN+YOLO features for '{sequence_name}' "
            f"({len(img_paths)} frames) on {extractor.device}..."
        )
        for idx, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            if img is None:
                continue
            gt_box = (
                utils.find_gt_location(gt_lines, idx)
                if gt_lines is not None and idx < len(gt_lines)
                else None
            )
            feat, selected_box = extractor.extract_frame(
                img, gt_tlwh=gt_box, prev_box_cxcywh=prev_box
            )
            if selected_box is not None:
                prev_box = selected_box
            np.save(yolo_dir / f"{idx + 1:04d}.npy", feat)
            if (idx + 1) % 25 == 0 or idx + 1 == len(img_paths):
                print(f"  Processed {idx + 1}/{len(img_paths)} frames")
    finally:
        extractor.close()
    print(f"Saved CNN+YOLO features to '{yolo_dir}'.")
