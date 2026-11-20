"""Step 2: Evaluate and test the CNN+RNN Stacked LSTM Object Tracker.

Usage:
    python test.py --sequence David3 --save-video --show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from src.model import get_device, load_checkpoint
from src import utils


def resolve_sequence(
    data_dir: str | Path, sequence: Optional[str]
) -> str:
    if sequence:
        return sequence
    available = utils.discover_sequences(data_dir, require_gt=False)
    if not available:
        raise RuntimeError(
            f"No sequences with `yolo_out/` found in '{data_dir}'. "
            "Extract features first with `python extract.py --sequence <name>`."
        )
    return available[0]


def test(args: argparse.Namespace) -> None:
    device = get_device(prefer_cuda=not args.cpu)
    model, extra = load_checkpoint(args.weights, device=device)
    num_steps = args.num_steps or int(extra.get("num_steps", 6))

    sequence_name = resolve_sequence(args.data_dir, args.sequence)
    seq_dir = Path(args.data_dir) / sequence_name
    img_dir = seq_dir / "img"
    x_path = seq_dir / "yolo_out"
    y_path = seq_dir / "groundtruth_rect.txt"
    out_path = seq_dir / "rolo_out_test"
    utils.createFolder(out_path)

    w_img, h_img, n_frames = utils.get_sequence_metadata(args.data_dir, sequence_name)
    img_files = utils.load_folder(img_dir, extensions=utils.IMAGE_EXTENSIONS)
    yolo_files = utils.load_folder(x_path, extensions={".npy"})
    n_frames = min(n_frames, len(yolo_files), len(img_files))

    has_gt = y_path.is_file()
    gt_lines = utils.load_dataset_gt(y_path) if has_gt else None
    if has_gt and gt_lines is not None:
        n_frames = min(n_frames, len(gt_lines))

    if n_frames < num_steps:
        raise RuntimeError(
            f"Sequence '{sequence_name}' has only {n_frames} frames (< num_steps={num_steps})."
        )

    video_writer = None
    video_path = None
    frames_dir = None
    if args.save_video:
        out_root = Path(args.output_dir)
        video_dir = out_root / "videos"
        frames_dir = out_root / "frames" / sequence_name
        utils.createFolder(video_dir)
        utils.createFolder(frames_dir)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = video_dir / f"{sequence_name}_tracked.mp4"
        video_writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (w_img, h_img))

    print(
        f"Evaluating CNN+RNN Tracker on '{sequence_name}' "
        f"({w_img}x{h_img}, {n_frames} frames, num_steps={num_steps}) on {device}..."
    )

    feats = np.stack(
        [np.load(p).reshape(4102).astype(np.float32) for p in yolo_files[:n_frames]],
        axis=0,
    )

    total_mse = 0.0
    total_rolo_iou = 0.0
    total_yolo_iou = 0.0
    count = 0

    model.eval()
    with torch.no_grad():
        for start_idx in range(0, n_frames - num_steps + 1):
            target_idx = start_idx + num_steps - 1
            frame_id = target_idx + 1

            window = (
                torch.from_numpy(feats[start_idx : start_idx + num_steps])
                .unsqueeze(0)
                .to(device)
            )
            pred_box, _ = model(window)
            pred_np = pred_box.squeeze(0).cpu().numpy().astype(np.float32)
            utils.save_rolo_output(out_path, pred_np, frame_id)

            pred_px = utils.locations_normal(w_img, h_img, pred_np)
            yolo_px = utils.locations_normal(w_img, h_img, feats[target_idx, 4097:4101])
            gt_px = None
            r_iou = None
            y_iou = None

            if has_gt and gt_lines is not None:
                gt_px = utils.find_gt_location(gt_lines, target_idx)
                gt_norm = utils.locations_from_0_to_1(w_img, h_img, [gt_px])[0]

                mse = float(
                    np.mean((pred_np - np.asarray(gt_norm, dtype=np.float32)) ** 2) * 100.0
                )
                r_iou = utils.cal_rolo_IOU(pred_px, gt_px)
                y_iou = utils.cal_yolo_IOU(yolo_px, gt_px)

                total_mse += mse
                total_rolo_iou += r_iou
                total_yolo_iou += y_iou
                if args.verbose:
                    print(
                        f"  Frame {frame_id:04d} | MSE*100={mse:.4f} | "
                        f"YOLO IoU={y_iou:.3f} | Tracker IoU={r_iou:.3f}"
                    )

            if args.save_video or args.show:
                img = utils.file_to_img(img_files[target_idx])
                if img is not None:
                    annotated = utils.debug_3_locations(
                        img,
                        gt_px,
                        yolo_px,
                        pred_px,
                        show=args.show,
                        wait_ms=max(1, int(1000 / args.fps)),
                        frame_idx=frame_id,
                        yolo_iou=y_iou,
                        rolo_iou=r_iou,
                    )
                    if video_writer is not None:
                        video_writer.write(annotated)
                    if frames_dir is not None:
                        cv2.imwrite(str(frames_dir / f"{frame_id:04d}.jpg"), annotated)

            count += 1

    if video_writer is not None:
        video_writer.release()
    if args.show:
        cv2.destroyAllWindows()

    print(f"Evaluation complete! Processed {count} frames.")
    if video_path is not None:
        print(f"  Tracked Video  : {video_path}")
        print(f"  Tracked Frames : {frames_dir}")
    if has_gt and count > 0:
        print(f"  Average MSE*100     : {total_mse / count:.6f}")
        print(f"  YOLO Average IoU    : {total_yolo_iou / count:.4f}")
        print(f"  CNN+RNN Average IoU : {total_rolo_iou / count:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CNN+RNN Stacked LSTM Object Tracker.")
    parser.add_argument("--data-dir", type=str, default="data", help="Root dataset directory.")
    parser.add_argument("--sequence", type=str, default=None, help="Sequence folder name in data/.")
    parser.add_argument("--weights", type=str, default="weights/tracker_model.pt", help="Model checkpoint path.")
    parser.add_argument("--num-steps", type=int, default=None, help="Override temporal sequence length.")
    parser.add_argument("--save-video", action="store_true", help="Export annotated MP4 video & frames.")
    parser.add_argument("--output-dir", type=str, default="results", help="Folder to save video output.")
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS.")
    parser.add_argument("--show", action="store_true", help="Display live OpenCV playback window.")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame metrics.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test(args)


if __name__ == "__main__":
    main()
