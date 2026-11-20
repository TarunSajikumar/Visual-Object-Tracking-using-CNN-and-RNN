"""Utility: Extract 4102-dim CNN+YOLO features for training sequences.

Usage:
    # Extract features for all sequences in data/:
    python extract.py --all

    # Extract for a single sequence:
    python extract.py --sequence Car4

    # Extract frames & features from a video file into data/<name>:
    python extract.py --video path/to/video.mp4 --sequence MyVideo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.feature_extractor import extract_sequence_features, video_to_frames
from src import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract 4102-dim CNN+YOLO features for OTB/MOT sequences or videos."
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Root dataset directory.")
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Sequence folder name inside data/ (must contain img/).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract features for every sequence inside data/ that has an img/ folder.",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Video file (.mp4/.avi) to unpack into data/<sequence>/img/ before extracting features.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Optional frame limit when reading --video."
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="weights/yolov8n.pt",
        help="YOLO model path.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Optional COCO class ID to track (e.g. 0=person, 2=car).",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.video:
        if not args.sequence:
            raise ValueError("Provide --sequence <name> when using --video.")
        img_dir = Path(args.data_dir) / args.sequence / "img"
        n = video_to_frames(args.video, img_dir, max_frames=args.max_frames)
        print(f"Extracted {n} frames from '{args.video}' to '{img_dir}'.")
        extract_sequence_features(
            data_dir=args.data_dir,
            sequence_name=args.sequence,
            model_name=args.yolo_model,
            conf=args.conf,
            target_class=args.target_class,
            prefer_cuda=not args.cpu,
        )
        return

    if args.all:
        root = Path(args.data_dir)
        if not root.is_dir():
            raise FileNotFoundError(
                f"Data directory '{root}' does not exist. Place sequences inside '{root}/<SequenceName>/img/'."
            )
        seqs = [
            p.name
            for p in sorted(root.iterdir(), key=lambda p: utils.alphanum_key(p.name))
            if p.is_dir() and (p / "img").is_dir()
        ]
        if not seqs:
            raise RuntimeError(f"No sequence folders with img/ found inside '{root}'.")
        for seq in seqs:
            extract_sequence_features(
                data_dir=args.data_dir,
                sequence_name=seq,
                model_name=args.yolo_model,
                conf=args.conf,
                target_class=args.target_class,
                prefer_cuda=not args.cpu,
            )
        return

    if not args.sequence:
        raise ValueError("Specify --sequence <name>, --all, or --video <path> --sequence <name>.")

    extract_sequence_features(
        data_dir=args.data_dir,
        sequence_name=args.sequence,
        model_name=args.yolo_model,
        conf=args.conf,
        target_class=args.target_class,
        prefer_cuda=not args.cpu,
    )


if __name__ == "__main__":
    main()
