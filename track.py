"""Step 3: Real-time Object Tracking on any video file or live webcam.

Usage:
    # Run on a video file:
    python track.py --source path/to/video.mp4 --output results/tracked.mp4 --show

    # Run on live webcam (press 'q' to quit):
    python track.py --source 0 --show
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Deque, List, Optional

import cv2
import numpy as np
import torch

from src.model import get_device, load_checkpoint
from src.feature_extractor import YOLOFeatureExtractor
from src import utils


def run_tracking(args: argparse.Namespace) -> None:
    device = get_device(prefer_cuda=not args.cpu)
    model, extra = load_checkpoint(args.weights, device=device)
    num_steps = args.num_steps or int(extra.get("num_steps", 6))

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise IOError(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w_img = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_img = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w_img, h_img))

    extractor = YOLOFeatureExtractor(
        model_name=args.yolo_model,
        conf=args.conf,
        target_class=args.target_class,
        device=device,
    )

    window_buf: Deque[np.ndarray] = deque(maxlen=num_steps)
    prev_box: Optional[List[float]] = None
    frame_idx = 0

    print(
        f"Tracking source '{args.source}' ({w_img}x{h_img} @ {fps:.1f} FPS) "
        f"with CNN+RNN model on {device} (num_steps={num_steps})..."
    )

    try:
        model.eval()
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                feat_4102, selected_box = extractor.extract_frame(
                    frame, prev_box_cxcywh=prev_box
                )
                feat_vec = feat_4102.reshape(4102)
                window_buf.append(feat_vec)

                yolo_px = utils.locations_normal(w_img, h_img, feat_vec[4097:4101])
                rolo_px = None

                if len(window_buf) == num_steps:
                    seq_tensor = (
                        torch.from_numpy(np.stack(window_buf, axis=0))
                        .unsqueeze(0)
                        .to(device)
                    )
                    pred_box, _ = model(seq_tensor)
                    pred_np = pred_box.squeeze(0).cpu().numpy()
                    rolo_px = utils.locations_normal(w_img, h_img, pred_np)
                    prev_box = pred_np.tolist()
                elif selected_box is not None:
                    prev_box = selected_box

                annotated = utils.debug_3_locations(
                    frame,
                    gt_location=None,
                    yolo_location=yolo_px,
                    rolo_location=rolo_px,
                    show=False,
                    frame_idx=frame_idx,
                )

                if writer is not None:
                    writer.write(annotated)

                if args.show:
                    cv2.imshow("CNN+RNN Live Tracker (press 'q' to quit)", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                if args.max_frames and frame_idx >= args.max_frames:
                    break
    finally:
        extractor.close()
        cap.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Finished tracking {frame_idx} frames.")
    if args.output:
        print(f"Saved tracked video to: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track an object in video or webcam using YOLOv8 CNN + trained LSTM."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Path to video file (.mp4/.avi) or webcam index (default: 0).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/tracker_model.pt",
        help="Path to trained LSTM checkpoint from train.py.",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        default="weights/yolov8n.pt",
        help="YOLO model path in weights/.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="COCO class ID to track (e.g. 0=person, 2=car, 16=dog).",
    )
    parser.add_argument("--num-steps", type=int, default=None, help="Sequence window length.")
    parser.add_argument("--output", type=str, default="results/tracked.mp4", help="Output MP4 path.")
    parser.add_argument("--show", action="store_true", help="Display live OpenCV tracking window.")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU execution.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_tracking(args)


if __name__ == "__main__":
    main()
