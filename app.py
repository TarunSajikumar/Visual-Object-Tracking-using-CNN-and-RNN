"""Step 4: Live Webcam Web App for CNN + RNN Object Tracking.

Usage:
    python app.py --port 5000
Then open http://localhost:5000 in your browser.
"""

from __future__ import annotations

import argparse
import base64
import collections
import ctypes
import json
import mimetypes
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
DEFAULT_WEIGHTS = ROOT_DIR / "weights" / "tracker_model.pt"
DEFAULT_YOLO_WEIGHTS = ROOT_DIR / "weights" / "yolov8n.pt"


def windows_avail_commit_gb() -> Optional[float]:
    """Return available Windows commit memory (RAM + pagefile) in GiB."""
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


class PythonTrackerBackend:
    """Lazily loaded YOLOv8 CNN + PyTorch Stacked LSTM tracker backend."""

    def __init__(
        self,
        weights_path: Path = DEFAULT_WEIGHTS,
        yolo_model: Path = DEFAULT_YOLO_WEIGHTS,
    ) -> None:
        self.weights_path = weights_path
        self.yolo_model = yolo_model
        self.lock = threading.Lock()
        self.ready = False
        self.loading = False
        self.error: Optional[str] = None
        self.device_name: str = "not loaded"
        self.has_trained_lstm = False
        self.num_steps = 6

        self._cv2 = None
        self._np = None
        self._torch = None
        self._extractor = None
        self._lstm_model = None
        self._window_buf: Deque[Any] = collections.deque(maxlen=self.num_steps)
        self._prev_box: Optional[List[float]] = None

    def status(self) -> Dict[str, Any]:
        avail = windows_avail_commit_gb()
        return {
            "ready": self.ready,
            "loading": self.loading,
            "error": self.error,
            "device": self.device_name,
            "has_trained_weights": self.weights_path.is_file(),
            "using_trained_lstm": self.has_trained_lstm,
            "weights_path": str(self.weights_path.relative_to(ROOT_DIR)) if self.weights_path.is_file() else "weights/tracker_model.pt",
            "avail_commit_gb": round(avail, 2) if avail is not None else None,
        }

    def reset(self) -> None:
        with self.lock:
            self._window_buf.clear()
            self._prev_box = None

    def load(self, force: bool = False) -> Dict[str, Any]:
        with self.lock:
            if self.ready:
                return self.status()
            if self.loading:
                return self.status()

            avail = windows_avail_commit_gb()
            if not force and avail is not None and avail < 1.5:
                self.error = (
                    f"Only {avail:.2f} GiB of free Windows virtual memory available "
                    "(Python YOLOv8 + PyTorch needs ~1.5+ GiB). Close extra apps or enable "
                    "a Windows pagefile, or continue using Browser WebGL CNN+RNN mode."
                )
                return self.status()

            self.loading = True
            self.error = None

        try:
            import cv2
            import numpy as np
            from src.model import CNNRNNTracker, get_device, load_checkpoint
            import torch
            from src.feature_extractor import YOLOFeatureExtractor

            device = get_device(prefer_cuda=True)
            yolo_p = str(self.yolo_model) if self.yolo_model.is_file() else "yolov8n.pt"
            extractor = YOLOFeatureExtractor(model_name=yolo_p, device=device)

            has_trained = False
            num_steps = 6
            if self.weights_path.is_file():
                lstm_model, extra = load_checkpoint(self.weights_path, device=device)
                num_steps = int(extra.get("num_steps", 6))
                has_trained = True
            else:
                lstm_model = CNNRNNTracker(
                    num_feat=4096, num_predict=6, feat_proj_dim=128, hidden_size=128
                ).to(device)
                lstm_model.eval()

            with self.lock:
                self._cv2 = cv2
                self._np = np
                self._torch = torch
                self._extractor = extractor
                self._lstm_model = lstm_model
                self.num_steps = num_steps
                self._window_buf = collections.deque(maxlen=num_steps)
                self._prev_box = None
                self.has_trained_lstm = has_trained
                self.device_name = str(device)
                self.ready = True
                self.loading = False
        except Exception as exc:
            with self.lock:
                self.loading = False
                self.ready = False
                self.error = f"{exc.__class__.__name__}: {exc}"

        return self.status()

    def track_base64_frame(
        self,
        image_data_url: str,
        target_class: Optional[int] = None,
        reset: bool = False,
    ) -> Dict[str, Any]:
        if not self.ready:
            raise RuntimeError(self.error or "Python backend is not loaded.")

        t0 = time.perf_counter()
        raw_b64 = image_data_url.split(",", 1)[1] if "," in image_data_url else image_data_url
        jpg_bytes = base64.b64decode(raw_b64)

        with self.lock:
            if reset:
                self._window_buf.clear()
                self._prev_box = None

            np = self._np
            cv2 = self._cv2
            torch = self._torch

            arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Could not decode JPEG frame.")

            h_img, w_img = frame.shape[:2]
            self._extractor.target_class = target_class
            feat_4102, selected_box = self._extractor.extract_frame(
                frame, prev_box_cxcywh=self._prev_box
            )
            feat_vec = feat_4102.reshape(4102)
            self._window_buf.append(feat_vec)

            cls_id = int(round(float(feat_vec[4096])))
            yolo_norm = [float(v) for v in feat_vec[4097:4101]]
            conf = float(feat_vec[4101])
            has_det = yolo_norm[2] > 0.0 and yolo_norm[3] > 0.0

            cnn_box = None
            if has_det:
                cx, cy, bw, bh = (
                    yolo_norm[0] * w_img,
                    yolo_norm[1] * h_img,
                    yolo_norm[2] * w_img,
                    yolo_norm[3] * h_img,
                )
                cnn_box = [cx - bw / 2.0, cy - bh / 2.0, bw, bh]

            rnn_box = None
            if self.has_trained_lstm and len(self._window_buf) == self.num_steps:
                seq = (
                    torch.from_numpy(np.stack(self._window_buf, axis=0))
                    .unsqueeze(0)
                    .to(self.device_name)
                )
                with torch.no_grad():
                    pred_box, _ = self._lstm_model(seq)
                pred_norm = pred_box.squeeze(0).cpu().numpy().tolist()
                cx, cy, bw, bh = (
                    pred_norm[0] * w_img,
                    pred_norm[1] * h_img,
                    pred_norm[2] * w_img,
                    pred_norm[3] * h_img,
                )
                rnn_box = [cx - bw / 2.0, cy - bh / 2.0, bw, bh]
                self._prev_box = pred_norm
            elif selected_box is not None:
                self._prev_box = selected_box
                rnn_box = cnn_box

            names = getattr(self._extractor.yolo, "names", {})
            label = names.get(cls_id, f"class_{cls_id}") if has_det else "target"

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "cnn_box": cnn_box,
            "rnn_box": rnn_box,
            "label": label,
            "conf": conf,
            "occluded": (not has_det) and (rnn_box is not None),
            "using_trained_lstm": self.has_trained_lstm,
            "latency_ms": round(latency_ms, 1),
        }


BACKEND = PythonTrackerBackend(weights_path=DEFAULT_WEIGHTS, yolo_model=DEFAULT_YOLO_WEIGHTS)

EXPLICIT_MIME_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
}


class TrackerWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        return

    def _send_json(self, payload: Dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            self._send_json(BACKEND.status())
            return

        if path == "/":
            file_path = WEB_DIR / "index.html"
        else:
            rel = path.lstrip("/")
            file_path = (WEB_DIR / rel).resolve()
            if not str(file_path).startswith(str(WEB_DIR.resolve())):
                self.send_error(HTTPStatus.FORBIDDEN)
                return

        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        ctype = EXPLICIT_MIME_TYPES.get(file_path.suffix.lower())
        if not ctype:
            ctype, _ = mimetypes.guess_type(str(file_path))
        data = file_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", ctype or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/backend/load":
            body = self._read_json()
            status = BACKEND.load(force=bool(body.get("force", False)))
            self._send_json(status)
            return

        if path == "/api/reset":
            BACKEND.reset()
            self._send_json({"ok": True})
            return

        if path == "/api/track":
            try:
                body = self._read_json()
                result = BACKEND.track_base64_frame(
                    image_data_url=body["image"],
                    target_class=body.get("target_class"),
                    reset=bool(body.get("reset", False)),
                )
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return

        self.send_error(HTTPStatus.NOT_FOUND)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Webcam CNN+RNN Object Tracking Web App.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface.")
    parser.add_argument("--port", type=int, default=5000, help="HTTP port.")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(DEFAULT_WEIGHTS),
        help="Path to trained LSTM checkpoint (weights/tracker_model.pt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    BACKEND.weights_path = Path(args.weights)
    server = ThreadingHTTPServer((args.host, args.port), TrackerWebHandler)
    print(f"CNN+RNN Live Webcam Tracker running at http://localhost:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
