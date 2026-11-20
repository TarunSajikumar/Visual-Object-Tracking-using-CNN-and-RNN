# Object Tracking using CNN and RNN (ROLO Architecture)

Modernized for **Python 3.12 + PyTorch + Ultralytics YOLOv8 + OpenCV 4**.

Based on:
- *Spatially Supervised Recurrent Convolutional Neural Networks for Visual Object Tracking*: https://arxiv.org/pdf/1607.05781.pdf
- *You Only Look Once: Unified, Real-Time Object Detection*: https://arxiv.org/pdf/1506.02640.pdf

---

## Clean Project Structure

```text
Object-Tracking-using-CNN-and-RNN/
├── src/                          # Core Deep Learning Modules
│   ├── model.py                  # PyTorch CNN + Stacked LSTM Tracker model
│   ├── feature_extractor.py      # YOLOv8 CNN backbone feature extractor (4102-D)
│   └── utils.py                  # Bounding box, IoU, metrics & helper functions
│
├── train.py                      # Step 1: Train the LSTM tracker on datasets
├── test.py                       # Step 2: Evaluate model on test video sequences
├── track.py                      # Step 3: Run tracker on any video file or webcam
├── extract.py                    # Utility: Pre-extract CNN features from video folders
├── app.py                        # Step 4: Live Webcam Tracking Web UI
│
├── web/                          # Web UI assets (HTML, CSS, JS)
├── weights/                      # Model checkpoints (tracker_model.pt, yolov8n.pt)
├── data/                         # Datasets & Sequences (Car4, David3, etc.)
├── results/                      # Output videos, logs, tracking results
├── presentation/                 # Class presentation (PowerPoint .pptx & notes)
├── requirements.txt              # Dependencies
└── README.md
```

---

## Quickstart

### 1. Activate Environment
```powershell
cd Object-Tracking-using-CNN-and-RNN
.\.venv\Scripts\Activate.ps1
```

### 2. Live Webcam Web App
```powershell
python app.py
```
Open **http://localhost:5000** in your browser and click **Start Camera**!
- Detects 80 COCO objects automatically.
- **Drag a box** around any object (like a pen, keys, watch) to classify (1,000 ImageNet classes) and track it visually!

### 3. Training & Evaluation Pipeline
```powershell
# 1. Extract 4102-D features from datasets in data/:
python extract.py --all

# 2. Train the Stacked LSTM Tracker:
python train.py --sequences Car4 David3 --epochs 100

# 3. Test and evaluate IoU:
python test.py --sequence David3 --save-video --show

# 4. Run tracking directly on any video or webcam:
python track.py --source "path/to/video.mp4" --output results/tracked.mp4 --show
python track.py --source 0 --show
```
