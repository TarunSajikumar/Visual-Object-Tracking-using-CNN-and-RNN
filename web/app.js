/**
 * Live Webcam CNN + RNN Object Tracker Frontend
 *
 * Supports:
 * 1. Auto Detect Mode (80 COCO classes via MobileNetV2-SSD + 6-step Recurrent Tracker)
 * 2. Track ANY Object Mode (Drag a box around a pen, keys, watch, etc.):
 *    - Classifies the object using MobileNetV2 CNN (1,000 ImageNet classes, incl. Ballpoint Pen!)
 *    - Tracks it continuously via Visual Template Matching + Recurrent Velocity/Occlusion State
 * 3. Python Backend Mode (YOLOv8 CNN + PyTorch Stacked LSTM via /api/track)
 */

const COCO_NAME_TO_ID = {
  person: 0,
  car: 2,
  cat: 15,
  dog: 16,
  bottle: 39,
  cup: 41,
  chair: 56,
  laptop: 63,
  "cell phone": 67,
  book: 73,
};

function computeIoU(a, b) {
  const x1 = Math.max(a[0], b[0]);
  const y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[0] + a[2], b[0] + b[2]);
  const y2 = Math.min(a[1] + a[3], b[1] + b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a[2] * a[3] + b[2] * b[3] - inter;
  return union > 0 ? inter / union : 0;
}

function formatImageNetLabel(raw) {
  if (!raw) return "Custom Object";
  const lower = raw.toLowerCase();
  if (lower.includes("ballpoint")) return "Ballpoint Pen";
  if (lower.includes("fountain pen")) return "Fountain Pen";
  if (lower.includes("quill")) return "Pen / Quill";
  if (lower.includes("pencil")) return "Pencil";
  const first = raw.split(",")[0].trim();
  return first.replace(/\b\w/g, (c) => c.toUpperCase());
}

/**
 * Recurrent 6-step Temporal Tracker for CNN detections (COCO / YOLO).
 */
class RecurrentTemporalTracker {
  constructor(windowSize = 6, maxMissedFrames = 14) {
    this.windowSize = windowSize;
    this.maxMissedFrames = maxMissedFrames;
    this.reset();
  }

  reset() {
    this.history = [];
    this.hiddenBox = null;
    this.velocity = [0, 0, 0, 0];
    this.label = null;
    this.conf = 0;
    this.missed = 0;
    this.trail = [];
    this.clickTarget = null;
  }

  lockPoint(x, y) {
    this.clickTarget = { x, y };
  }

  lockDetection(det) {
    const [x, y, w, h] = det.bbox;
    this.history = [];
    this.trail = [];
    this.hiddenBox = [x + w / 2, y + h / 2, w, h];
    this.velocity = [0, 0, 0, 0];
    this.label = det.class;
    this.conf = det.score;
    this.missed = 0;
  }

  update(detections) {
    let chosen = null;

    if (this.clickTarget && detections.length > 0) {
      let bestDist = Infinity;
      for (const det of detections) {
        const [x, y, w, h] = det.bbox;
        const cx = x + w / 2;
        const cy = y + h / 2;
        const inside =
          this.clickTarget.x >= x &&
          this.clickTarget.x <= x + w &&
          this.clickTarget.y >= y &&
          this.clickTarget.y <= y + h;
        const dist =
          Math.hypot(cx - this.clickTarget.x, cy - this.clickTarget.y) - (inside ? 1200 : 0);
        if (dist < bestDist) {
          bestDist = dist;
          chosen = det;
        }
      }
      this.clickTarget = null;
      if (chosen) {
        this.history = [];
        this.trail = [];
        this.hiddenBox = null;
        this.velocity = [0, 0, 0, 0];
      }
    } else if (this.hiddenBox && detections.length > 0) {
      const prior = [
        this.hiddenBox[0] + this.velocity[0],
        this.hiddenBox[1] + this.velocity[1],
        Math.max(12, this.hiddenBox[2] + this.velocity[2]),
        Math.max(12, this.hiddenBox[3] + this.velocity[3]),
      ];
      const priorTlwh = [
        prior[0] - prior[2] / 2,
        prior[1] - prior[3] / 2,
        prior[2],
        prior[3],
      ];

      let bestScore = -Infinity;
      for (const det of detections) {
        const iou = computeIoU(det.bbox, priorTlwh);
        const cx = det.bbox[0] + det.bbox[2] / 2;
        const cy = det.bbox[1] + det.bbox[3] / 2;
        const distNorm =
          Math.hypot(cx - prior[0], cy - prior[1]) / Math.max(40, Math.hypot(prior[2], prior[3]));
        const classBonus = det.class === this.label ? 0.25 : 0;
        const score = iou * 2.0 - distNorm * 0.65 + classBonus + det.score * 0.15;
        if (score > bestScore && (iou > 0.05 || distNorm < 1.8)) {
          bestScore = score;
          chosen = det;
        }
      }
    } else if (detections.length > 0) {
      chosen = detections.reduce((best, cur) => (cur.score > best.score ? cur : best), detections[0]);
    }

    if (chosen) {
      const [x, y, w, h] = chosen.bbox;
      const obs = [x + w / 2, y + h / 2, w, h];

      if (!this.hiddenBox) {
        this.hiddenBox = [...obs];
        this.velocity = [0, 0, 0, 0];
      } else {
        const alphaPos = 0.56;
        const alphaSize = 0.34;
        const alphaVel = 0.42;
        const pred = [
          this.hiddenBox[0] + this.velocity[0],
          this.hiddenBox[1] + this.velocity[1],
          this.hiddenBox[2] + this.velocity[2],
          this.hiddenBox[3] + this.velocity[3],
        ];
        const nextBox = [
          pred[0] + alphaPos * (obs[0] - pred[0]),
          pred[1] + alphaPos * (obs[1] - pred[1]),
          Math.max(12, pred[2] + alphaSize * (obs[2] - pred[2])),
          Math.max(12, pred[3] + alphaSize * (obs[3] - pred[3])),
        ];
        this.velocity = [
          (1 - alphaVel) * this.velocity[0] + alphaVel * (nextBox[0] - this.hiddenBox[0]),
          (1 - alphaVel) * this.velocity[1] + alphaVel * (nextBox[1] - this.hiddenBox[1]),
          (1 - alphaVel) * this.velocity[2] + alphaVel * (nextBox[2] - this.hiddenBox[2]),
          (1 - alphaVel) * this.velocity[3] + alphaVel * (nextBox[3] - this.hiddenBox[3]),
        ];
        this.hiddenBox = nextBox;
      }

      this.label = chosen.class;
      this.conf = chosen.score;
      this.missed = 0;
      this.history.push([...this.hiddenBox]);
      if (this.history.length > this.windowSize) this.history.shift();
      this.trail.push([this.hiddenBox[0], this.hiddenBox[1]]);
      if (this.trail.length > 36) this.trail.shift();

      return {
        cnnBox: chosen.bbox,
        rnnBox: this._currentTlwh(),
        label: this.label,
        conf: this.conf,
        state: "LOCKED",
      };
    }

    if (this.hiddenBox && this.missed < this.maxMissedFrames) {
      this.missed += 1;
      const damp = 0.82;
      this.velocity = this.velocity.map((v) => v * damp);
      this.hiddenBox = [
        this.hiddenBox[0] + this.velocity[0],
        this.hiddenBox[1] + this.velocity[1],
        this.hiddenBox[2],
        this.hiddenBox[3],
      ];
      this.trail.push([this.hiddenBox[0], this.hiddenBox[1]]);
      if (this.trail.length > 36) this.trail.shift();
      return {
        cnnBox: null,
        rnnBox: this._currentTlwh(),
        label: this.label,
        conf: this.conf * Math.pow(0.92, this.missed),
        state: "OCCLUSION",
      };
    }

    this.reset();
    return { cnnBox: null, rnnBox: null, label: null, conf: 0, state: "SEARCHING" };
  }

  _currentTlwh() {
    if (!this.hiddenBox) return null;
    const [cx, cy, w, h] = this.hiddenBox;
    return [cx - w / 2, cy - h / 2, w, h];
  }
}

/**
 * Class-Agnostic Visual Appearance + Recurrent Motion Tracker.
 * Tracks ANY user-boxed object (pen, keys, watch, glasses, etc.) using
 * downsampled zero-mean normalized RGB+edge patch correlation + velocity prior.
 */
class CustomVisualRNNTracker {
  constructor() {
    this.workCanvas = document.createElement("canvas");
    this.workCtx = this.workCanvas.getContext("2d", { willReadFrequently: true });
    this.cropCanvas = document.createElement("canvas");
    this.cropCtx = this.cropCanvas.getContext("2d");
    this.workWidth = 320;
    this.reset();
  }

  reset() {
    this.active = false;
    this.template = null;
    this.tplW = 0;
    this.tplH = 0;
    this.scale = 1;
    this.box = null; // [x, y, w, h] in full video coordinates
    this.velocity = [0, 0];
    this.label = "Custom Object";
    this.classConf = 0;
    this.matchScore = 1;
    this.missed = 0;
    this.frameCount = 0;
    this.trail = [];
  }

  _grabWorkFrame(videoEl) {
    const aspect = videoEl.videoHeight / videoEl.videoWidth;
    const w = this.workWidth;
    const h = Math.max(1, Math.round(w * aspect));
    if (this.workCanvas.width !== w || this.workCanvas.height !== h) {
      this.workCanvas.width = w;
      this.workCanvas.height = h;
    }
    this.scale = w / videoEl.videoWidth;
    this.workCtx.drawImage(videoEl, 0, 0, w, h);
    return this.workCtx.getImageData(0, 0, w, h);
  }

  _extractPatch(imgData, sx, sy, pw, ph) {
    const { data, width, height } = imgData;
    const patch = new Float32Array(pw * ph * 3);
    let meanR = 0, meanG = 0, meanB = 0;
    const count = pw * ph;

    let idx = 0;
    for (let dy = 0; dy < ph; dy++) {
      const py = Math.min(height - 1, Math.max(0, sy + dy));
      for (let dx = 0; dx < pw; dx++) {
        const px = Math.min(width - 1, Math.max(0, sx + dx));
        const off = (py * width + px) * 4;
        const r = data[off];
        const g = data[off + 1];
        const b = data[off + 2];
        patch[idx++] = r;
        patch[idx++] = g;
        patch[idx++] = b;
        meanR += r;
        meanG += g;
        meanB += b;
      }
    }
    meanR /= count;
    meanG /= count;
    meanB /= count;

    let normSq = 0;
    for (let i = 0; i < patch.length; i += 3) {
      patch[i] -= meanR;
      patch[i + 1] -= meanG;
      patch[i + 2] -= meanB;
      normSq += patch[i] * patch[i] + patch[i + 1] * patch[i + 1] + patch[i + 2] * patch[i + 2];
    }
    const norm = Math.sqrt(normSq) + 1e-6;
    for (let i = 0; i < patch.length; i++) {
      patch[i] /= norm;
    }
    return patch;
  }

  _correlate(imgData, sx, sy, pw, ph, template) {
    const { data, width, height } = imgData;
    const count = pw * ph;
    let sumR = 0, sumG = 0, sumB = 0;
    let dot = 0, normSq = 0;

    // 1-pass using subsampled stride=2 when patch is large for speed
    const step = count > 900 ? 2 : 1;
    let n = 0;
    for (let dy = 0; dy < ph; dy += step) {
      const py = Math.min(height - 1, Math.max(0, sy + dy));
      for (let dx = 0; dx < pw; dx += step) {
        const px = Math.min(width - 1, Math.max(0, sx + dx));
        const off = (py * width + px) * 4;
        sumR += data[off];
        sumG += data[off + 1];
        sumB += data[off + 2];
        n++;
      }
    }
    const meanR = sumR / n;
    const meanG = sumG / n;
    const meanB = sumB / n;

    for (let dy = 0; dy < ph; dy += step) {
      const py = Math.min(height - 1, Math.max(0, sy + dy));
      const rowOff = dy * pw;
      for (let dx = 0; dx < pw; dx += step) {
        const px = Math.min(width - 1, Math.max(0, sx + dx));
        const off = (py * width + px) * 4;
        const r = data[off] - meanR;
        const g = data[off + 1] - meanG;
        const b = data[off + 2] - meanB;
        const tIdx = (rowOff + dx) * 3;
        dot += r * template[tIdx] + g * template[tIdx + 1] + b * template[tIdx + 2];
        normSq += r * r + g * g + b * b;
      }
    }
    if (normSq < 1e-3) return -1;
    // Correct for step subsampling on template norm
    const scaleFactor = step === 2 ? 2.0 : 1.0;
    return (dot * scaleFactor) / (Math.sqrt(normSq) + 1e-6);
  }

  init(videoEl, boxFull) {
    const imgData = this._grabWorkFrame(videoEl);
    const [x, y, w, h] = boxFull;
    const sx = Math.max(0, Math.round(x * this.scale));
    const sy = Math.max(0, Math.round(y * this.scale));
    const pw = Math.max(8, Math.min(imgData.width - sx, Math.round(w * this.scale)));
    const ph = Math.max(8, Math.min(imgData.height - sy, Math.round(h * this.scale)));

    this.tplW = pw;
    this.tplH = ph;
    this.template = this._extractPatch(imgData, sx, sy, pw, ph);
    this.box = [sx / this.scale, sy / this.scale, pw / this.scale, ph / this.scale];
    this.velocity = [0, 0];
    this.active = true;
    this.missed = 0;
    this.frameCount = 0;
    this.matchScore = 1.0;
    this.label = "Identifying…";
    this.classConf = 0;
    this.trail = [[this.box[0] + this.box[2] / 2, this.box[1] + this.box[3] / 2]];
  }

  getCropCanvas(videoEl) {
    if (!this.box) return null;
    const [x, y, w, h] = this.box;
    // Pad crop slightly (25%) for better MobileNet context
    const padX = w * 0.25;
    const padY = h * 0.25;
    const cx1 = Math.max(0, Math.floor(x - padX));
    const cy1 = Math.max(0, Math.floor(y - padY));
    const cx2 = Math.min(videoEl.videoWidth, Math.ceil(x + w + padX));
    const cy2 = Math.min(videoEl.videoHeight, Math.ceil(y + h + padY));
    const cw = Math.max(16, cx2 - cx1);
    const ch = Math.max(16, cy2 - cy1);

    this.cropCanvas.width = 224;
    this.cropCanvas.height = 224;
    // Draw centered on square canvas preserving aspect ratio
    this.cropCtx.fillStyle = "#111";
    this.cropCtx.fillRect(0, 0, 224, 224);
    const ratio = Math.min(224 / cw, 224 / ch);
    const dw = cw * ratio;
    const dh = ch * ratio;
    this.cropCtx.drawImage(videoEl, cx1, cy1, cw, ch, (224 - dw) / 2, (224 - dh) / 2, dw, dh);
    return this.cropCanvas;
  }

  update(videoEl) {
    if (!this.active || !this.template) {
      return { cnnBox: null, rnnBox: null, label: null, conf: 0, state: "DRAW BOX" };
    }

    this.frameCount++;
    const imgData = this._grabWorkFrame(videoEl);
    const maxX = Math.max(0, imgData.width - this.tplW);
    const maxY = Math.max(0, imgData.height - this.tplH);

    // RNN velocity-predicted center in work-canvas coordinates
    const predX = Math.min(maxX, Math.max(0, Math.round((this.box[0] + this.velocity[0]) * this.scale)));
    const predY = Math.min(maxY, Math.max(0, Math.round((this.box[1] + this.velocity[1]) * this.scale)));

    const searchRadius = this.missed > 0 ? 48 : 34;
    const xStart = Math.max(0, predX - searchRadius);
    const xEnd = Math.min(maxX, predX + searchRadius);
    const yStart = Math.max(0, predY - searchRadius);
    const yEnd = Math.min(maxY, predY + searchRadius);

    // Coarse search (step 3)
    let bestScore = -2;
    let bestX = predX;
    let bestY = predY;
    for (let sy = yStart; sy <= yEnd; sy += 3) {
      for (let sx = xStart; sx <= xEnd; sx += 3) {
        const sc = this._correlate(imgData, sx, sy, this.tplW, this.tplH, this.template);
        if (sc > bestScore) {
          bestScore = sc;
          bestX = sx;
          bestY = sy;
        }
      }
    }

    // Fine search (step 1 around coarse winner)
    const fxStart = Math.max(0, bestX - 3);
    const fxEnd = Math.min(maxX, bestX + 3);
    const fyStart = Math.max(0, bestY - 3);
    const fyEnd = Math.min(maxY, bestY + 3);
    for (let sy = fyStart; sy <= fyEnd; sy++) {
      for (let sx = fxStart; sx <= fxEnd; sx++) {
        const sc = this._correlate(imgData, sx, sy, this.tplW, this.tplH, this.template);
        if (sc > bestScore) {
          bestScore = sc;
          bestX = sx;
          bestY = sy;
        }
      }
    }

    this.matchScore = Math.max(0, Math.min(1, bestScore));

    if (bestScore >= 0.42) {
      const obsX = bestX / this.scale;
      const obsY = bestY / this.scale;
      const prevX = this.box[0];
      const prevY = this.box[1];

      // Recurrent smoothing
      const nextX = prevX + this.velocity[0] + 0.65 * (obsX - (prevX + this.velocity[0]));
      const nextY = prevY + this.velocity[1] + 0.65 * (obsY - (prevY + this.velocity[1]));
      this.velocity = [
        0.55 * this.velocity[0] + 0.45 * (nextX - prevX),
        0.55 * this.velocity[1] + 0.45 * (nextY - prevY),
      ];
      this.box = [nextX, nextY, this.box[2], this.box[3]];
      this.missed = 0;

      // Gentle template adaptation only on very confident matches
      if (bestScore >= 0.72) {
        const fresh = this._extractPatch(imgData, bestX, bestY, this.tplW, this.tplH);
        const lr = 0.05;
        let normSq = 0;
        for (let i = 0; i < this.template.length; i++) {
          this.template[i] = (1 - lr) * this.template[i] + lr * fresh[i];
          normSq += this.template[i] * this.template[i];
        }
        const norm = Math.sqrt(normSq) + 1e-6;
        for (let i = 0; i < this.template.length; i++) this.template[i] /= norm;
      }

      this.trail.push([this.box[0] + this.box[2] / 2, this.box[1] + this.box[3] / 2]);
      if (this.trail.length > 40) this.trail.shift();

      return {
        cnnBox: [obsX, obsY, this.box[2], this.box[3]],
        rnnBox: [...this.box],
        label: this.label,
        conf: Math.max(this.classConf, this.matchScore),
        state: "LOCKED",
      };
    }

    // Occlusion prediction
    this.missed++;
    this.velocity = [this.velocity[0] * 0.78, this.velocity[1] * 0.78];
    this.box = [
      Math.max(0, Math.min(videoEl.videoWidth - this.box[2], this.box[0] + this.velocity[0])),
      Math.max(0, Math.min(videoEl.videoHeight - this.box[3], this.box[1] + this.velocity[1])),
      this.box[2],
      this.box[3],
    ];
    this.trail.push([this.box[0] + this.box[2] / 2, this.box[1] + this.box[3] / 2]);
    if (this.trail.length > 40) this.trail.shift();

    return {
      cnnBox: null,
      rnnBox: [...this.box],
      label: this.label,
      conf: this.matchScore,
      state: "OCCLUSION",
    };
  }
}

// DOM Elements
const video = document.getElementById("webcamVideo");
const canvas = document.getElementById("overlayCanvas");
const ctx = canvas.getContext("2d");
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

const statusDot = document.getElementById("statusDot");
const engineBadge = document.getElementById("engineBadge");
const fpsBadge = document.getElementById("fpsBadge");
const stagePlaceholder = document.getElementById("stagePlaceholder");
const hudTop = document.getElementById("hudTop");
const hudTargetChip = document.getElementById("hudTargetChip");
const resolutionText = document.getElementById("resolutionText");
const engineNote = document.getElementById("engineNote");
const sourceTag = document.getElementById("sourceTag");
const modeTag = document.getElementById("modeTag");
const modeSwitch = document.getElementById("modeSwitch");
const classFilterWrap = document.getElementById("classFilterWrap");

const startCamBtn = document.getElementById("startCamBtn");
const heroStartBtn = document.getElementById("heroStartBtn");
const stopCamBtn = document.getElementById("stopCamBtn");
const videoUpload = document.getElementById("videoUpload");
const resetTrackBtn = document.getElementById("resetTrackBtn");
const snapshotBtn = document.getElementById("snapshotBtn");
const classChips = document.getElementById("classChips");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");
const engineSelect = document.getElementById("engineSelect");
const mirrorToggle = document.getElementById("mirrorToggle");
const trailToggle = document.getElementById("trailToggle");

const metricLabel = document.getElementById("metricLabel");
const metricConf = document.getElementById("metricConf");
const metricState = document.getElementById("metricState");
const metricCount = document.getElementById("metricCount");
const metricBox = document.getElementById("metricBox");
const confBarFill = document.getElementById("confBarFill");
const detectedList = document.getElementById("detectedList");

// State
let mediaStream = null;
let uploadedUrl = null;
let isWebcam = false;
let running = false;
let busy = false;
let cocoModel = null;
let mobileNetModel = null;
let classifyingCrop = false;
let engineMode = "browser"; // "browser" | "python"
let trackingMode = "auto";  // "auto" | "custom"
let selectedClass = "all";
let pyTrail = [];
let lastFrameTime = performance.now();
let smoothFps = 0;

// Drag-to-select state (video pixel coordinates)
let isDragging = false;
let dragStart = null;
let dragCurrent = null;

const tracker = new RecurrentTemporalTracker(6, 14);
const customTracker = new CustomVisualRNNTracker();

function setEngineBadgeText(text) {
  engineBadge.innerHTML = `<span class="pill-dot"></span><span>${text}</span>`;
}

function isMirrored() {
  return isWebcam && mirrorToggle.checked;
}

function applyMirrorClass() {
  video.classList.toggle("mirrored", isMirrored());
}

function mapX(x, w = 0) {
  return isMirrored() ? canvas.width - x - w : x;
}

function setTrackingMode(mode) {
  trackingMode = mode;
  modeSwitch.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.mode === mode);
  });
  if (mode === "custom") {
    modeTag.textContent = "1000-Class + Visual";
    classFilterWrap.style.opacity = "0.45";
    classFilterWrap.style.pointerEvents = "none";
  } else {
    modeTag.textContent = "Auto + Drag";
    classFilterWrap.style.opacity = "1";
    classFilterWrap.style.pointerEvents = "auto";
    customTracker.reset();
  }
}

async function loadBrowserModels() {
  if (!cocoModel) {
    setEngineBadgeText("Loading CNN Model…");
    if (typeof tf === "undefined" || typeof cocoSsd === "undefined") {
      for (let i = 0; i < 15; i++) {
        if (typeof tf !== "undefined" && typeof cocoSsd !== "undefined") break;
        await new Promise((r) => setTimeout(r, 200));
      }
    }
    if (typeof tf === "undefined" || typeof cocoSsd === "undefined") {
      setEngineBadgeText("Model Load Error");
      throw new Error(
        "TensorFlow.js or COCO-SSD model scripts could not be loaded from CDN. " +
        "Please check your internet connection, or switch to 'Python Backend' in the dropdown."
      );
    }
    try {
      await tf.ready();
    } catch {
      await tf.setBackend("cpu").catch(() => {});
    }
    try {
      cocoModel = await cocoSsd.load({ base: "mobilenet_v2" });
    } catch (err) {
      console.warn("Retrying with lite_mobilenet_v2 fallback:", err);
      cocoModel = await cocoSsd.load({ base: "lite_mobilenet_v2" });
    }
  }
  setEngineBadgeText(`Browser CNN+RNN (${tf.getBackend().toUpperCase()})`);
  return cocoModel;
}

async function ensureMobileNet() {
  if (mobileNetModel) return mobileNetModel;
  await tf.ready();
  mobileNetModel = await mobilenet.load({ version: 2, alpha: 1.0 });
  return mobileNetModel;
}

async function classifyCustomTarget() {
  if (classifyingCrop || !customTracker.active) return;
  classifyingCrop = true;
  try {
    const clf = await ensureMobileNet();
    const crop = customTracker.getCropCanvas(video);
    if (!crop) return;
    const preds = await clf.classify(crop, 3);
    if (preds && preds.length > 0 && customTracker.active) {
      // Prefer pen/office object if among top-3 predictions
      const preferred =
        preds.find((p) => /pen|pencil|marker|ruler|eraser|watch|glasses|key|bottle|cup|phone/i.test(p.className)) ||
        preds[0];
      customTracker.label = formatImageNetLabel(preferred.className);
      customTracker.classConf = preferred.probability;
    }
  } catch (err) {
    console.warn("MobileNet classify error:", err);
  } finally {
    classifyingCrop = false;
  }
}

async function initServerStatus() {
  try {
    const res = await fetch("/api/status");
    const st = await res.json();
    const memTxt = st.avail_commit_gb !== null ? `${st.avail_commit_gb} GiB free RAM` : "";
    const wtTxt = st.has_trained_weights ? "Trained LSTM ready" : "Untrained LSTM";
    if (engineNote) {
      engineNote.textContent = `Browser WebGL GPU active. Python backend: ${wtTxt}${memTxt ? ` · ${memTxt}` : ""}.`;
    }
  } catch {
    if (engineNote) {
      engineNote.textContent = "Browser WebGL CNN+RNN ready.";
    }
  }
}

async function switchEngine(mode) {
  engineMode = mode;
  tracker.reset();
  customTracker.reset();
  pyTrail = [];
  if (mode === "python") {
    setTrackingMode("auto");
    setEngineBadgeText("Loading Python YOLOv8+LSTM…");
    if (engineNote) {
      engineNote.textContent = "Loading YOLOv8 CNN + PyTorch LSTM on server…";
    }
    try {
      const res = await fetch("/api/backend/load", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: false }),
      });
      const st = await res.json();
      if (st.ready) {
        const lstmTag = st.using_trained_lstm ? "Trained LSTM" : "YOLOv8+LSTM";
        setEngineBadgeText(`Python ${lstmTag} (${st.device})`);
        if (engineNote) {
          engineNote.textContent = `Python backend active on ${st.device}. ${
            st.using_trained_lstm
              ? "Loaded weights/tracker_model.pt."
              : "Train weights/tracker_model.pt to enable trained LSTM predictions."
          }`;
        }
      } else {
        engineSelect.value = "browser";
        engineMode = "browser";
        if (engineNote) {
          engineNote.textContent = st.error || "Python backend unavailable; staying on Browser WebGL.";
        }
        await loadBrowserModels();
      }
    } catch (err) {
      engineSelect.value = "browser";
      engineMode = "browser";
      if (engineNote) {
        engineNote.textContent = `Python backend error: ${err.message}.`;
      }
    }
  } else {
    await loadBrowserModels();
    initServerStatus();
  }
}

function syncCanvasSize() {
  if (video.videoWidth > 0 && video.videoHeight > 0) {
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      resolutionText.textContent = `${video.videoWidth} × ${video.videoHeight}`;
    }
  }
}

function drawViewfinderCorners(x, y, w, h, color) {
  const len = Math.max(12, Math.min(34, Math.min(w, h) * 0.25));
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3.5;
  ctx.lineCap = "round";
  ctx.shadowColor = color;
  ctx.shadowBlur = 10;

  ctx.beginPath();
  ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y);
  ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
  ctx.moveTo(x + w, y + h - len); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - len, y + h);
  ctx.moveTo(x + len, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - len);
  ctx.stroke();
  ctx.restore();
}

function drawOverlay(allDetections, result, trail) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Secondary detections
  for (const det of allDetections) {
    const [rx, ry, rw, rh] = det.bbox;
    const x = mapX(rx, rw);
    ctx.save();
    ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([6, 5]);
    ctx.strokeRect(x, ry, rw, rh);
    ctx.restore();

    ctx.fillStyle = "rgba(56, 189, 248, 0.9)";
    ctx.font = "600 13px Inter, system-ui, sans-serif";
    ctx.fillText(`${det.class} ${Math.round(det.score * 100)}%`, x + 6, Math.max(54, ry - 6));
  }

  // Smooth gradient motion trail
  if (trailToggle.checked && trail.length > 1) {
    for (let i = 1; i < trail.length; i++) {
      const ratio = i / trail.length;
      ctx.beginPath();
      ctx.moveTo(mapX(trail[i - 1][0]), trail[i - 1][1]);
      ctx.lineTo(mapX(trail[i][0]), trail[i][1]);
      ctx.strokeStyle = `rgba(245, 158, 11, ${0.15 + 0.8 * ratio})`;
      ctx.lineWidth = 1.5 + 2.5 * ratio;
      ctx.lineCap = "round";
      ctx.stroke();
    }
  }

  // Primary CNN / observation box
  if (result.cnnBox) {
    const [rx, ry, rw, rh] = result.cnnBox;
    const x = mapX(rx, rw);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.85)";
    ctx.lineWidth = 1.8;
    ctx.strokeRect(x, ry, rw, rh);
  }

  // Primary RNN tracked box
  if (result.rnnBox) {
    const [rx, ry, rw, rh] = result.rnnBox;
    const x = mapX(rx, rw);
    const isOcc = result.state === "OCCLUSION";
    const color = isOcc ? "#f59e0b" : "#22c55e";

    ctx.fillStyle = isOcc ? "rgba(245, 158, 11, 0.07)" : "rgba(34, 197, 94, 0.07)";
    ctx.fillRect(x, ry, rw, rh);
    ctx.strokeStyle = isOcc ? "rgba(245, 158, 11, 0.45)" : "rgba(34, 197, 94, 0.45)";
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, ry, rw, rh);

    drawViewfinderCorners(x, ry, rw, rh, color);

    const cx = x + rw / 2;
    const cy = ry + rh / 2;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(cx - 8, cy); ctx.lineTo(cx + 8, cy);
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx, cy + 8);
    ctx.stroke();

    const tag = `${(result.label || "target").toUpperCase()}  ${Math.round((result.conf || 0) * 100)}%  ·  ${result.state}`;
    ctx.font = "700 13px Inter, system-ui, sans-serif";
    const textW = ctx.measureText(tag).width;
    const pillW = textW + 18;
    const pillH = 26;
    const pillX = Math.max(8, Math.min(x, canvas.width - pillW - 8));
    // Avoid colliding with the top-left floating HUD bar
    let pillY = ry > 60 ? ry - pillH - 6 : ry + 10;
    if (pillX < 460 && pillY < 52) pillY = 52;

    ctx.fillStyle = isOcc ? "rgba(245, 158, 11, 0.94)" : "rgba(34, 197, 94, 0.94)";
    ctx.beginPath();
    ctx.roundRect(pillX, pillY, pillW, pillH, 6);
    ctx.fill();

    ctx.fillStyle = "#04130a";
    ctx.fillText(tag, pillX + 9, pillY + 17);
  }

  // Active user drag rectangle
  if (isDragging && dragStart && dragCurrent) {
    const x1 = Math.min(dragStart.x, dragCurrent.x);
    const y1 = Math.min(dragStart.y, dragCurrent.y);
    const w = Math.abs(dragCurrent.x - dragStart.x);
    const h = Math.abs(dragCurrent.y - dragStart.y);
    const dx = mapX(x1, w);

    ctx.save();
    ctx.fillStyle = "rgba(59, 130, 246, 0.16)";
    ctx.fillRect(dx, y1, w, h);
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.5;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(dx, y1, w, h);
    ctx.restore();
  }
}

function renderDetectedPills(detections, trackedLabel) {
  if (trackingMode === "custom") {
    if (customTracker.active) {
      detectedList.innerHTML = `<span class="det-pill tracked">${customTracker.label} (Custom Lock)</span>`;
    } else {
      detectedList.innerHTML = `<span class="empty-msg">Drag a box around your pen or any object on the video</span>`;
    }
    return;
  }

  if (!detections.length) {
    detectedList.innerHTML = `<span class="empty-msg">No COCO objects detected — drag a box around any object to track it</span>`;
    return;
  }
  detectedList.innerHTML = "";
  detections.forEach((det, idx) => {
    const btn = document.createElement("button");
    btn.className = `det-pill${det.class === trackedLabel && idx === 0 ? " tracked" : ""}`;
    btn.textContent = `${det.class} ${Math.round(det.score * 100)}%`;
    btn.onclick = () => tracker.lockDetection(det);
    detectedList.appendChild(btn);
  });
}

function updateTelemetry(result, detections) {
  const confPct = Math.round((result.conf || 0) * 100);
  metricLabel.textContent = result.label ? result.label.toUpperCase() : "No Target";
  metricConf.textContent = result.conf ? `${confPct}%` : "0%";
  confBarFill.style.width = `${confPct}%`;

  metricState.textContent = result.state;
  if (result.state === "LOCKED") {
    metricState.style.color = "#4ade80";
    metricState.style.borderColor = "rgba(34, 197, 94, 0.4)";
    hudTargetChip.textContent = `Locked: ${(result.label || "").toUpperCase()} (${confPct}%)`;
  } else if (result.state === "OCCLUSION") {
    metricState.style.color = "#fbbf24";
    metricState.style.borderColor = "rgba(245, 158, 11, 0.45)";
    hudTargetChip.textContent = "Occlusion · RNN Predicting";
  } else if (result.state === "DRAW BOX") {
    metricState.style.color = "#38bdf8";
    metricState.style.borderColor = "rgba(56, 189, 248, 0.4)";
    hudTargetChip.textContent = "Drag box around any object…";
  } else {
    metricState.style.color = "#94a3b8";
    metricState.style.borderColor = "rgba(148, 163, 184, 0.2)";
    hudTargetChip.textContent = "Searching…";
  }

  metricCount.textContent = String(detections.length);
  if (result.rnnBox) {
    const [x, y, w, h] = result.rnnBox.map((v) => Math.round(v));
    metricBox.textContent = `${x}, ${y}, ${w}×${h}`;
  } else {
    metricBox.textContent = "—";
  }

  renderDetectedPills(detections, result.label);
}

async function processFrameLoop() {
  if (!running) return;

  if (!busy && video.readyState >= 2 && !video.paused && !video.ended) {
    busy = true;
    try {
      syncCanvasSize();
      const now = performance.now();
      const dt = Math.max(1, now - lastFrameTime);
      lastFrameTime = now;
      smoothFps = smoothFps * 0.82 + (1000 / dt) * 0.18;
      fpsBadge.textContent = `${smoothFps.toFixed(1)} FPS`;

      if (trackingMode === "custom") {
        const result = customTracker.update(video);
        if (customTracker.active && (customTracker.frameCount === 1 || customTracker.frameCount % 30 === 0)) {
          classifyCustomTarget();
        }
        drawOverlay([], result, customTracker.trail);
        updateTelemetry(result, customTracker.active ? [result] : []);
      } else if (engineMode === "browser") {
        const model = await loadBrowserModels();
        const minScore = Number(confSlider.value) / 100;
        const rawDets = await model.detect(video, 12, minScore);
        const filtered =
          selectedClass === "all"
            ? rawDets
            : rawDets.filter((d) => d.class === selectedClass);

        const result = tracker.update(filtered);
        drawOverlay(filtered, result, tracker.trail);
        updateTelemetry(result, filtered);
      } else {
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        captureCtx.drawImage(video, 0, 0);
        const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.72);
        const targetClassId =
          selectedClass === "all" ? null : COCO_NAME_TO_ID[selectedClass] ?? null;

        const res = await fetch("/api/track", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image: dataUrl, target_class: targetClassId }),
        });
        const data = await res.json();
        if (!data.error) {
          if (data.rnn_box) {
            pyTrail.push([
              data.rnn_box[0] + data.rnn_box[2] / 2,
              data.rnn_box[1] + data.rnn_box[3] / 2,
            ]);
            if (pyTrail.length > 36) pyTrail.shift();
          }
          const result = {
            cnnBox: data.cnn_box,
            rnnBox: data.rnn_box,
            label: data.label,
            conf: data.conf,
            state: data.occluded ? "OCCLUSION" : data.rnn_box ? "LOCKED" : "SEARCHING",
          };
          const synthDet = data.cnn_box
            ? [{ class: data.label, score: data.conf, bbox: data.cnn_box }]
            : [];
          drawOverlay([], result, pyTrail);
          updateTelemetry(result, synthDet);
        }
      }
    } catch (err) {
      console.error(err);
    } finally {
      busy = false;
    }
  }

  requestAnimationFrame(processFrameLoop);
}

function clientToVideoCoords(clientX, clientY) {
  const rect = canvas.getBoundingClientRect();
  const elemAspect = rect.width / rect.height;
  const vidAspect = canvas.width / canvas.height;

  let drawW = rect.width;
  let drawH = rect.height;
  let offsetX = 0;
  let offsetY = 0;

  if (elemAspect > vidAspect) {
    drawW = rect.height * vidAspect;
    offsetX = (rect.width - drawW) / 2;
  } else {
    drawH = rect.width / vidAspect;
    offsetY = (rect.height - drawH) / 2;
  }

  const relX = Math.max(0, Math.min(1, (clientX - rect.left - offsetX) / drawW));
  const relY = Math.max(0, Math.min(1, (clientY - rect.top - offsetY) / drawH));
  return {
    x: (isMirrored() ? 1 - relX : relX) * canvas.width,
    y: relY * canvas.height,
  };
}

// Mouse drag (draw box on ANY object like a pen) OR single click (lock onto detected COCO box)
canvas.addEventListener("pointerdown", (ev) => {
  if (!running || !canvas.width || !canvas.height) return;
  canvas.setPointerCapture(ev.pointerId);
  isDragging = true;
  dragStart = clientToVideoCoords(ev.clientX, ev.clientY);
  dragCurrent = { ...dragStart };
});

canvas.addEventListener("pointermove", (ev) => {
  if (!isDragging) return;
  dragCurrent = clientToVideoCoords(ev.clientX, ev.clientY);
});

canvas.addEventListener("pointerup", (ev) => {
  if (!isDragging || !dragStart) return;
  isDragging = false;
  dragCurrent = clientToVideoCoords(ev.clientX, ev.clientY);

  const x1 = Math.min(dragStart.x, dragCurrent.x);
  const y1 = Math.min(dragStart.y, dragCurrent.y);
  const w = Math.abs(dragCurrent.x - dragStart.x);
  const h = Math.abs(dragCurrent.y - dragStart.y);

  dragStart = null;
  dragCurrent = null;

  if (w >= 20 && h >= 20) {
    // User dragged a box -> track & classify ANY object (pen, keys, watch, etc.)!
    setTrackingMode("custom");
    customTracker.init(video, [x1, y1, w, h]);
    classifyCustomTarget();
  } else if (trackingMode === "auto") {
    // Simple click -> lock onto detected COCO object
    tracker.lockPoint(x1, y1);
  }
});

async function startCamera() {
  stopCamera();
  try {
    stagePlaceholder.classList.add("hidden");
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      audio: false,
    });
    isWebcam = true;
    applyMirrorClass();
    video.srcObject = mediaStream;
    video.loop = false;
    await video.play();
    sourceTag.textContent = "Webcam";
    onStreamStarted();
  } catch (err) {
    stagePlaceholder.classList.remove("hidden");
    alert(`Could not access webcam: ${err.message}\nPlease allow camera permission in your browser.`);
  }
}

function startUploadedVideo(file) {
  stopCamera();
  isWebcam = false;
  applyMirrorClass();
  stagePlaceholder.classList.add("hidden");
  uploadedUrl = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = uploadedUrl;
  video.loop = true;
  sourceTag.textContent = "Video";
  video.play().then(onStreamStarted);
}

async function onStreamStarted() {
  running = true;
  statusDot.classList.add("live");
  hudTop.classList.remove("hidden");
  startCamBtn.disabled = true;
  stopCamBtn.disabled = false;
  snapshotBtn.disabled = false;
  tracker.reset();
  customTracker.reset();
  pyTrail = [];
  if (engineMode === "browser") {
    await loadBrowserModels();
    // Preload 1000-class MobileNet silently in background for instant pen/object classification
    ensureMobileNet().catch(() => {});
  }
  lastFrameTime = performance.now();
  requestAnimationFrame(processFrameLoop);
}

function stopCamera() {
  running = false;
  isWebcam = false;
  applyMirrorClass();
  statusDot.classList.remove("live");
  hudTop.classList.add("hidden");
  startCamBtn.disabled = false;
  stopCamBtn.disabled = true;
  snapshotBtn.disabled = true;
  sourceTag.textContent = "Idle";

  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
  }
  if (uploadedUrl) {
    URL.revokeObjectURL(uploadedUrl);
    uploadedUrl = null;
  }
  video.pause();
  video.srcObject = null;
  video.removeAttribute("src");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  stagePlaceholder.classList.remove("hidden");
  fpsBadge.textContent = "0.0 FPS";
  metricState.textContent = "IDLE";
  metricLabel.textContent = "No Target";
  metricConf.textContent = "0%";
  confBarFill.style.width = "0%";
  metricBox.textContent = "—";
  metricCount.textContent = "0";
  detectedList.innerHTML = `<span class="empty-msg">Start camera to see live detections</span>`;
}

// Event Listeners
startCamBtn.addEventListener("click", startCamera);
heroStartBtn.addEventListener("click", startCamera);
stopCamBtn.addEventListener("click", stopCamera);
mirrorToggle.addEventListener("change", applyMirrorClass);

modeSwitch.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".mode-btn");
  if (!btn) return;
  setTrackingMode(btn.dataset.mode);
});

videoUpload.addEventListener("change", (ev) => {
  const file = ev.target.files?.[0];
  if (file) startUploadedVideo(file);
});

classChips.addEventListener("click", (ev) => {
  const btn = ev.target.closest(".chip");
  if (!btn) return;
  selectedClass = btn.dataset.class;
  classChips.querySelectorAll(".chip").forEach((el) => el.classList.remove("active"));
  btn.classList.add("active");
  tracker.reset();
});

resetTrackBtn.addEventListener("click", () => {
  tracker.reset();
  customTracker.reset();
  pyTrail = [];
  if (trackingMode === "custom") {
    setTrackingMode("auto");
  }
  fetch("/api/reset", { method: "POST" }).catch(() => {});
});

snapshotBtn.addEventListener("click", () => {
  if (!video.videoWidth) return;
  const snap = document.createElement("canvas");
  snap.width = video.videoWidth;
  snap.height = video.videoHeight;
  const sctx = snap.getContext("2d");
  if (isMirrored()) {
    sctx.save();
    sctx.translate(snap.width, 0);
    sctx.scale(-1, 1);
    sctx.drawImage(video, 0, 0, snap.width, snap.height);
    sctx.restore();
  } else {
    sctx.drawImage(video, 0, 0, snap.width, snap.height);
  }
  sctx.drawImage(canvas, 0, 0, snap.width, snap.height);
  const a = document.createElement("a");
  a.href = snap.toDataURL("image/png");
  a.download = `tracked-frame-${Date.now()}.png`;
  a.click();
});

confSlider.addEventListener("input", () => {
  confValue.textContent = `${confSlider.value}%`;
});

engineSelect.addEventListener("change", () => {
  switchEngine(engineSelect.value);
});

window.addEventListener("DOMContentLoaded", () => {
  initServerStatus();
  loadBrowserModels().catch((err) => {
    setEngineBadgeText("Model Load Failed");
    if (engineNote) {
      engineNote.textContent = `Could not load TensorFlow.js model: ${err.message}`;
    }
  });
});
