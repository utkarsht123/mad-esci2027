"""Shared definitions for the controlled RGB / Grayscale / Thermal / RGB-T study on LLVIP.

Every modality is derived from the *same* resized, co-registered image pair so the only
variable between experiments is what the detector is allowed to see.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

# Working resolution. LLVIP frames are 1280x1024; we store everything at 640x512, which is
# exactly what YOLO sees at imgsz=640, so no information the detector could use is lost.
WORK_W, WORK_H = 640, 512

# Modality -> (dataset folder name, number of input channels)
MODALITIES = {
    "rgb": ("rgb", 3),
    "gray": ("gray", 3),  # stored single-channel, read by YOLO as 3 identical channels
    "ir": ("ir", 3),  # stored single-channel, read by YOLO as 3 identical channels
    "rgbt": ("rgbt", 4),  # early fusion: 4-channel TIFF, channel order R,G,B,T
    "rgbt_unpaired": ("rgbt_unpaired", 4),  # thermal channel taken from a *different* frame
}
TRAINED_MODALITIES = list(MODALITIES)

TIFF_LZW = [cv2.IMWRITE_TIFF_COMPRESSION, 5]


def read_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def write_json(p: Path, obj) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2))


def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def luminance(bgr: np.ndarray) -> float:
    """Mean ITU-R BT.601 luma of a BGR uint8 image, in [0, 255]."""
    return float(to_gray(bgr).mean())


def make_input(modality: str, rgb_bgr: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Build the array a trained model of `modality` expects at prediction time.

    3-channel inputs are BGR (Ultralytics flips them to RGB internally, exactly as it does for
    training images read with OpenCV). 4-channel inputs are passed through unflipped by
    Ultralytics, so they are built in the stored R,G,B,T order.
    """
    if modality == "rgb":
        return rgb_bgr
    if modality == "gray":
        g = to_gray(rgb_bgr)
        return np.dstack([g, g, g])
    if modality == "ir":
        return np.dstack([ir, ir, ir])
    if modality in ("rgbt", "rgbt_unpaired"):
        return np.dstack([rgb_bgr[..., ::-1], ir])
    raise ValueError(modality)


def load_test_pair(ds_root: Path, stem: str) -> tuple[np.ndarray, np.ndarray]:
    rgb = cv2.imread(str(ds_root / "rgb" / "images" / "test" / f"{stem}.png"), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(ds_root / "ir" / "images" / "test" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or ir is None:
        raise FileNotFoundError(f"missing test pair for {stem} under {ds_root}")
    return rgb, ir


def read_yolo_labels(txt: Path, w: int = WORK_W, h: int = WORK_H) -> np.ndarray:
    """Return absolute xyxy boxes (N,4) from a YOLO label file."""
    if not txt.exists() or txt.stat().st_size == 0:
        return np.zeros((0, 4), np.float32)
    a = np.loadtxt(txt, ndmin=2)[:, 1:5]
    cx, cy, bw, bh = a[:, 0] * w, a[:, 1] * h, a[:, 2] * w, a[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1).astype(np.float32)
