"""YOLO (v8) face detector. Auto-downloads weights if missing."""
import logging
import os
import urllib.request
from typing import List

from .logging_setup import ev
from .tracker import Detection


def ensure_weights(path: str, url: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ev("WEIGHTS_DOWNLOAD", url=url, dest=path)
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download YOLO face weights ({exc}). Download manually from {url} "
            f"and save as {path}") from exc
    return path


class FaceDetector:
    def __init__(self, cfg):
        from ultralytics import YOLO  # lazy import keeps unit tests light
        self.cfg = cfg
        weights = ensure_weights(cfg.weights_path, cfg.weights_url)
        self.model = YOLO(weights)
        device = cfg.device
        if device == "auto":
            try:
                import torch
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device = device
        ev("DETECTOR_READY", weights=weights, device=device)

    def detect(self, frame) -> List[Detection]:
        res = self.model.predict(frame, imgsz=self.cfg.image_size, conf=self.cfg.confidence,
                                 iou=self.cfg.nms_iou, device=self.device, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []
        xyxy = res.boxes.xyxy.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        out = []
        for b, c in zip(xyxy, conf):
            if min(b[2] - b[0], b[3] - b[1]) < self.cfg.min_face_size_px:
                continue  # too small to recognise reliably -> ignore
            out.append(Detection(b.astype(float), float(c)))
        return out
