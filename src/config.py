"""Configuration loader: merges config.json over safe defaults and exposes dot-access."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

DEFAULTS = {
    "video_source": "data/sample_video.mp4",
    "detection_skip_frames": 2,
    "detector": {
        "weights_path": "models/yolov8n-face.pt",
        "weights_url": "https://github.com/akanametov/yolo-face/releases/download/1.0.0/yolov8n-face.pt",
        "confidence": 0.5, "nms_iou": 0.45, "image_size": 640,
        "device": "auto", "min_face_size_px": 32,
    },
    "recognition": {
        "model_pack": "buffalo_l", "det_size": 320, "similarity_threshold": 0.40,
        "max_embeddings_per_identity": 5, "max_attempts_per_track": 8,
        "min_sharpness": 10.0, "gallery_refresh_every_n_cycles": 20,
        "crop_margin": 0.25, "use_gpu": "auto", "allow_unaligned_fallback": False,
    },
    "tracking": {"iou_threshold": 0.25, "min_hits": 2, "max_age_frames": 45,
                 "tentative_max_age_frames": 10},
    "storage": {"log_dir": "logs", "db_path": "data/visitors.db", "reset_on_start": False},
    "stream": {"reconnect_delay_sec": 3, "max_reconnect_attempts": 20},
    "output": {"show_window": False, "save_annotated_video": True,
               "annotated_path": "output/annotated.mp4",
               "tracking_log_interval_frames": 60, "log_level": "INFO"},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def load_config(path="config.json", overrides: dict = None) -> SimpleNamespace:
    data = {}
    p = Path(path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
    merged = _merge(DEFAULTS, data)
    if overrides:
        merged = _merge(merged, overrides)
    return _ns(merged)
