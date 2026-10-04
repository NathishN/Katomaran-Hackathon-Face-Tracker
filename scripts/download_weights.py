"""Pre-download YOLOv8 face weights + InsightFace buffalo_l so the demo never stalls on network."""
import json
from src.detector import ensure_weights

cfg = json.load(open("config.json"))["detector"]
print("YOLO weights:", ensure_weights(cfg["weights_path"], cfg["weights_url"]))
from insightface.app import FaceAnalysis
FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
             providers=["CPUExecutionProvider"]).prepare(ctx_id=-1, det_size=(320, 320))
print("InsightFace buffalo_l ready")
