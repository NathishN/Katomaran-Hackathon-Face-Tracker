"""InsightFace (ArcFace w600k_r50, 512-d) embedding generator.

The YOLO box is padded and passed through InsightFace's SCRFD stage only to obtain 5 landmarks
for proper face alignment (this is what makes ArcFace accurate); embeddings are L2-normalised
so cosine similarity == dot product.
"""
import cv2
import numpy as np

from .logging_setup import ev


class FaceRecognizer:
    def __init__(self, cfg):
        from insightface.app import FaceAnalysis
        self.cfg = cfg
        use_gpu = False
        try:
            import onnxruntime as ort
            has_cuda = "CUDAExecutionProvider" in ort.get_available_providers()
            use_gpu = has_cuda if cfg.use_gpu == "auto" else (bool(cfg.use_gpu) and has_cuda)
        except Exception:
            pass
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"] if use_gpu
                     else ["CPUExecutionProvider"])
        self.app = FaceAnalysis(name=cfg.model_pack,
                                allowed_modules=["detection", "recognition"],
                                providers=providers)
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_thresh=0.3,
                         det_size=(cfg.det_size, cfg.det_size))
        ev("RECOGNIZER_READY", model=cfg.model_pack, gpu=use_gpu)

    def embed(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh, m = x2 - x1, y2 - y1, self.cfg.crop_margin
        cx1, cy1 = int(max(0, x1 - bw * m)), int(max(0, y1 - bh * m))
        cx2, cy2 = int(min(w, x2 + bw * m)), int(min(h, y2 + bh * m))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        short = min(crop.shape[:2])
        if short < 160:  # upscale small faces so the aligner/detector can see them
            s = 160.0 / short
            crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        faces = self.app.get(crop)
        if faces:
            f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return np.asarray(f.normed_embedding, dtype=np.float32)
        if self.cfg.allow_unaligned_fallback:
            rec = self.app.models["recognition"]
            e = rec.get_feat(cv2.resize(crop, (112, 112))).flatten().astype(np.float32)
            return e / (np.linalg.norm(e) + 1e-9)
        return None
