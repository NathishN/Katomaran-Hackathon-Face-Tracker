"""End-to-end logic test WITHOUT heavy models (fake detector + fake recogniser).

Scenario (video of 400 frames, 640x480):
  person A on the left : frames 0-120   and again 260-330 (re-entry)
  person B on the right: frames 60-200
  a 15-frame detector dropout for B (frames 100-115) -> must NOT create a new ID / extra events
Expected: unique=2, entries=3 (A, B, A-again), exits=3, all with images + DB rows.
Run:  python -m tests.test_pipeline_mock
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np

from src.config import load_config
from src.database import Database
from src.event_logger import EventLogger
from src.logging_setup import setup_logging
from src.pipeline import FacePipeline
from src.tracker import Detection

rng = np.random.default_rng(0)
VEC = {}
for name in "AB":
    v = rng.normal(size=512).astype(np.float32)
    VEC[name] = v / np.linalg.norm(v)


class FakeDetector:
    idx = 0
    def detect(self, frame):
        i, out = self.idx, []
        if 0 <= i <= 120 or 260 <= i <= 330:
            out.append(Detection(np.array([100.0 + i * 0.2, 150, 200 + i * 0.2, 270]), 0.9))
        if 60 <= i <= 200 and not (100 <= i <= 115):
            out.append(Detection(np.array([420.0, 150, 520, 270]), 0.9))
        return out


class FakeRecognizer:
    def embed(self, frame, bbox):
        who = "A" if (bbox[0] + bbox[2]) / 2 < 320 else "B"
        e = VEC[who] + rng.normal(scale=0.01, size=512).astype(np.float32)
        return e / np.linalg.norm(e)


def run():
    tmp = Path(tempfile.mkdtemp())
    cfg = load_config("config.json", overrides={
        "storage": {"log_dir": str(tmp / "logs"), "db_path": str(tmp / "t.db")},
        "recognition": {"min_sharpness": 0.0},
        "output": {"tracking_log_interval_frames": 50}})
    setup_logging(cfg.storage.log_dir, "INFO")
    db = Database(cfg.storage.db_path, reset=True)
    det = FakeDetector()
    pipe = FacePipeline(cfg, db, EventLogger(cfg.storage.log_dir, db), det, FakeRecognizer())
    for i in range(400):
        det.idx = i
        frame = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)  # textured => sharp
        pipe.process_frame(frame, i, i / 25)
    pipe.flush()

    counts, unique = db.event_counts(), db.unique_visitor_count()
    print("unique:", unique, "counts:", counts)
    assert unique == 2, f"expected 2 unique, got {unique}"
    assert counts == {"entry": 3, "exit": 3}, counts
    n_img = len(list((tmp / "logs").rglob("*.jpg")))
    assert n_img == 6, f"expected 6 images, got {n_img}"
    assert (tmp / "logs" / "events.log").stat().st_size > 0
    db.close()
    shutil.rmtree(tmp, ignore_errors=True)
    print("PASS")


if __name__ == "__main__":
    run()
