"""Lightweight multi-face tracker (IoU + constant-velocity prediction, Hungarian matching).

Designed for the *skip-frames* setting: detection runs every (N+1)-th frame, in between the
tracker extrapolates boxes with the last estimated velocity so IDs stay stable.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Detection:
    bbox: np.ndarray  # [x1, y1, x2, y2] float
    conf: float


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    conf: float
    first_frame: int
    last_seen_frame: int
    last_seen_ts: datetime
    last_seen_vts: Optional[float]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    display_bbox: Optional[np.ndarray] = None
    hits: int = 1
    # identity / evidence state
    face_id: Optional[int] = None
    entered: bool = False
    retired: bool = False       # merged into another track -> no exit event
    gave_up: bool = False       # embedding never succeeded
    embed_attempts: int = 0
    cycles_since_embed: int = 0
    best_crop: Optional[np.ndarray] = None
    best_score: float = -1.0
    last_crop: Optional[np.ndarray] = None
    cur_sharpness: float = 0.0

    def __post_init__(self):
        if self.display_bbox is None:
            self.display_bbox = self.bbox.copy()


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


class FaceTracker:
    MAX_EXTRAPOLATE = 10  # frames

    def __init__(self, iou_threshold=0.25, min_hits=2, max_age_frames=45,
                 tentative_max_age_frames=10, detect_interval=3):
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        # guard against configs where ageing is shorter than the detection cadence
        self.max_age = max(max_age_frames, 3 * detect_interval)
        self.tentative_max_age = max(tentative_max_age_frames, 2 * detect_interval + 1)
        self.tracks: List[Track] = []
        self._next_id = 1

    # ---- helpers -------------------------------------------------------
    def is_confirmed(self, t: Track) -> bool:
        return t.hits >= self.min_hits

    def _predict(self, t: Track, frame_idx: int) -> np.ndarray:
        dt = min(max(frame_idx - t.last_seen_frame, 0), self.MAX_EXTRAPOLATE)
        return t.bbox + t.velocity * dt

    def predict(self, frame_idx: int):
        """Called on frames where detection is skipped: just move boxes forward."""
        for t in self.tracks:
            t.display_bbox = self._predict(t, frame_idx)

    def remove(self, track: Track):
        self.tracks = [t for t in self.tracks if t.track_id != track.track_id]

    # ---- main update ---------------------------------------------------
    def update(self, dets: List[Detection], frame_idx: int, ts: datetime,
               vts: Optional[float]) -> List[Tuple[Track, Detection, bool]]:
        """Associate detections with tracks. Returns [(track, detection, is_new)]."""
        out: List[Tuple[Track, Detection, bool]] = []
        T, D = self.tracks, dets
        preds = np.array([self._predict(t, frame_idx) for t in T]).reshape(-1, 4)
        dboxes = np.array([d.bbox for d in D]).reshape(-1, 4)
        free_t, free_d = set(range(len(T))), set(range(len(D)))
        pairs = []

        if T and D:  # pass 1: IoU, globally optimal (Hungarian)
            iou = iou_matrix(preds, dboxes)
            rows, cols = linear_sum_assignment(1.0 - iou)
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    pairs.append((r, c)); free_t.discard(r); free_d.discard(c)

        if free_t and free_d:  # pass 2: centre-distance rescue for fast movers
            cands = []
            for r in free_t:
                pc = ((preds[r, 0] + preds[r, 2]) / 2, (preds[r, 1] + preds[r, 3]) / 2)
                size = max(preds[r, 2] - preds[r, 0], preds[r, 3] - preds[r, 1])
                for c in free_d:
                    dc = ((dboxes[c, 0] + dboxes[c, 2]) / 2, (dboxes[c, 1] + dboxes[c, 3]) / 2)
                    dist = np.hypot(pc[0] - dc[0], pc[1] - dc[1])
                    if dist < 0.6 * size:
                        cands.append((dist, r, c))
            for _, r, c in sorted(cands):
                if r in free_t and c in free_d:
                    pairs.append((r, c)); free_t.discard(r); free_d.discard(c)

        for r, c in pairs:
            t, d = T[r], D[c]
            dt = frame_idx - t.last_seen_frame
            if dt > 0:
                v = (d.bbox - t.bbox) / dt
                t.velocity = 0.5 * t.velocity + 0.5 * v
            t.bbox, t.conf = d.bbox.copy(), d.conf
            t.display_bbox = t.bbox.copy()
            t.hits += 1
            t.last_seen_frame, t.last_seen_ts, t.last_seen_vts = frame_idx, ts, vts
            out.append((t, d, False))

        for c in sorted(free_d):  # unmatched detections -> new tracks
            d = D[c]
            t = Track(self._next_id, d.bbox.copy(), d.conf, frame_idx, frame_idx, ts, vts)
            self._next_id += 1
            self.tracks.append(t)
            out.append((t, d, True))
        return out

    def expire(self, frame_idx: int) -> List[Track]:
        """Remove and return tracks not seen for too long (=> the face left the frame)."""
        gone, keep = [], []
        for t in self.tracks:
            limit = self.max_age if self.is_confirmed(t) else self.tentative_max_age
            (gone if frame_idx - t.last_seen_frame > limit else keep).append(t)
        self.tracks = keep
        return gone
