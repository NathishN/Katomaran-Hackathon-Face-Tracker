"""Core pipeline: detect -> track -> embed/recognise/register -> entry/exit logging -> counting.

Event semantics (exactly-once guarantees)
  * ENTRY : logged once when a track first receives an identity (new OR recognised face).
  * EXIT  : logged once when that track has not been seen for `max_age_frames` (or stream end).
  * A face re-identified while its previous track is merely *lost* (occlusion / missed detections)
    is MERGED into the old track -> no duplicate entry/exit, and never a new unique ID.
  * Unique count = number of rows in `faces`; re-identification never creates a row.
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np

from .database import Database, label
from .event_logger import EventLogger
from .logging_setup import ev
from .tracker import FaceTracker, Track


class Gallery:
    """In-memory embedding gallery (cosine similarity on L2-normalised vectors)."""

    def __init__(self, max_per_id: int = 5):
        self.max_per_id = max_per_id
        self.data: Dict[int, List[np.ndarray]] = {}
        self._ids = np.zeros(0, dtype=int)
        self._mat = np.zeros((0, 512), dtype=np.float32)
        self._dirty = False

    def add(self, face_id: int, emb: np.ndarray):
        lst = self.data.setdefault(face_id, [])
        lst.append(emb.astype(np.float32))
        if len(lst) > self.max_per_id:
            del lst[1]  # keep the first (anchor) embedding, drop the oldest refresh
        self._dirty = True

    def _rebuild(self):
        ids, vecs = [], []
        for fid, lst in self.data.items():
            for v in lst:
                ids.append(fid); vecs.append(v)
        self._ids = np.array(ids, dtype=int)
        self._mat = np.stack(vecs) if vecs else np.zeros((0, 512), dtype=np.float32)
        self._dirty = False

    def match(self, emb: np.ndarray):
        """Return (best_face_id | None, best_similarity)."""
        if self._dirty:
            self._rebuild()
        if len(self._ids) == 0:
            return None, -1.0
        sims = self._mat @ emb
        i = int(np.argmax(sims))
        return int(self._ids[i]), float(sims[i])

    def sim_to(self, face_id: int, emb: np.ndarray) -> float:
        lst = self.data.get(face_id, [])
        return max((float(v @ emb) for v in lst), default=-1.0)

    def __len__(self):
        return len(self.data)


def _sharpness(crop) -> float:
    if crop is None or crop.size == 0:
        return 0.0
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


class FacePipeline:
    def __init__(self, cfg, db: Database, events: EventLogger, detector, recognizer):
        self.cfg, self.db, self.events = cfg, db, events
        self.detector, self.recognizer = detector, recognizer
        self.skip = max(0, int(cfg.detection_skip_frames))
        r, t = cfg.recognition, cfg.tracking
        self.sim_thr = r.similarity_threshold
        self.max_attempts = r.max_attempts_per_track
        self.min_sharp = r.min_sharpness
        self.refresh_every = r.gallery_refresh_every_n_cycles
        self.tracker = FaceTracker(t.iou_threshold, t.min_hits, t.max_age_frames,
                                   t.tentative_max_age_frames, detect_interval=self.skip + 1)
        self.gallery = Gallery(r.max_embeddings_per_identity)
        for fid, vec in db.load_gallery():  # resume after restart
            self.gallery.add(fid, vec)
        if len(self.gallery):
            ev("GALLERY_LOADED", identities=len(self.gallery))
        self.log_every = cfg.output.tracking_log_interval_frames

    # ------------------------------------------------------------------
    def process_frame(self, frame, frame_idx: int, vts: Optional[float] = None) -> List[Track]:
        now = datetime.now()
        if frame_idx % (self.skip + 1) == 0:  # ---- detection cycle ----
            dets = self.detector.detect(frame)
            for tr, det, is_new in self.tracker.update(dets, frame_idx, now, vts):
                if is_new:
                    ev("TRACK_STARTED", track=tr.track_id, frame=frame_idx, conf=round(det.conf, 2))
                self._update_crops(tr, frame)
                if self.tracker.is_confirmed(tr):
                    if tr.face_id is None and not tr.gave_up:
                        self._identify(tr, frame, now, vts, frame_idx)
                    elif tr.face_id is not None:
                        self._maybe_refresh(tr, frame, now)
        else:  # ---- skipped frame: motion extrapolation only ----
            self.tracker.predict(frame_idx)

        for tr in self.tracker.expire(frame_idx):
            self._on_track_end(tr, reason="left_frame")

        if self.log_every and frame_idx % self.log_every == 0:
            for tr in self.tracker.tracks:
                if tr.face_id is not None and frame_idx - tr.last_seen_frame <= self.skip:
                    ev("TRACKING", face=label(tr.face_id), track=tr.track_id, frame=frame_idx,
                       bbox=[int(v) for v in tr.display_bbox])
        return [t for t in self.tracker.tracks if self.tracker.is_confirmed(t)]

    # ------------------------------------------------------------------
    def _update_crops(self, tr: Track, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in tr.bbox]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return
        crop = frame[y1:y2, x1:x2].copy()
        sharp = _sharpness(crop)
        tr.cur_sharpness, tr.last_crop = sharp, crop
        score = tr.conf * np.sqrt(min(crop.shape[0] * crop.shape[1], 160 * 160)) * np.log1p(sharp)
        if score > tr.best_score:
            tr.best_score, tr.best_crop = score, crop

    def _identify(self, tr: Track, frame, now, vts, frame_idx):
        if tr.cur_sharpness < self.min_sharp:
            return  # blurry frame: wait for a better one (doesn't count as an attempt)
        tr.embed_attempts += 1
        emb = self.recognizer.embed(frame, tr.bbox)
        if emb is None:
            ev("EMBEDDING_FAILED", logging.WARNING, track=tr.track_id, attempt=tr.embed_attempts)
            if tr.embed_attempts >= self.max_attempts:
                tr.gave_up = True
                ev("TRACK_UNIDENTIFIED", logging.WARNING, track=tr.track_id)
            return
        ev("EMBEDDING_GENERATED", track=tr.track_id, dim=int(emb.shape[0]), frame=frame_idx)
        face_id, sim = self.gallery.match(emb)

        if face_id is not None and sim >= self.sim_thr:  # ---- known face ----
            ev("FACE_RECOGNIZED", face=label(face_id), track=tr.track_id, similarity=round(sim, 3))
            old = next((t for t in self.tracker.tracks
                        if t is not tr and t.face_id == face_id and t.last_seen_frame < frame_idx), None)
            if old is not None:  # same person, fragmented track -> continue, don't re-log entry
                self.tracker.remove(old)
                old.retired = True
                tr.face_id, tr.entered = face_id, True
                ev("TRACK_MERGED", face=label(face_id), old_track=old.track_id, new_track=tr.track_id)
                return
            tr.face_id = face_id
            self.db.increment_visit(face_id)
            self._log_entry(tr, now, vts, frame_idx, new=False, sim=round(sim, 3))
        else:  # ---- brand-new face ----
            face_id = self.db.register_face(emb, now.isoformat(timespec="milliseconds"))
            self.gallery.add(face_id, emb)
            tr.face_id = face_id
            self.db.increment_visit(face_id)
            ev("FACE_REGISTERED", face=label(face_id), track=tr.track_id,
               best_sim_to_existing=round(sim, 3), unique_count=self.db.unique_visitor_count())
            self._log_entry(tr, now, vts, frame_idx, new=True)

    def _log_entry(self, tr: Track, now, vts, frame_idx, new: bool, **extra):
        path = self.events.record(tr.face_id, "entry", tr.best_crop if tr.best_crop is not None else tr.last_crop,
                                  now, vts, frame_idx, tr.track_id, new_face=new, **extra)
        if new and path:
            self.db.set_face_image(tr.face_id, path)
        tr.entered = True

    def _maybe_refresh(self, tr: Track, frame, now):
        """Occasionally add a fresh embedding (pose/lighting diversity) for recognised tracks."""
        if not self.refresh_every:
            return
        tr.cycles_since_embed += 1
        if tr.cycles_since_embed < self.refresh_every or tr.cur_sharpness < self.min_sharp:
            return
        tr.cycles_since_embed = 0
        emb = self.recognizer.embed(frame, tr.bbox)
        if emb is None:
            return
        sim = self.gallery.sim_to(tr.face_id, emb)
        if sim >= self.sim_thr:
            self.gallery.add(tr.face_id, emb)
            self.db.add_embedding(tr.face_id, emb, now.isoformat(timespec="milliseconds"))
        else:
            ev("IDENTITY_DRIFT_WARNING", logging.WARNING, face=label(tr.face_id),
               track=tr.track_id, similarity=round(sim, 3))

    def _on_track_end(self, tr: Track, reason: str):
        if tr.retired:
            return
        if tr.face_id is None or not tr.entered:
            ev("TRACK_DROPPED", track=tr.track_id, reason="never_identified", hits=tr.hits)
            return
        self.events.record(tr.face_id, "exit", tr.last_crop, tr.last_seen_ts, tr.last_seen_vts,
                           tr.last_seen_frame, tr.track_id, reason=reason)

    def flush(self):
        """Stream ended / shutdown: emit the exit event for everyone still in frame."""
        for tr in list(self.tracker.tracks):
            self._on_track_end(tr, reason="stream_end")
        self.tracker.tracks.clear()
