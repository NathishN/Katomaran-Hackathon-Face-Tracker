"""Writes entry/exit evidence: cropped image (structured folders) + DB row + log line."""
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2

from .database import Database, label
from .logging_setup import ev


class EventLogger:
    def __init__(self, log_dir: str, db: Database):
        self.root = Path(log_dir)
        self.db = db

    def _image_path(self, face_id: int, event_type: str, ts: datetime, track_id: int) -> Path:
        folder = self.root / ("entries" if event_type == "entry" else "exits") / ts.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{label(face_id)}_{event_type}_{ts.strftime('%H%M%S_%f')[:-3]}_t{track_id}.jpg"
        return folder / name

    def record(self, face_id: int, event_type: str, crop, ts: datetime, video_ts,
               frame_idx: int, track_id: int, **extra) -> str:
        """Persist ONE event. Image is written atomically first, then the DB row."""
        path_str = None
        if crop is not None and getattr(crop, "size", 0) > 0:
            path = self._image_path(face_id, event_type, ts, track_id)
            tmp = str(path) + ".tmp.jpg"
            try:
                cv2.imwrite(tmp, crop)
                os.replace(tmp, path)
                path_str = str(path)
            except Exception as exc:  # disk problems must not kill the pipeline
                ev("IMAGE_WRITE_FAILED", logging.ERROR, face=label(face_id), error=exc)
        try:
            self.db.add_event(face_id, event_type, ts.isoformat(timespec="milliseconds"),
                              video_ts, frame_idx, track_id, path_str)
        except Exception as exc:
            ev("DB_WRITE_FAILED", logging.ERROR, face=label(face_id), event=event_type, error=exc)
        ev(event_type.upper(), face=label(face_id), track=track_id, frame=frame_idx,
           video_t=None if video_ts is None else round(video_ts, 2), image=path_str, **extra)
        return path_str
