"""SQLite persistence (WAL mode, commit-per-write => resilient to crashes/interruptions)."""
import os
import sqlite3
import threading
from pathlib import Path
from typing import List, Tuple

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS faces (
    face_id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT,
    registered_at TEXT NOT NULL,
    first_image_path TEXT,
    visit_count INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id INTEGER NOT NULL REFERENCES faces(face_id),
    vector BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    face_id INTEGER NOT NULL REFERENCES faces(face_id),
    event_type TEXT NOT NULL CHECK (event_type IN ('entry','exit')),
    timestamp TEXT NOT NULL,
    video_time_sec REAL,
    frame_idx INTEGER,
    track_id INTEGER,
    image_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_face ON events(face_id);
"""


def label(face_id: int) -> str:
    return f"FACE-{face_id:04d}"


class Database:
    def __init__(self, path: str, reset: bool = False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if reset:
            for suffix in ("", "-wal", "-shm"):
                if os.path.exists(path + suffix):
                    os.remove(path + suffix)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def _exec(self, sql, params=()):
        with self._lock, self.conn:  # `with conn` => commit / rollback
            cur = self.conn.execute(sql, params)
            return cur.lastrowid

    def register_face(self, embedding: np.ndarray, ts_iso: str) -> int:
        with self._lock, self.conn:
            cur = self.conn.execute(
                "INSERT INTO faces(registered_at, visit_count) VALUES (?, 0)", (ts_iso,))
            fid = cur.lastrowid
            self.conn.execute("UPDATE faces SET label=? WHERE face_id=?", (label(fid), fid))
            self.conn.execute(
                "INSERT INTO embeddings(face_id, vector, created_at) VALUES (?,?,?)",
                (fid, embedding.astype(np.float32).tobytes(), ts_iso))
        return fid

    def add_embedding(self, face_id: int, embedding: np.ndarray, ts_iso: str):
        self._exec("INSERT INTO embeddings(face_id, vector, created_at) VALUES (?,?,?)",
                   (face_id, embedding.astype(np.float32).tobytes(), ts_iso))

    def set_face_image(self, face_id: int, path: str):
        self._exec("UPDATE faces SET first_image_path=? WHERE face_id=?", (path, face_id))

    def increment_visit(self, face_id: int):
        self._exec("UPDATE faces SET visit_count = visit_count + 1 WHERE face_id=?", (face_id,))

    def add_event(self, face_id, event_type, ts_iso, video_ts, frame_idx, track_id, image_path):
        return self._exec(
            "INSERT INTO events(face_id,event_type,timestamp,video_time_sec,frame_idx,track_id,image_path)"
            " VALUES (?,?,?,?,?,?,?)",
            (face_id, event_type, ts_iso, video_ts, frame_idx, track_id, image_path))

    def unique_visitor_count(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]

    def event_counts(self) -> dict:
        with self._lock:
            rows = self.conn.execute(
                "SELECT event_type, COUNT(*) FROM events GROUP BY event_type").fetchall()
        d = {"entry": 0, "exit": 0}
        d.update({k: v for k, v in rows})
        return d

    def load_gallery(self) -> List[Tuple[int, np.ndarray]]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT face_id, vector FROM embeddings ORDER BY id").fetchall()
        return [(fid, np.frombuffer(blob, dtype=np.float32).copy()) for fid, blob in rows]

    def close(self):
        with self._lock:
            try:
                self.conn.commit()
                self.conn.close()
            except Exception:
                pass
