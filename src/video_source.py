"""Video input: file (sequential) or RTSP/HTTP stream (threaded, always-latest-frame, auto-reconnect)."""
import logging
import os
import threading
import time

import cv2

from .logging_setup import ev


def _is_stream(src) -> bool:
    return isinstance(src, str) and src.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))


class VideoSource:
    def __init__(self, source, stream_cfg):
        if isinstance(source, str) and source.isdigit():
            source = int(source)  # webcam index
        self.source = source
        self.cfg = stream_cfg
        self.is_stream = _is_stream(source) or isinstance(source, int)
        self.cap = None
        self.fps = 25.0
        self._t0 = None
        # threaded reader state
        self._lock = threading.Lock()
        self._frame = None
        self._fresh = False
        self._stop = False
        self._dead = False
        self._thread = None

    def _open_cap(self) -> bool:
        if isinstance(self.source, str) and self.source.lower().startswith("rtsp"):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
        self.cap = cv2.VideoCapture(self.source)
        return bool(self.cap and self.cap.isOpened())

    def open(self) -> bool:
        ok = self._open_cap()
        if not ok:
            return False
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and 1 < fps < 240 else 25.0
        self._t0 = time.time()
        ev("SOURCE_OPENED", source=self.source, fps=round(self.fps, 2), stream=self.is_stream)
        if self.is_stream:
            self._thread = threading.Thread(target=self._reader, daemon=True)
            self._thread.start()
        return True

    def _reader(self):
        fails = 0
        while not self._stop:
            ok, frame = self.cap.read()
            if ok:
                fails = 0
                with self._lock:
                    self._frame, self._fresh = frame, True
                continue
            fails += 1
            ev("STREAM_READ_FAILED", logging.WARNING, attempt=fails)
            if self.cfg.max_reconnect_attempts and fails > self.cfg.max_reconnect_attempts:
                ev("STREAM_GAVE_UP", logging.ERROR)
                self._dead = True
                return
            time.sleep(self.cfg.reconnect_delay_sec)
            try:
                self.cap.release()
            except Exception:
                pass
            if self._open_cap():
                ev("STREAM_RECONNECTED")

    def read(self):
        """Returns (ok, frame)."""
        if not self.is_stream:
            return self.cap.read()
        t_end = time.time() + 5
        while time.time() < t_end:
            with self._lock:
                if self._fresh:
                    self._fresh = False
                    return True, self._frame.copy()
            if self._dead or self._stop:
                return False, None
            time.sleep(0.003)
        return (not self._dead), None  # timeout (no new frame yet): caller retries

    def video_time(self, frame_idx: int) -> float:
        return (time.time() - self._t0) if self.is_stream else frame_idx / self.fps

    def release(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)
        if self.cap:
            self.cap.release()
