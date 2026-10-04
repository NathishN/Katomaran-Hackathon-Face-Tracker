"""Intelligent Face Tracker - entry point.

    python main.py                         # uses config.json (video file by default)
    python main.py --source rtsp://...     # live RTSP camera
    python main.py --fresh                 # wipe DB/gallery first
"""
import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import cv2

from src.config import load_config
from src.database import Database
from src.event_logger import EventLogger
from src.logging_setup import ev, setup_logging
from src.pipeline import FacePipeline
from src.video_source import VideoSource
from src.visualize import draw

_STOP = False


def _on_signal(signum, _frame):
    global _STOP
    _STOP = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--source", help="video path / rtsp url / webcam index (overrides config)")
    ap.add_argument("--fresh", action="store_true", help="reset database before starting")
    ap.add_argument("--show", action="store_true", help="show live preview window")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.source:
        cfg.video_source = args.source
    if args.show:
        cfg.output.show_window = True

    setup_logging(cfg.storage.log_dir, cfg.output.log_level)
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    db = Database(cfg.storage.db_path, reset=args.fresh or cfg.storage.reset_on_start)
    events = EventLogger(cfg.storage.log_dir, db)
    ev("SYSTEM_START", source=cfg.video_source, skip_frames=cfg.detection_skip_frames,
       db=cfg.storage.db_path)

    from src.detector import FaceDetector      # heavy imports after logging is ready
    from src.recognizer import FaceRecognizer
    pipeline = FacePipeline(cfg, db, events, FaceDetector(cfg.detector), FaceRecognizer(cfg.recognition))

    src = VideoSource(cfg.video_source, cfg.stream)
    if not src.open():
        ev("SOURCE_OPEN_FAILED", logging.ERROR, source=cfg.video_source)
        sys.exit(1)

    writer, frame_idx, consecutive_errors, t0 = None, 0, 0, time.time()
    try:
        while not _STOP:
            ok, frame = src.read()
            if not ok:
                break                      # file ended / stream permanently dead
            if frame is None:
                continue                   # stream hiccup, wait for next frame
            try:
                tracks = pipeline.process_frame(frame, frame_idx, src.video_time(frame_idx))
                consecutive_errors = 0
            except Exception as exc:       # one bad frame must never kill the service
                consecutive_errors += 1
                ev("FRAME_ERROR", logging.ERROR, frame=frame_idx, error=repr(exc))
                if consecutive_errors > 50:
                    ev("TOO_MANY_ERRORS", logging.CRITICAL)
                    break
                frame_idx += 1
                continue

            if cfg.output.save_annotated_video or cfg.output.show_window:
                vis = draw(frame, tracks, db.unique_visitor_count(), frame_idx)
                if cfg.output.save_annotated_video:
                    if writer is None:
                        Path(cfg.output.annotated_path).parent.mkdir(parents=True, exist_ok=True)
                        h, w = vis.shape[:2]
                        writer = cv2.VideoWriter(cfg.output.annotated_path,
                                                 cv2.VideoWriter_fourcc(*"mp4v"), src.fps, (w, h))
                    writer.write(vis)
                if cfg.output.show_window:
                    cv2.imshow("Face Tracker", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        pipeline.flush()                   # guarantees an EXIT for everyone still tracked
        counts = db.event_counts()
        unique = db.unique_visitor_count()
        ev("SYSTEM_STOP", frames=frame_idx, seconds=round(time.time() - t0, 1),
           unique_visitors=unique, entries=counts["entry"], exits=counts["exit"])
        if writer:
            writer.release()
        src.release()
        db.close()
        cv2.destroyAllWindows()
    print(f"\n=== UNIQUE VISITORS: {unique} | entries: {counts['entry']} | exits: {counts['exit']} ===")


if __name__ == "__main__":
    main()
