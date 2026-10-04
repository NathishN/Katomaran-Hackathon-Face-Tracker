"""Overlay drawing for the annotated output video / live window."""
import cv2

from .database import label


def draw(frame, tracks, unique_count: int, frame_idx: int):
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.display_bbox]
        color = (0, 200, 0) if t.face_id is not None else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        text = label(t.face_id) if t.face_id is not None else f"track {t.track_id}"
        cv2.putText(frame, text, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.rectangle(frame, (0, 0), (300, 34), (0, 0, 0), -1)
    cv2.putText(frame, f"Unique visitors: {unique_count}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame
