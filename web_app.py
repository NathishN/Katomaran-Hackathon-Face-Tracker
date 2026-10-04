"""
Flask web backend for the Intelligent Face Tracker dashboard.
Run: python web_app.py  -> open http://localhost:5000
"""
import json, logging, os, queue, sqlite3, threading, time, shutil
from pathlib import Path
from typing import Optional

import cv2
from flask import (Flask, Response, jsonify, render_template,
                   request, send_from_directory, stream_with_context)

BASE = Path(__file__).parent
UPLOAD_DIR = BASE / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="frontend", static_folder="frontend/static")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

_state = {
    "running": False, "thread": None,
    "stop_event": threading.Event(),
    "source": None,
    "frame_queue": queue.Queue(maxsize=2),
    "sse_clients": [], "sse_lock": threading.Lock(),
    "active_tracks": 0,
    "db_path": str(BASE / "data" / "visitors.db"),
    "error": None,
}

log = logging.getLogger("web_app")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

def _push_sse(data: dict):
    payload = f"data: {json.dumps(data)}\n\n"
    with _state["sse_lock"]:
        dead = []
        for q in _state["sse_clients"]:
            try: q.put_nowait(payload)
            except queue.Full: dead.append(q)
        for q in dead: _state["sse_clients"].remove(q)

def _web_path(path: Optional[str]) -> Optional[str]:
    if not path: return None
    p = Path(path)
    try:
        rel = p.relative_to(BASE / "logs")
        return f"/logs/{rel.as_posix()}"
    except ValueError:
        return None

def _run_pipeline(source: str, fresh: bool, cfg_overrides: dict):
    os.chdir(BASE)
    try:
        from src.config import load_config
        from src.database import Database
        from src.detector import FaceDetector
        from src.event_logger import EventLogger
        from src.logging_setup import setup_logging
        from src.pipeline import FacePipeline
        from src.recognizer import FaceRecognizer
        from src.video_source import VideoSource
        from src.visualize import draw

        cfg = load_config("config.json", overrides=cfg_overrides)
        if source: cfg.video_source = source
        _state["db_path"] = cfg.storage.db_path
        setup_logging(cfg.storage.log_dir, cfg.output.log_level)

        db = Database(cfg.storage.db_path, reset=fresh)
        events = EventLogger(cfg.storage.log_dir, db)

        _orig_record = events.record
        def _patched_record(face_id, event_type, crop, ts, video_ts, frame_idx, track_id, **extra):
            path = _orig_record(face_id, event_type, crop, ts, video_ts, frame_idx, track_id, **extra)
            from src.database import label
            _push_sse({"type":"event","event_type":event_type,"face":label(face_id),
                       "track":track_id,"frame":frame_idx,
                       "video_t":round(video_ts,2) if video_ts else None,
                       "image":_web_path(path),"ts":ts.isoformat(timespec="milliseconds"),
                       **{k:str(v) for k,v in extra.items()}})
            return path
        events.record = _patched_record

        pipeline = FacePipeline(cfg, db, events, FaceDetector(cfg.detector), FaceRecognizer(cfg.recognition))
        src = VideoSource(cfg.video_source, cfg.stream)
        if not src.open():
            _state["error"] = f"Cannot open source: {cfg.video_source}"
            _state["running"] = False; return

        stop = _state["stop_event"]
        frame_idx = 0; consecutive_errors = 0
        while not stop.is_set():
            ok, frame = src.read()
            if not ok: break
            if frame is None: continue
            try:
                tracks = pipeline.process_frame(frame, frame_idx, src.video_time(frame_idx))
                consecutive_errors = 0
                _state["active_tracks"] = sum(1 for t in tracks if t.face_id is not None)
            except Exception as exc:
                consecutive_errors += 1
                log.error("Frame %d error: %s", frame_idx, exc)
                if consecutive_errors > 50: break
                frame_idx += 1; continue

            vis = draw(frame, tracks, db.unique_visitor_count(), frame_idx)
            ret, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                try: _state["frame_queue"].put_nowait(buf.tobytes())
                except queue.Full:
                    try: _state["frame_queue"].get_nowait()
                    except queue.Empty: pass
                    try: _state["frame_queue"].put_nowait(buf.tobytes())
                    except queue.Full: pass

            if frame_idx % 30 == 0:
                _push_sse({"type":"stats","unique":db.unique_visitor_count(),
                           "active":_state["active_tracks"],**db.event_counts()})
            frame_idx += 1

        pipeline.flush()
        unique = db.unique_visitor_count(); counts = db.event_counts()
        src.release(); db.close()
        _push_sse({"type":"stopped","unique":unique,**counts})
    except Exception as exc:
        log.exception("Pipeline error: %s", exc)
        _state["error"] = str(exc)
        _push_sse({"type":"error","message":str(exc)})
    finally:
        _state["running"] = False; _state["active_tracks"] = 0

def _get_db():
    db_path = _state["db_path"]
    if not Path(db_path).exists(): return None
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row; return conn

@app.route("/")
def index(): return render_template("index.html")

@app.route("/api/stats")
def api_stats():
    conn = _get_db()
    if not conn:
        return jsonify({"unique":0,"entry":0,"exit":0,"active":0,
                        "running":_state["running"],"error":_state["error"]})
    try:
        unique = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
        rows = conn.execute("SELECT event_type,COUNT(*) as n FROM events GROUP BY event_type").fetchall()
        counts = {"entry":0,"exit":0}; counts.update({r["event_type"]:r["n"] for r in rows})
        return jsonify({"unique":unique,"active":_state["active_tracks"],
                        "running":_state["running"],"error":_state["error"],**counts})
    finally: conn.close()

@app.route("/api/visitors")
def api_visitors():
    conn = _get_db()
    if not conn: return jsonify([])
    try:
        rows = conn.execute("SELECT face_id,label,registered_at,first_image_path,visit_count "
                            "FROM faces ORDER BY face_id DESC LIMIT 100").fetchall()
        return jsonify([{"face_id":r["face_id"],"label":r["label"],
                         "registered_at":r["registered_at"],
                         "image":_web_path(r["first_image_path"]),
                         "visit_count":r["visit_count"]} for r in rows])
    finally: conn.close()

@app.route("/api/events")
def api_events():
    limit = int(request.args.get("limit", 50))
    conn = _get_db()
    if not conn: return jsonify([])
    try:
        rows = conn.execute(
            "SELECT e.id,e.face_id,f.label,e.event_type,e.timestamp,"
            "e.video_time_sec,e.frame_idx,e.track_id,e.image_path "
            "FROM events e JOIN faces f ON f.face_id=e.face_id "
            "ORDER BY e.id DESC LIMIT ?", (limit,)).fetchall()
        return jsonify([{"id":r["id"],"face_id":r["face_id"],"label":r["label"],
                         "event_type":r["event_type"],"timestamp":r["timestamp"],
                         "video_time_sec":r["video_time_sec"],"frame_idx":r["frame_idx"],
                         "track_id":r["track_id"],"image":_web_path(r["image_path"])} for r in rows])
    finally: conn.close()

@app.route("/api/start", methods=["POST"])
def api_start():
    if _state["running"]: return jsonify({"ok":False,"error":"Already running"})
    data = request.get_json(silent=True) or {}
    source = data.get("source",""); fresh = bool(data.get("fresh",False))
    overrides = data.get("overrides",{})
    _state["stop_event"].clear(); _state["error"] = None; _state["running"] = True
    t = threading.Thread(target=_run_pipeline, args=(source,fresh,overrides), daemon=True)
    t.start(); _state["thread"] = t
    return jsonify({"ok":True})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    _state["stop_event"].set(); return jsonify({"ok":True})

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files: return jsonify({"ok":False,"error":"No file part"}),400
    f = request.files["file"]
    if f.filename == "": return jsonify({"ok":False,"error":"No selected file"}),400
    dest = UPLOAD_DIR / f.filename; f.save(str(dest))
    shutil.copy(str(dest), str(BASE/"data"/"sample_video.mp4"))
    return jsonify({"ok":True,"path":str(dest),"source":str(dest)})

@app.route("/api/config", methods=["GET"])
def api_config_get():
    p = BASE/"config.json"
    if p.exists(): return p.read_text(encoding="utf-8"),200,{"Content-Type":"application/json"}
    return jsonify({})

@app.route("/api/config", methods=["POST"])
def api_config_post():
    data = request.get_json()
    if not data: return jsonify({"ok":False,"error":"No JSON body"}),400
    p = BASE/"config.json"; p.write_text(json.dumps(data,indent=2),encoding="utf-8")
    return jsonify({"ok":True})

def _make_placeholder():
    import numpy as np
    img = np.zeros((360,640,3),dtype=np.uint8)
    cv2.putText(img,"No video feed — press Start",(120,175),cv2.FONT_HERSHEY_SIMPLEX,1.0,(80,80,80),2)
    cv2.putText(img,"Upload a video or set RTSP source",(100,220),cv2.FONT_HERSHEY_SIMPLEX,0.65,(60,60,60),1)
    _,buf=cv2.imencode(".jpg",img); return buf.tobytes()

def _generate_mjpeg():
    placeholder = _make_placeholder()
    while True:
        try: frame_bytes = _state["frame_queue"].get(timeout=0.5)
        except queue.Empty: frame_bytes = placeholder
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.route("/video_feed")
def video_feed():
    return Response(stream_with_context(_generate_mjpeg()),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/stream")
def api_stream():
    client_q: queue.Queue = queue.Queue(maxsize=50)
    with _state["sse_lock"]: _state["sse_clients"].append(client_q)
    def _gen():
        try:
            yield "data: {\"type\":\"connected\"}\n\n"
            while True:
                try: msg = client_q.get(timeout=20); yield msg
                except queue.Empty: yield ": heartbeat\n\n"
        except GeneratorExit: pass
        finally:
            with _state["sse_lock"]:
                try: _state["sse_clients"].remove(client_q)
                except ValueError: pass
    return Response(stream_with_context(_gen()), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/logs/<path:filename>")
def serve_log_image(filename): return send_from_directory(BASE/"logs", filename)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Face Tracker Web Dashboard")
    print("  Open: http://localhost:5000")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
