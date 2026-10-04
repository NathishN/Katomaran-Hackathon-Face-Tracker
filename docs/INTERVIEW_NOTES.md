# Interview cheat-sheet
* **Why YOLO + InsightFace?** YOLO = fast localisation; ArcFace = discriminative embeddings. Alignment via SCRFD landmarks on the YOLO crop.
* **Why not embed every frame?** Cost. Embed once per track (+refresh every 20 cycles); tracker keeps identity between.
* **Skip frames?** `detection_skip_frames` in config.json; tracker extrapolates with velocity. Aging thresholds auto-adjust to cadence.
* **Unique count?** `SELECT COUNT(*) FROM faces`; a row is only created when best cosine < 0.40 against every stored embedding.
* **Exactly-once entry/exit?** `entered` flag on track; exit emitted only from `_on_track_end`; merged tracks are `retired`.
* **Re-ID after occlusion?** Same face re-recognised while old track lost → `TRACK_MERGED`, no new events.
* **Resilience?** WAL + synchronous FULL, atomic image rename, try/except per frame, SIGINT flush, gallery reload.
* **RTSP lag?** Reader thread keeps only the newest frame; TCP transport; auto-reconnect.
* **Tuning knobs?** similarity_threshold, max_age_frames, detection_skip_frames, confidence, min_face_size_px.
* **Known weaknesses?** Twins, masks, extreme angles, very low resolution; similarity threshold is dataset-dependent.
