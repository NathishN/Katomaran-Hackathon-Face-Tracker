# Katomaran Hackathon - Demo Video Script (2.5 Minutes)
**Project:** Intelligent Face Tracker with Auto-Registration & Unique Visitor Counting  
**Target Platform:** Loom / YouTube (Unlisted)  
**Presenter:** [Your Name]  
**Setup Before Recording:**
1. Have browser open at `http://localhost:5000` (Dashboard tab).
2. Have terminal ready or video already loaded.
3. Keep screen resolution clear (1080p recommended).

---

## Video Timeline & Script

### [0:00 - 0:25] Hook & Problem Statement
* **Screen:** FaceTracker AI Dashboard (`http://localhost:5000`). Point cursor to the header and KPI cards.
* **Spoken Script:**
  > "Hello everyone! Welcome to my submission for the Katomaran Hackathon: **FaceTracker AI**, an intelligent, real-time visitor analytics and face recognition system.
  > 
  > In high-traffic venues, retail stores, or security checkpoints, standard object detectors fail because they simply count bounding boxes—leading to massive over-counting whenever people turn away, get occluded, or re-enter. 
  > 
  > Our objective was to build a robust, production-grade edge pipeline that accurately tracks unique individuals, auto-registers unseen visitors, deduplicates returning faces, and emits exactly one entry and one exit event with photo evidence."

---

### [0:25 - 0:55] Architecture & Core Innovations
* **Screen:** Highlight the live video stream player and briefly toggle to the Architecture section in the README or keep focus on the visual detections.
* **Spoken Script:**
  > "To achieve real-time speed with high accuracy, our pipeline combines four specialized stages:
  > 1. **Detection:** A lightweight **YOLOv8-face** model running with configurable frame skipping.
  > 2. **Multi-Object Tracking:** An **IoU + constant-velocity Kalman-style tracker** using Hungarian bipartite matching to maintain tracks between detection frames.
  > 3. **Feature Extraction & Alignment:** Rather than embedding every frame, we extract features only once per confirmed track. We use **InsightFace's SCRFD** for 5-point facial landmark alignment, followed by **ArcFace (buffalo_l)** to produce 512-dimensional L2-normalized embeddings.
  > 4. **Gallery Deduplication:** A dynamic cosine similarity gallery matches incoming faces against stored visitors with a calibrated threshold of 0.40."

---

### [0:55 - 1:40] Live System Walkthrough
* **Screen:** 
  - Show the **Live Video Feed** playing with bounding boxes, tracking labels (`FACE-0001`, `FACE-0002`), and detection confidence.
  - Show the **Real-Time KPIs** updating: Unique Visitors (30), Entries (34), Exits (34).
  - Scroll the **Real-Time Events Stream** on the right sidebar showing live face thumbnail crops.
* **Spoken Script:**
  > "Let's see it in action on the hackathon's benchmark dataset.
  > 
  > As you can see on the live dashboard, faces are detected and tracked smoothly. Every time a new person enters the frame, the system automatically registers them as a new face ID, captures an aligned face crop, and emits an `ENTRY` event in our real-time SSE stream.
  > 
  > Notice the metric counters: on this sample dataset, the system detected **34 total entry appearances**, but correctly resolved them to **30 unique visitors**. 
  > 
  > For example, visitor `FACE-0010` and `FACE-0024` stepped out of the camera view and re-entered later. Instead of registering duplicate identities, our ArcFace gallery recognized their cosine similarity and matched them back to their original ID—keeping the unique visitor count 100% accurate."

---

### [1:40 - 2:05] Visitor Gallery & Audit Log
* **Screen:** 
  - Click **Visitor Gallery** on the left navigation bar.
  - Hover over a few cards showing registered face crops and visit counts (e.g., visits: 2).
  - Click **Events Log** to show the tabular audit log.
* **Spoken Script:**
  > "Navigating to the **Visitor Gallery**, you can inspect every registered unique visitor, their initial enrollment snapshot, and their visit count.
  > 
  > Clicking on the **Events Log** gives security teams a complete, chronological audit trail. Every single entry and exit includes the exact timestamp, video time in seconds, frame number, track ID, and the cropped face evidence.
  > 
  > Behind the scenes, everything is committed into a crash-resilient SQLite database configured with WAL mode and atomic disk writes, ensuring zero data loss even during unexpected power loss."

---

### [2:05 - 2:30] Flexibility & Conclusion
* **Screen:** 
  - Click **Start Tracking** to reveal the modal showing **Upload Video**, **RTSP / Network**, and **Webcam / Local**.
  - Return to the Dashboard overview.
* **Spoken Script:**
  > "Finally, the system is built for real-world deployment: it natively supports drag-and-drop video uploads, local webcams, and live **RTSP IP camera streams** with auto-reconnection and a threaded latest-frame buffer to eliminate latency.
  > 
  > The entire solution is self-contained, fully tested with unit test coverage, and ready to deploy.
  > 
  > Thank you to the Katomaran team for this exciting challenge!"

---

## Tips for Recording
1. **Pacing:** Speak clearly and at a steady, natural pace.
2. **Audio:** Use a headset mic or quiet room to avoid background noise.
3. **Cursor:** Use your mouse pointer intentionally to direct the judges' attention to the cards and charts as you speak.
4. **Length:** Keep total recording time between **2:15 and 2:45 minutes**.
