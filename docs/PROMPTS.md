# Prompts used (edit this file to match what you actually asked; be ready to explain every line of code)

1. **Planning** – "Act as a CV/surveillance engineer. Analyse this hackathon brief and produce a feature list, architecture, compute estimate and module breakdown for a YOLO + InsightFace unique visitor counter."
2. **Detector/Recognizer** – "Write a modular Python YOLOv8 face detector wrapper and an InsightFace buffalo_l embedding wrapper that aligns faces and returns L2-normalised 512-d vectors."
3. **Tracker** – "Write an IoU + constant-velocity tracker with Hungarian matching that works when detection is skipped every N frames."
4. **Events/DB** – "Design a SQLite schema (faces, embeddings, events) and an event logger that writes crops to logs/entries|exits/YYYY-MM-DD, DB rows and events.log atomically."
5. **Exactly-once logic** – "How do I guarantee exactly one entry and one exit per appearance and avoid double counting when a track breaks?"
6. **Testing** – "Create a mock-detector/recogniser test with re-entry and a detector dropout; assert unique=2, entries=3, exits=3."
7. **Docs** – "Write the README with planning doc, mermaid architecture diagram, config sample, assumptions."
