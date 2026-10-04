"""Print unique-visitor count + event summary straight from the database (and export CSV)."""
import argparse
import csv
import sqlite3

ap = argparse.ArgumentParser()
ap.add_argument("--db", default="data/visitors.db")
ap.add_argument("--csv", help="optional path to export events as CSV")
a = ap.parse_args()

c = sqlite3.connect(a.db)
print("Unique visitors :", c.execute("SELECT COUNT(*) FROM faces").fetchone()[0])
for t, n in c.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type"):
    print(f"{t:>6} events   :", n)
print("\nface_id | label     | registered_at            | visits")
for r in c.execute("SELECT face_id,label,registered_at,visit_count FROM faces"):
    print(" | ".join(str(x) for x in r))
print("\nlast 15 events:")
for r in c.execute("SELECT face_id,event_type,timestamp,frame_idx,image_path FROM events "
                   "ORDER BY id DESC LIMIT 15"):
    print(r)
if a.csv:
    rows = c.execute("SELECT * FROM events").fetchall()
    with open(a.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([d[0] for d in c.execute("SELECT * FROM events").description])
        w.writerows(rows)
    print("exported ->", a.csv)
