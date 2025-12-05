"""
Optimized Face Detection + Recognition pipeline for CPU (i3-14100).
Features:
- SCRFD + ArcFace (insightface) for fast detection + embeddings on CPU
- Simple IOU-based tracker (greedy matching) to maintain track_id per face
- Threaded capture + processing + recognition via ThreadPoolExecutor
- Unknown-person logic: if not recognized in 3s -> save bbox every 0.5s up to 5 photos
- SQLite storage for records (embeddings stored as blobs or as files; here we store paths + metadata)
"""

import os
import time
import cv2
import numpy as np
import threading
import queue
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Dict

# ---- insightface imports (SCRFD + ArcFace) ----
# from insightface import FaceAnalysis
from insightface.app import FaceAnalysis

from insightface.data import get_image as ins_get_yimage  # optional

# ---------------- CONFIG ------------------------
RTSP_URLS = [
    # example: f"rtsp://{username}:{password}@{ip}:{port}/ISAPI/Streaming/Channels/101"
    # подставьте свои url(ы)
    "rtsp://admin:12345678@10.1.84.231:554/ISAPI/Streaming/Channels/101",
]

DISPLAY_SCALE = 0.7
FRAME_QUEUE_MAX = 2

# Timing / saving behavior
RECOGNITION_TIMEOUT = 3.0   # seconds until we start saving unknown
SAVE_INTERVAL = 0.5         # seconds between saves
MAX_SAVE_PHOTOS = 5

# Matching (tracker) settings
IOU_MATCH_THRESHOLD = 0.3
MAX_MISSES = 10   # if not seen for N frames we drop track

# Paths
KNOWN_DIR = "faces_db/known"
UNKNOWN_DIR = "faces_db/unknown"
DB_PATH = "faces_db/faces.db"
os.makedirs(KNOWN_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

# InsightFace prepare settings
DET_SIZE = (640, 640)  # SCRFD input size (can be smaller for speed)
FACE_MIN_AREA = 1000   # ignore too-small detections


# ---------------- Utilities ---------------------
def iou(boxA, boxB):
    # box = [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    if union <= 0:
        return 0.0
    return inter / union


def rect_area(box):
    return max(0, box[2]-box[0]) * max(0, box[3]-box[1])


def crop_box_safe(img, box):
    h, w = img.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(w, int(box[2]))
    y2 = min(h, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


# ---------------- Database (SQLite) ----------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS unknowns (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_tag TEXT,
        first_seen REAL,
        last_seen REAL,
        saved_count INTEGER,
        assigned_name TEXT,
        sample_photo TEXT
    )
    """)
    con.commit()
    con.close()

init_db()


# ---------------- Face engine (InsightFace) --------------
print("[INFO] Инициализация InsightFace (SCRFD + ArcFace)...")
fa = FaceAnalysis(allowed_modules=['detection', 'recognition'])
fa.prepare(ctx_id=-1, det_size=DET_SIZE)  # ctx_id=-1 -> CPU
print("[INFO] InsightFace готов.")


# ---------------- Load known faces (embeddings) -----------
known_embeddings: List[np.ndarray] = []
known_names: List[str] = []

def load_known_faces():
    global known_embeddings, known_names
    known_embeddings = []
    known_names = []
    print("[INFO] Загрузка известных лиц из:", KNOWN_DIR)
    for person in os.listdir(KNOWN_DIR):
        person_folder = os.path.join(KNOWN_DIR, person)
        if not os.path.isdir(person_folder):
            continue
        for imgfile in os.listdir(person_folder):
            path = os.path.join(person_folder, imgfile)
            try:
                img = cv2.imread(path)
                if img is None:
                    continue
                faces = fa.get(img)
                if len(faces) > 0:
                    emb = faces[0].embedding
                    known_embeddings.append(np.array(emb, dtype=np.float32))
                    known_names.append(person)
            except Exception as e:
                print("[WARN] load_known_faces error:", e)
    print(f"[INFO] Загружено {len(known_embeddings)} известных эмбеддингов.")

load_known_faces()


# ---------------- Simple IOU-based tracker -----------------
class Track:
    def __init__(self, track_id, box, timestamp, embedding=None):
        self.id = track_id
        self.box = box  # [x1,y1,x2,y2]
        self.last_box = box
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.misses = 0
        self.recognized = False
        self.name = None
        self.recog_prob = 0.0
        self.saved = 0
        self.last_save_time = 0.0
        self.embeddings = []  # recent embeddings
        if embedding is not None:
            self.embeddings.append(np.array(embedding, dtype=np.float32))

    def update(self, box, timestamp, embedding=None):
        self.last_box = self.box
        self.box = box
        self.last_seen = timestamp
        self.misses = 0
        if embedding is not None:
            self.embeddings.append(np.array(embedding, dtype=np.float32))
            if len(self.embeddings) > 8:
                self.embeddings.pop(0)

    def mark_missed(self):
        self.misses += 1


next_track_id = 1
tracks: Dict[int, Track] = {}
tracks_lock = threading.Lock()


def match_and_update(detections: List[Tuple[List[int], np.ndarray]], timestamp: float):
    """
    detections: list of (box, embedding_or_None)
    Greedy IOU matching between existing tracks and new detections.
    """
    global next_track_id, tracks
    with tracks_lock:
        # build list of unmatched detections/tracks
        if len(tracks) == 0:
            # create new tracks for all
            for box, emb in detections:
                tracks[next_track_id] = Track(next_track_id, box, timestamp, embedding=emb)
                next_track_id += 1
            return

        track_ids = list(tracks.keys())
        track_boxes = [tracks[tid].box for tid in track_ids]
        det_boxes = [d[0] for d in detections]

        iou_matrix = np.zeros((len(track_boxes), len(det_boxes)), dtype=np.float32)
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(det_boxes):
                iou_matrix[i, j] = iou(tb, db)

        # greedy matching: find max, if > threshold assign, repeat
        assigned_tracks = set()
        assigned_dets = set()
        while True:
            i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            maxv = iou_matrix[i, j]
            if maxv < IOU_MATCH_THRESHOLD:
                break
            tid = track_ids[i]
            if tid in assigned_tracks or j in assigned_dets:
                iou_matrix[i, j] = -1
                continue
            # match
            det_box, det_emb = detections[j]
            tracks[tid].update(det_box, timestamp, embedding=det_emb)
            assigned_tracks.add(tid)
            assigned_dets.add(j)
            # invalidate row/col
            iou_matrix[i, :] = -1
            iou_matrix[:, j] = -1

        # unmatched detections -> new tracks
        for j, (dbox, demb) in enumerate(detections):
            if j in assigned_dets:
                continue
            tracks[next_track_id] = Track(next_track_id, dbox, timestamp, embedding=demb)
            next_track_id += 1

        # unmatched tracks -> mark missed
        for tid in track_ids:
            if tid not in assigned_tracks:
                tracks[tid].mark_missed()

        # prune old tracks
        to_delete = []
        for tid, tr in tracks.items():
            if tr.misses > MAX_MISSES:
                to_delete.append(tid)
        for tid in to_delete:
            del tracks[tid]


# ---------------- Recognition helper ----------------
def recognize_embedding(embedding: np.ndarray) -> Tuple[str, float]:
    """
    Compare embedding to known embeddings using cosine similarity.
    Return (name, score) if score > threshold else (None, score).
    We convert cosine similarity into 0..1-ish score.
    """
    if len(known_embeddings) == 0:
        return None, 0.0
    emb = embedding.astype(np.float32)
    # normalize
    def norm(v): return v / (np.linalg.norm(v) + 1e-8)
    emb_n = norm(emb)
    sims = []
    for k in known_embeddings:
        sims.append(float(np.dot(emb_n, norm(k))))
    best_idx = int(np.argmax(sims))
    best_sim = sims[best_idx]
    # similarity roughly [-1,1]; map to [0,1]
    prob = (best_sim + 1.0) / 2.0
    # require high threshold (e.g. 0.85)
    if prob > 0.85:
        return known_names[best_idx], prob
    return None, prob


# ---------------- Saving unknown record ----------------
def save_unknown_sample(track: Track, frame):
    # folder per track
    folder = os.path.join(UNKNOWN_DIR, f"track_{track.id}")
    os.makedirs(folder, exist_ok=True)
    existing = len([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
    if existing >= MAX_SAVE_PHOTOS:
        return False
    # crop bbox and save
    crop = crop_box_safe(frame, track.box)
    if crop is None:
        return False
    fname = os.path.join(folder, f"{existing+1}.jpg")
    cv2.imwrite(fname, crop)
    # update sqlite (insert or update)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT id FROM unknowns WHERE track_tag=?", (f"track_{track.id}",))
    row = cur.fetchone()
    now = time.time()
    if row is None:
        cur.execute("INSERT INTO unknowns (track_tag, first_seen, last_seen, saved_count, assigned_name, sample_photo) VALUES (?, ?, ?, ?, ?, ?)",
                    (f"track_{track.id}", track.first_seen, now, 1, None, fname))
    else:
        uid = row[0]
        cur.execute("UPDATE unknowns SET last_seen=?, saved_count=saved_count+1, sample_photo=? WHERE id=?",
                    (now, fname, uid))
    con.commit()
    con.close()
    print(f"[SAVE] Unknown track_{track.id} -> {fname}")
    return True


# ---------------- Threaded capture + processing ----------------
frame_q = queue.Queue(maxsize=FRAME_QUEUE_MAX)
display_q = queue.Queue(maxsize=1)
stop_event = threading.Event()

def capture_thread_fn(urls: List[str]):
    cap = None
    # try urls in order
    for url in urls:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print("[CAPTURE] connected to", url)
            break
    if cap is None or not cap.isOpened():
        print("[CAPTURE] Failed to open any stream. Trying default webcam.")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[CAPTURE] No camera available. Exiting capture thread.")
            stop_event.set()
            return

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret or frame is None:
            # small sleep and retry
            time.sleep(0.1)
            continue
        # push latest frame (drop older)
        try:
            if frame_q.full():
                try:
                    frame_q.get_nowait()
                except queue.Empty:
                    pass
            frame_q.put_nowait(frame)
        except queue.Full:
            pass
    cap.release()


def process_thread_fn():
    executor = ThreadPoolExecutor(max_workers=2)  # for async recognition if needed
    while not stop_event.is_set():
        try:
            frame = frame_q.get(timeout=0.5)
        except queue.Empty:
            continue
        t0 = time.time()
        # InsightFace detection + recog (we'll extract faces and embeddings)
        faces = fa.get(frame)  # list of Face objects with bbox, embedding, det_score etc.
        detections = []
        for f in faces:
            # bbox: [x1,y1,x2,y2]
            bbox = [float(f.bbox[0]), float(f.bbox[1]), float(f.bbox[2]), float(f.bbox[3])]
            # filter too small
            if rect_area(bbox) < FACE_MIN_AREA:
                continue
            emb = None
            if hasattr(f, 'embedding') and f.embedding is not None:
                emb = np.array(f.embedding, dtype=np.float32)
            detections.append((bbox, emb))

        # match + update tracks
        match_and_update(detections, time.time())

        # For every track, decide recognition or saving unknown
        with tracks_lock:
            for tid, tr in list(tracks.items()):
                now = time.time()
                # compute average embedding if we have some
                emb = None
                if len(tr.embeddings) > 0:
                    emb = np.mean(np.stack(tr.embeddings, axis=0), axis=0)
                # If not yet recognized and have embedding -> try recognize (offload to executor)
                if not tr.recognized and emb is not None:
                    # recognize synchronously (fast)
                    name, prob = recognize_embedding(emb)
                    if name:
                        tr.recognized = True
                        tr.name = name
                        tr.recog_prob = prob
                        print(f"[RECOGNIZED] track_{tid} -> {name} ({prob:.2f})")
                # If not recognized and timeout exceeded -> save periodically
                if not tr.recognized:
                    if now - tr.first_seen > RECOGNITION_TIMEOUT:
                        if (now - tr.last_save_time) > SAVE_INTERVAL and tr.saved < MAX_SAVE_PHOTOS:
                            saved = save_unknown_sample(tr, frame)
                            if saved:
                                tr.saved += 1
                                tr.last_save_time = now
                # update DB last_seen for recognized ones or unknowns
                # (optional: implement DB updates here)

        # Build display frame with annotations (draw all current tracks)
        disp = frame.copy()
        with tracks_lock:
            for tid, tr in tracks.items():
                x1, y1, x2, y2 = map(int, tr.box)
                color = (0,255,0) if tr.recognized else (0,0,255)
                label = f"#{tid}"
                if tr.recognized:
                    label = f"{tr.name} ({tr.recog_prob*100:.1f}%)"
                # draw
                cv2.rectangle(disp, (x1,y1), (x2,y2), color, 2)
                cv2.putText(disp, label, (x1, max(15,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # put to display queue (single latest)
        try:
            if display_q.full():
                _ = display_q.get_nowait()
            display_q.put_nowait(disp)
        except queue.Full:
            pass

    executor.shutdown()


# ---------------- Main GUI loop ----------------
def main():
    cap_thread = threading.Thread(target=capture_thread_fn, args=(RTSP_URLS,), daemon=True)
    proc_thread = threading.Thread(target=process_thread_fn, daemon=True)
    cap_thread.start()
    proc_thread.start()

    print("[MAIN] Нажмите Space — пауза/воспроизведение, Q — выход")
    paused = False

    last_frame_time = time.time()
    while True:
        if not paused:
            try:
                disp = display_q.get(timeout=1.0)
                last_frame_time = time.time()
            except queue.Empty:
                # nothing to display, continue
                disp = None
        else:
            disp = None

        if disp is not None:
            h, w = disp.shape[:2]
            ws = int(w * DISPLAY_SCALE)
            hs = int(h * DISPLAY_SCALE)
            out = cv2.resize(disp, (ws, hs))
            cv2.imshow("FaceID (optimized)", out)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            paused = not paused
            print("[MAIN] Paused" if paused else "[MAIN] Resume")
        elif key == ord('q') or key == 27:
            print("[MAIN] Exit requested.")
            break

        # safety: stop if no frames for long
        if time.time() - last_frame_time > 30:
            print("[MAIN] No frames displayed for 30s -> exiting.")
            break

    stop_event.set()
    proc_thread.join(timeout=2.0)
    cap_thread.join(timeout=2.0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
