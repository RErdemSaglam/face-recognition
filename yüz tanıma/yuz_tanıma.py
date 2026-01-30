import cv2
from deepface import DeepFace
import os
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk

# =====================
# AYARLAR
# =====================
DB_PATH = "faces"
MODEL_NAME = "ArcFace"
THRESHOLD = 0.68
CHECK_INTERVAL = 2

# =====================
# GLOBAL
# =====================
running = False
cap = None
frame = None
name = "Bilinmiyor"
last_check = 0

# =====================
# KAMERA THREAD
# =====================
def camera_thread():
    global cap, frame, running, name, last_check

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        running = False
        return

    while running:
        ret, img = cap.read()
        if not ret:
            break

        now = time.time()

        if now - last_check > CHECK_INTERVAL:
            last_check = now
            try:
                results = DeepFace.find(
                    img_path=img,
                    db_path=DB_PATH,
                    model_name=MODEL_NAME,
                    distance_metric="cosine",
                    threshold=THRESHOLD,
                    enforce_detection=False,
                    silent=True
                )

                if isinstance(results, list) and len(results) > 0 and len(results[0]) > 0:
                    identity = results[0].iloc[0]["identity"]
                    name = os.path.splitext(os.path.basename(identity))[0]
                else:
                    name = "Bilinmiyor"
            except Exception:
                name = "Bilinmiyor"

        try:
            faces = DeepFace.extract_faces(img, enforce_detection=False)
            if len(faces) > 0:
                fa = faces[0]["facial_area"]
                x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception:
            pass

        frame = img

    cap.release()

# =====================
# UI GÜNCELLEME
# =====================
def update_ui():
    if frame is not None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    root.after(15, update_ui)

# =====================
# BUTONLAR
# =====================
def start_camera():
    global running
    if running:
        return
    running = True
    threading.Thread(target=camera_thread, daemon=True).start()

def stop_camera():
    global running
    running = False

def on_close():
    global running
    running = False
    root.destroy()

# =====================
# TKINTER
# =====================
root = tk.Tk()
root.title("Yüz Tanıma Sistemi")
root.geometry("800x600")
root.protocol("WM_DELETE_WINDOW", on_close)

video_label = tk.Label(root)
video_label.pack(pady=10)

btn_frame = tk.Frame(root)
btn_frame.pack()

tk.Button(btn_frame, text="Başlat", width=15, command=start_camera).pack(side=tk.LEFT, padx=10)
tk.Button(btn_frame, text="Durdur", width=15, command=stop_camera).pack(side=tk.LEFT, padx=10)

root.after(0, update_ui)
root.mainloop()
