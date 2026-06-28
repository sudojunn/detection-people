import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

try:
    import winsound
except ImportError:
    winsound = None


MODEL_PATH = "runs/detect/danger_zone/yolo26n_person_20260627_134601/weights/best.pt"
ALARM_COOLDOWN_SEC = 1.5

BOX_COLOR = (29, 158, 117)
ZONE_COLOR = (42, 176, 255)
ALERT_COLOR = (36, 36, 230)
LABEL_COLOR = (255, 255, 255)
BG = "#0f0f0f"
SURFACE = "#1a1a1a"
ACCENT = "#1D9E75"
WARN = "#E24B4A"
ZONE_TEXT = "#FFB02A"
TEXT = "#e8e8e8"
MUTED = "#888780"

WINDOW_WIDTH = 1220
WINDOW_HEIGHT = 760


state = {
    "model": None,
    "video_path": None,
    "cap": None,
    "worker": None,
    "stop_event": threading.Event(),
    "running": False,
    "paused": False,
    "current_frame": None,
    "current_frame_index": 0,
    "total_frames": 0,
    "seek_request": None,
    "updating_seek": False,
    "playback_speed": 1.0,
    "result_img": None,
    "tk_img": None,
    "display": None,
    "zones": [],            
    "zone_points": [],      
    "zones_config": [4],    
    "target_zones_count": 1,
    "setting_zone": False,
    "breach_inside": False,
    "alarm_enabled": False, 
    "last_alarm_at": 0.0,
}

widgets = {}


def load_model():
    try:
        state["model"] = YOLO(MODEL_PATH)
        root.after(0, lambda: set_status("модель готова", ACCENT))
    except Exception as exc:
        root.after(0, lambda: set_status(f"ошибка модели: {exc}", WARN))


def set_status(text, color=MUTED):
    widgets["status_label"].config(text=text, fg=color)


def format_time(seconds):
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def update_seek_label(frame_index=None):
    total_frames = state["total_frames"]
    fps = state.get("fps") or 25
    if frame_index is None:
        frame_index = state["current_frame_index"]

    current = format_time(frame_index / fps)
    total = format_time(total_frames / fps) if total_frames else "--:--"
    widgets["seek_label"].config(text=f"{current} / {total}")


def play_alarm():
    if not state["alarm_enabled"]:
        return

    def worker():
        if winsound:
            winsound.Beep(1200, 180)
            winsound.Beep(900, 180)
        else:
            root.after(0, root.bell)

    threading.Thread(target=worker, daemon=True).start()


def draw_label(img, text, origin, color):
    scale = max(img.shape[:2]) / 900
    fs = max(0.45, 0.45 * scale)
    ft = max(1, int(scale))
    pad = max(4, int(4 * scale))
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, ft)
    y1 = max(0, y - th - pad * 2)
    y2 = y1 + th + pad * 2
    cv2.rectangle(img, (x, y1), (x + tw + pad * 2, y2), color, -1)
    cv2.putText(
        img,
        text,
        (x + pad, y2 - pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        fs,
        LABEL_COLOR,
        ft,
        cv2.LINE_AA,
    )


def draw_person_box(img, box, idx, color=BOX_COLOR):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    conf = float(box.conf[0])
    scale = max(img.shape[:2]) / 900
    lw = max(2, int(2 * scale))
    corner = max(12, int(18 * scale))
    cw = max(2, lw + 1)

    # Отрисовка основной рамки детекции
    cv2.rectangle(img, (x1, y1), (x2, y2), color, lw)
    for cx_point, cy, dx, dy in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        cv2.line(img, (cx_point, cy), (cx_point + dx * corner, cy), color, cw)
        cv2.line(img, (cx_point, cy), (cx_point, cy + dy * corner), color, cw)

    draw_label(img, f"person #{idx} {conf:.0%}", (x1, y1), color)
    
    # Визуализация датчиков в зависимости от режима (Весь бокс или Нижняя грань)
    if widgets["full_box_var"].get():
        # Подсветка внутреннего периметра рамки (весь прямоугольник — датчик)
        cv2.rectangle(img, (x1 + 3, y1 + 3), (x2 - 3, y2 - 3), (255, 255, 0), 1)
    else:
        # Расчет динамической длины линии-датчика на основе ползунка
        z_scale = widgets["zone_scale_var"].get()
        cx = (x1 + x2) / 2
        hw_adj = ((x2 - x1) / 2) * z_scale
        bl_x = int(cx - hw_adj)
        br_x = int(cx + hw_adj)
        
        if z_scale > 0:
            cv2.line(img, (bl_x, y2), (br_x, y2), (255, 255, 0), lw + 1)
        else:
            cv2.circle(img, (int(cx), y2), lw + 2, (255, 255, 0), -1)


def draw_zones(img):
    for z_idx, zone in enumerate(state["zones"], start=1):
        pts = np.array(zone, dtype=np.int32)
        cv2.polylines(img, [pts], True, ZONE_COLOR, 2)
        
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], ZONE_COLOR)
        cv2.addWeighted(overlay, 0.15, img, 0.85, 0, img)
        
        if len(zone) > 0:
            cv2.putText(
                img,
                f"Zone {z_idx}",
                (zone[0][0], zone[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                ZONE_COLOR,
                2,
                cv2.LINE_AA
            )

    points = state["zone_points"]
    curr_zone_idx = len(state["zones"])
    
    if not points:
        return

    if state["setting_zone"] and curr_zone_idx < len(state["zones_config"]):
        max_pts = state["zones_config"][curr_zone_idx]
    else:
        max_pts = 4

    for idx, point in enumerate(points, start=1):
        cv2.circle(img, point, 6, ZONE_COLOR, -1)
        cv2.putText(
            img,
            str(idx),
            (point[0] + 8, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ZONE_COLOR,
            2,
            cv2.LINE_AA,
        )

    if len(points) >= 2:
        cv2.polylines(img, [np.array(points, dtype=np.int32)], len(points) == max_pts, ZONE_COLOR, 2)


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def segments_intersect(a, b, c, d):
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


def box_in_any_zone(box):
    if not state["zones"]:
        return False

    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    # РЕЖИМ 1: Детекция по всему прямоугольнику (Камера сверху)
    if widgets["full_box_var"].get():
        corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        box_edges = [(corners[0], corners[1]), (corners[1], corners[2]), 
                     (corners[2], corners[3]), (corners[3], corners[0])]
                     
        for zone in state["zones"]:
            polygon = np.array(zone, dtype=np.int32)
            # Проверка: находится ли хоть один угол рамки внутри зоны
            if any(cv2.pointPolygonTest(polygon, pt, False) >= 0 for pt in corners):
                return True
            # Проверка: пересекает ли хоть одна грань рамки границы зоны
            zone_edges = list(zip(zone, zone[1:] + zone[:1]))
            if any(segments_intersect(b_p1, b_p2, z_p1, z_p2) 
                   for b_p1, b_p2 in box_edges for z_p1, z_p2 in zone_edges):
                return True
        return False

    # РЕЖИМ 2: Динамический сегментарный контроль нижней грани
    else:
        z_scale = widgets["zone_scale_var"].get()
        cx = (x1 + x2) / 2
        hw_adj = ((x2 - x1) / 2) * z_scale
        bottom_left = (int(cx - hw_adj), y2)
        bottom_right = (int(cx + hw_adj), y2)

        for zone in state["zones"]:
            polygon = np.array(zone, dtype=np.int32)
            if cv2.pointPolygonTest(polygon, bottom_left, False) >= 0:
                return True
            if cv2.pointPolygonTest(polygon, bottom_right, False) >= 0:
                return True
            zone_edges = list(zip(zone, zone[1:] + zone[:1]))
            if any(segments_intersect(bottom_left, bottom_right, z_p1, z_p2) for z_p1, z_p2 in zone_edges):
                return True
        return False


def analyze_frame(frame):
    annotated = frame.copy()
    draw_zones(annotated)

    if state["model"] is None:
        return annotated, "модель ещё загружается", False

    result = state["model"](frame, conf=widgets["conf_var"].get(), verbose=False)[0]
    person_count = len(result.boxes) if result.boxes is not None else 0

    body_inside_now = False
    if result.boxes is not None:
        for idx, box in enumerate(result.boxes, start=1):
            body_inside = box_in_any_zone(box)
            body_inside_now = body_inside_now or body_inside
            draw_person_box(annotated, box, idx, ALERT_COLOR if body_inside else BOX_COLOR)

    alert = False
    if state["zones"] and body_inside_now and not state["breach_inside"]:
        now = time.monotonic()
        if now - state["last_alarm_at"] >= ALARM_COOLDOWN_SEC:
            state["last_alarm_at"] = now
            alert = True
            play_alarm()
            
            current_time = time.strftime("%Y-%m-%d %H:%M:%S")
            video_name = Path(state["video_path"]).name if state["video_path"] else "Неизвестная камера"
            log_message = f"[{current_time}] Камера: {video_name} | ТРЕВОГА: Человек в опасной зоне!\n"
            
            try:
                log_path = Path(__file__).parent / "log.txt"
                with open(log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(log_message)
            except Exception as e:
                print(f"Ошибка записи лога: {e}")

    state["breach_inside"] = body_inside_now

    zone_count = len(state["zones"])
    if state["setting_zone"]:
        curr_zone_idx = zone_count
        if curr_zone_idx < len(state["zones_config"]):
            max_pts = state["zones_config"][curr_zone_idx]
            zone_state = f"рисуем зону {curr_zone_idx + 1} ({len(state['zone_points'])}/{max_pts})"
        else:
            zone_state = f"зон задано: {zone_count}"
    else:
        zone_state = f"зон задано: {zone_count}"
        
    breach_state = "ТРЕВОГА: человек в зоне" if body_inside_now else "зоны свободны"
    info = f"людей: {person_count} | {zone_state} | {breach_state}"

    return annotated, info, alert


def show_frame(img_bgr, info_text, frame_index=None):
    widgets["progress"].stop()
    widgets["progress"].pack_forget()

    canvas = widgets["canvas"]
    cw = canvas.winfo_width() or 760
    ch = canvas.winfo_height() or 500
    ih, iw = img_bgr.shape[:2]
    ratio = min(cw / iw, ch / ih, 1.0)
    dw = max(1, int(iw * ratio))
    dh = max(1, int(ih * ratio))
    dx = (cw - dw) // 2
    dy = (ch - dh) // 2

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb).resize((dw, dh), Image.LANCZOS)

    state["tk_img"] = ImageTk.PhotoImage(pil_img)
    state["display"] = {"x": dx, "y": dy, "w": dw, "h": dh, "scale": ratio}

    canvas.delete("all")
    canvas.create_image(dx, dy, anchor="nw", image=state["tk_img"])
    canvas.config(scrollregion=canvas.bbox("all"))
    widgets["info_label"].config(text=info_text)
    if frame_index is not None:
        state["current_frame_index"] = frame_index
        if state["total_frames"]:
            state["updating_seek"] = True
            widgets["seek_var"].set(frame_index)
            state["updating_seek"] = False
        update_seek_label(frame_index)

    if "ТРЕВОГА" in info_text:
        set_status("тревога", WARN)
    elif state["running"]:
        set_status("пауза" if state["paused"] else "видео работает", MUTED if state["paused"] else ACCENT)


def process_frame(frame, frame_index):
    state["current_frame"] = frame.copy()
    state["current_frame_index"] = frame_index
    annotated, info, _ = analyze_frame(frame)
    state["result_img"] = annotated
    root.after(0, lambda img=annotated.copy(), text=info, idx=frame_index: show_frame(img, text, idx))


def render_current_frame():
    frame = state.get("current_frame")
    if frame is None:
        return

    annotated, info, _ = analyze_frame(frame)
    state["result_img"] = annotated
    show_frame(annotated, info, state["current_frame_index"])


def configure_seek(total_frames, fps):
    state["total_frames"] = total_frames
    state["fps"] = fps
    widgets["seek_scale"].config(to=max(0, total_frames - 1), state="normal" if total_frames else "disabled")
    state["updating_seek"] = True
    widgets["seek_var"].set(0)
    state["updating_seek"] = False
    update_seek_label(0)


def video_loop(path, stop_event):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        root.after(0, lambda: set_status("не удалось открыть видео", WARN))
        return

    state["cap"] = cap
    state["running"] = True
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    state["fps"] = fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    root.after(0, lambda total=total_frames, video_fps=fps: configure_seek(total, video_fps))
    frame_delay = 1.0 / max(1, min(fps, 30))

    while not stop_event.is_set():
        seek_to = state["seek_request"]
        if seek_to is not None:
            state["seek_request"] = None
            seek_to = max(0, min(total_frames - 1, int(seek_to))) if total_frames else max(0, int(seek_to))
            cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
            state["breach_inside"] = False
            ok, frame = cap.read()
            if ok:
                process_frame(frame, seek_to)
            continue

        if state["paused"] and state["current_frame"] is not None:
            time.sleep(0.05)
            continue

        started = time.monotonic()
        ok, frame = cap.read()
        if not ok:
            state["paused"] = True
            root.after(0, lambda: widgets["pause_btn"].config(text="продолжить"))
            root.after(0, lambda: set_status("видео закончено", MUTED))
            time.sleep(0.05)
            continue

        frame_index = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
        process_frame(frame, frame_index)

        speed = max(0.25, float(state["playback_speed"]))
        elapsed = time.monotonic() - started
        time.sleep(max(0.001, (frame_delay / speed) - elapsed))

    cap.release()
    if state.get("cap") is cap:
        state["cap"] = None
        state["running"] = False


def stop_video():
    state["stop_event"].set()
    state["running"] = False
    state["paused"] = False
    state["seek_request"] = None
    widgets["pause_btn"].config(text="пауза")
    cap = state.get("cap")
    if cap is not None:
        cap.release()


def show_zone_config_dialog():
    if state["current_frame"] is None:
        set_status("сначала откройте видео", MUTED)
        return

    dialog = tk.Toplevel(root)
    dialog.title("Настройка опасных зон")
    dialog.configure(bg=BG)
    dialog.geometry("380x220")
    dialog.resizable(False, False)
    dialog.transient(root)
    dialog.grab_set()
    
    dialog.update_idletasks()
    rx = root.winfo_x() + (root.winfo_width() // 2) - (dialog.winfo_width() // 2)
    ry = root.winfo_y() + (root.winfo_height() // 2) - (dialog.winfo_height() // 2)
    dialog.geometry(f"+{rx}+{ry}")

    tk.Label(dialog, text="ПАРАМЕТРЫ ОПАСНЫХ ЗОН", bg=BG, fg=TEXT, font=("Courier", 12, "bold")).pack(pady=(15, 10))
    
    f1 = tk.Frame(dialog, bg=BG)
    f1.pack(pady=5)
    tk.Label(f1, text="Кол-во зон:", bg=BG, fg=MUTED, font=("Courier", 11), width=15, anchor="e").pack(side="left")
    
    zones_var = tk.IntVar(value=state.get("target_zones_count", 1))
    rows_frame = tk.Frame(dialog, bg=BG)
    rows_frame.pack(fill="x", pady=5)
    
    zone_vars = []
    
    def update_dynamic_rows(*args):
        try:
            n_zones = int(zones_var.get())
        except (ValueError, tk.TclError):
            return
        
        if n_zones < 1: n_zones = 1
        if n_zones > 10: n_zones = 10  
        
        for widget in rows_frame.winfo_children():
            widget.destroy()
        zone_vars.clear()
        
        for i in range(n_zones):
            row = tk.Frame(rows_frame, bg=BG)
            row.pack(pady=2)
            
            tk.Label(row, text=f"{i+1} зона | точек: ", bg=BG, fg=MUTED, font=("Courier", 10)).pack(side="left")
            
            default_val = state["zones_config"][i] if i < len(state["zones_config"]) else 4
            v = tk.IntVar(value=default_val)
            zone_vars.append(v)
            
            tk.Spinbox(
                row, 
                from_=3, 
                to=25, 
                textvariable=v, 
                width=5, 
                bg=SURFACE, 
                fg=TEXT, 
                bd=0, 
                font=("Courier", 10), 
                justify="center", 
                buttonbackground=SURFACE
            ).pack(side="left", padx=5)
            
        dialog.geometry(f"380x{140 + n_zones * 32}")

    zones_var.trace_add("write", update_dynamic_rows)
    
    tk.Spinbox(
        f1, 
        from_=1, 
        to=10, 
        textvariable=zones_var, 
        width=5, 
        bg=SURFACE, 
        fg=TEXT, 
        bd=0, 
        font=("Courier", 11), 
        justify="center", 
        buttonbackground=SURFACE,
        command=update_dynamic_rows
    ).pack(side="left", padx=5)
    
    update_dynamic_rows()
    
    def on_ok():
        try:
            state["target_zones_count"] = max(1, min(10, int(zones_var.get())))
            state["zones_config"] = [max(3, int(v.get())) for v in zone_vars]
        except ValueError:
            state["target_zones_count"] = 1
            state["zones_config"] = [4]
        
        dialog.destroy()
        start_zone_setup_auto()
        
    btn = tk.Button(
        dialog,
        text="ПРИМЕНИТЬ",
        command=on_ok,
        bg=SURFACE,
        fg=TEXT,
        relief="flat",
        font=("Courier", 11, "bold"),
        activebackground=ACCENT,
        activeforeground="white",
        width=12,
        pady=4,
        cursor="hand2"
    )
    btn.pack(side="bottom", pady=10)


def open_video():
    if state["model"] is None:
        set_status("модель ещё загружается", MUTED)
        return

    path = filedialog.askopenfilename(
        title="Выберите видео",
        filetypes=[
            ("Видео", "*.mp4 *.avi *.mov *.mkv *.webm"),
            ("Все файлы", "*.*"),
        ],
    )
    if not path:
        return

    stop_video()
    state["stop_event"] = threading.Event()
    state["video_path"] = path
    state["breach_inside"] = False
    state["current_frame"] = None
    state["current_frame_index"] = 0
    state["total_frames"] = 0
    state["seek_request"] = None
    state["paused"] = True
    widgets["save_btn"].config(state="normal", fg=TEXT)
    widgets["pause_btn"].config(text="продолжить")
    widgets["seek_scale"].config(state="disabled")
    update_seek_label(0)
    widgets["progress"].pack(fill="x", padx=16, pady=2)
    widgets["progress"].start(12)
    set_status("запускаю видео на паузе", MUTED)

    state["worker"] = threading.Thread(target=video_loop, args=(path, state["stop_event"]), daemon=True)
    state["worker"].start()
    
    root.after(200, show_zone_config_dialog)


def toggle_pause():
    if not state["running"]:
        return

    state["paused"] = not state["paused"]
    widgets["pause_btn"].config(text="продолжить" if state["paused"] else "пауза")
    set_status("пауза" if state["paused"] else "видео работает", MUTED if state["paused"] else ACCENT)


def toggle_alarm():
    state["alarm_enabled"] = not state["alarm_enabled"]
    if state["alarm_enabled"]:
        widgets["alarm_btn"].config(text="звук: вкл", fg=ACCENT)
        set_status("звуковой сигнал включен", ACCENT)
    else:
        widgets["alarm_btn"].config(text="звук: выкл", fg=MUTED)
        set_status("звуковой сигнал выключен", MUTED)


def start_zone_setup_auto():
    state["zones"] = []
    state["zone_points"] = []
    state["setting_zone"] = True
    state["breach_inside"] = False
    
    first_max = state["zones_config"][0] if state["zones_config"] else 4
    set_status(f"зона 1: кликните {first_max} точек", ZONE_TEXT)
    render_current_frame()


def reset_zones():
    state["zones"] = []
    state["zone_points"] = []
    state["setting_zone"] = False
    state["breach_inside"] = False
    set_status("все zones сброшены", MUTED)
    render_current_frame()


def canvas_to_frame(x, y):
    display = state.get("display")
    frame = state.get("current_frame")
    if not display or frame is None:
        return None

    if not (display["x"] <= x <= display["x"] + display["w"] and display["y"] <= y <= display["y"] + display["h"]):
        return None

    fx = int((x - display["x"]) / display["scale"])
    fy = int((y - display["y"]) / display["scale"])
    h, w = frame.shape[:2]
    return max(0, min(w - 1, fx)), max(0, min(h - 1, fy))


def on_canvas_click(event):
    if not state["setting_zone"]:
        open_video()
        return

    point = canvas_to_frame(event.x, event.y)
    if point is None:
        return

    curr_zone_idx = len(state["zones"])
    target_zones = state.get("target_zones_count", 1)
    
    if curr_zone_idx >= len(state["zones_config"]):
        state["setting_zone"] = False
        return

    max_pts = state["zones_config"][curr_zone_idx]
    state["zone_points"].append(point)
    
    if len(state["zone_points"]) >= max_pts:
        state["zones"].append(list(state["zone_points"]))
        state["zone_points"] = []
        
        if len(state["zones"]) >= target_zones:
            state["setting_zone"] = False
            set_status(f"все зоны ({len(state['zones'])}) заданы", ACCENT)
        else:
            next_zone = len(state["zones"]) + 1
            next_max = state["zones_config"][next_zone - 1]
            set_status(f"зона {len(state['zones'])} сохранена. Рисуйте зону {next_zone} ({next_max} точ.)", ZONE_TEXT)
    else:
        set_status(f"зона {curr_zone_idx + 1}: точка {len(state['zone_points'])}/{max_pts}", ZONE_TEXT)
        
    render_current_frame()


def on_seek_change(value):
    if state["updating_seek"] or not state["video_path"]:
        return

    frame_index = int(float(value))
    state["paused"] = True
    widgets["pause_btn"].config(text="продолжить")
    state["seek_request"] = frame_index
    state["breach_inside"] = False
    update_seek_label(frame_index)
    set_status("перемотка", MUTED)


def on_conf_change(_=None):
    state["breach_inside"] = False
    render_current_frame()


def on_zone_scale_change(_=None):
    state["breach_inside"] = False
    render_current_frame()


def on_full_box_toggle():
    state["breach_inside"] = False
    
    # Блокируем или разблокируем ползунок датчика нижней грани для наглядности интерфейса
    if widgets["full_box_var"].get():
        widgets["zone_scale_slider"].config(state="disabled", troughcolor="#222")
    else:
        widgets["zone_scale_slider"].config(state="normal", troughcolor="#333")
        
    render_current_frame()


def on_speed_change(value):
    speed = float(value)
    state["playback_speed"] = speed
    widgets["speed_label"].config(text=f"speed: {speed:.2g}x")


def on_resize(event):
    if state["result_img"] is not None and event.widget == root:
        show_frame(state["result_img"], widgets["info_label"].cget("text"), state["current_frame_index"])


def save_snapshot():
    if state["result_img"] is None:
        return

    stem = Path(state["video_path"]).stem if state["video_path"] else "danger_zone"
    path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        initialfile=f"{stem}_alert.jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
    )
    if path:
        cv2.imwrite(path, state["result_img"])
        filename = Path(path).name
        set_status(f"сохранено: {filename}", ACCENT)


def build_button(parent, text, command, width, fg=TEXT):
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=SURFACE,
        fg=fg,
        relief="flat",
        font=("Courier", 11),
        activebackground=ACCENT,
        activeforeground="white",
        width=width,
        pady=5,
        cursor="hand2",
    )
    btn.pack(side="left", padx=4)
    return btn


def build_ui():
    top = tk.Frame(root, bg=BG, pady=12, padx=16)
    top.pack(fill="x")

    tk.Label(top, text="DANGER ZONE DETECTOR", bg=BG, fg=TEXT, font=("Courier", 14, "bold")).pack(side="left")

    widgets["status_label"] = tk.Label(top, text="загрузка модели...", bg=BG, fg=MUTED, font=("Courier", 11))
    widgets["status_label"].pack(side="left", padx=20)

    btn_frame = tk.Frame(top, bg=BG)
    btn_frame.pack(side="right")

    build_button(btn_frame, "открыть видео", open_video, width=15)
    widgets["pause_btn"] = build_button(btn_frame, "пауза", toggle_pause, width=12)
    widgets["alarm_btn"] = build_button(btn_frame, "звук: выкл", toggle_alarm, width=12, fg=MUTED)

    build_button(btn_frame, "задать зону", show_zone_config_dialog, width=12)
    build_button(btn_frame, "сброс зоны", reset_zones, width=12, fg=MUTED)

    widgets["save_btn"] = build_button(btn_frame, "кадр", save_snapshot, width=8, fg=MUTED)
    widgets["save_btn"].config(state="disabled")

    tk.Frame(root, bg="#2a2a2a", height=1).pack(fill="x")

    main = tk.Frame(root, bg=BG)
    main.pack(fill="both", expand=True, padx=16, pady=12)

    canvas_frame = tk.Frame(main, bg=SURFACE, bd=0, highlightthickness=1, highlightbackground="#2a2a2a")
    canvas_frame.pack(fill="both", expand=True)

    widgets["canvas"] = tk.Canvas(canvas_frame, bg=SURFACE, bd=0, highlightthickness=0)
    widgets["canvas"].pack(fill="both", expand=True)
    
    def center_welcome_text(event):
        if state["current_frame"] is None:
            event.widget.delete("welcome_text")
            event.widget.create_text(
                event.width // 2,
                event.height // 2,
                text="Откройте видео, выберите параметры и задайте опасные зоны.",
                fill=MUTED,
                font=("Courier", 13),
                justify="center",
                tags="welcome_text"
            )
            
    widgets["canvas"].bind("<Configure>", center_welcome_text)
    widgets["canvas"].bind("<Button-1>", on_canvas_click)

    seek_frame = tk.Frame(root, bg=SURFACE, pady=6, padx=16)
    seek_frame.pack(fill="x")

    widgets["seek_label"] = tk.Label(seek_frame, text="00:00 / --:--", bg=SURFACE, fg=MUTED, font=("Courier", 10))
    widgets["seek_label"].pack(side="right", padx=(10, 0))

    widgets["seek_var"] = tk.DoubleVar(value=0)
    widgets["seek_scale"] = tk.Scale(
        seek_frame,
        from_=0,
        to=0,
        resolution=1,
        orient="horizontal",
        variable=widgets["seek_var"],
        bg=SURFACE,
        fg=TEXT,
        troughcolor="#333",
        highlightthickness=0,
        bd=0,
        showvalue=False,
        state="disabled",
        command=on_seek_change,
    )
    widgets["seek_scale"].pack(side="left", fill="x", expand=True)

    bottom = tk.Frame(root, bg=SURFACE, pady=6, padx=16)
    bottom.pack(fill="x")

    widgets["info_label"] = tk.Label(bottom, text="", bg=SURFACE, fg=MUTED, font=("Courier", 11))
    widgets["info_label"].pack(side="left")

    # Конфигурация порогов детектора
    widgets["conf_var"] = tk.DoubleVar(value=0.35)
    tk.Label(bottom, text="conf:", bg=SURFACE, fg=MUTED, font=("Courier", 11)).pack(side="right", padx=(0, 4))
    tk.Scale(
        bottom,
        from_=0.1,
        to=0.9,
        resolution=0.05,
        orient="horizontal",
        variable=widgets["conf_var"],
        bg=SURFACE,
        fg=TEXT,
        troughcolor="#333",
        highlightthickness=0,
        bd=0,
        font=("Courier", 10),
        length=100,
        command=on_conf_change,
    ).pack(side="right")

    # ПОЛЗУНОК НАСТРОЙКИ РАЗМЕРА ДАТЧИКА (ОТ ТОЧКИ В ЦЕНТРЕ ДО ВСЕЙ НИЖНЕЙ СТОРОНЫ)
    widgets["zone_scale_var"] = tk.DoubleVar(value=1.0)
    tk.Label(bottom, text="датчик:", bg=SURFACE, fg=MUTED, font=("Courier", 11)).pack(side="right", padx=(12, 4))
    widgets["zone_scale_slider"] = tk.Scale(
        bottom,
        from_=0.0,
        to=1.0,
        resolution=0.05,
        orient="horizontal",
        variable=widgets["zone_scale_var"],
        bg=SURFACE,
        fg=TEXT,
        troughcolor="#333",
        highlightthickness=0,
        bd=0,
        font=("Courier", 10),
        length=100,
        command=on_zone_scale_change,
    )
    widgets["zone_scale_slider"].pack(side="right")

    # ГАЛОЧКА ДЛЯ ВКЛЮЧЕНИЯ ДЕТЕКЦИИ ПО ВСЕМУ ПРЯМОУГОЛЬНИКУ ЧЕЛОВЕКА (КАМЕРА СВЕРХУ)
    widgets["full_box_var"] = tk.BooleanVar(value=False)
    widgets["full_box_check"] = tk.Checkbutton(
        bottom,
        text="весь бокс",
        variable=widgets["full_box_var"],
        bg=SURFACE,
        fg=TEXT,
        selectcolor=SURFACE,
        activebackground=SURFACE,
        activeforeground=TEXT,
        font=("Courier", 11),
        bd=0,
        highlightthickness=0,
        command=on_full_box_toggle
    )
    widgets["full_box_check"].pack(side="right", padx=(14, 4))

    widgets["speed_var"] = tk.DoubleVar(value=1.0)
    widgets["speed_label"] = tk.Label(bottom, text="speed: 1x", bg=SURFACE, fg=MUTED, font=("Courier", 11), width=11, anchor="w")
    widgets["speed_label"].pack(side="right", padx=(14, 4))
    tk.Scale(
        bottom,
        from_=0.25,
        to=4.0,
        resolution=0.25,
        orient="horizontal",
        variable=widgets["speed_var"],
        bg=SURFACE,
        fg=TEXT,
        troughcolor="#333",
        highlightthickness=0,
        bd=0,
        font=("Courier", 10),
        length=110,
        command=on_speed_change,
    ).pack(side="right")

    widgets["progress"] = ttk.Progressbar(root, mode="indeterminate", length=200)
    root.bind("<Configure>", on_resize)


def on_close():
    stop_video()
    root.destroy()


root = tk.Tk()
root.title("Danger Zone Detector")
root.configure(bg=BG)
root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
root.resizable(False, False)
root.protocol("WM_DELETE_WINDOW", on_close)

style = ttk.Style()
style.theme_use("clam")
style.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=BG, bordercolor=BG)

build_ui()
threading.Thread(target=load_model, daemon=True).start()
root.mainloop()