"""
webcam_test.py
==============
Live webcam inference using trained YOLO11 Model 1 (irregular_specialized).

Controls:
  Q        — Quit
  S        — Save screenshot
  +/-      — Increase/decrease confidence threshold
  1-5      — Switch model (if weights exist)

Run:
    python webcam_test.py
    python webcam_test.py --model small_parts       (when ready tomorrow)
    python webcam_test.py --conf 0.4 --cam 0
"""
import sys, cv2, time, argparse
from pathlib import Path
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Available models ──────────────────────────────────────────────────────────
MODELS = {
    "irregular_specialized": "runs/detect/runs/train/irregular_specialized_v1/weights/best.pt",
    "small_parts":           "runs/detect/runs/train/small_parts_v1/weights/best.pt",
    "mro_tools":             "runs/detect/runs/train/mro_tools_v1/weights/best.pt",
    "medical_pharma":        "runs/detect/runs/train/medical_pharma_v1/weights/best.pt",
    "heavy_materials":       "runs/detect/runs/train/heavy_materials_v1/weights/best.pt",
}

# Distinct colors per class (BGR)
COLORS = [
    (255, 87, 51),  (51, 255, 87),  (51, 87, 255),  (255, 215, 0),
    (0, 255, 255),  (255, 0, 255),  (128, 255, 0),   (255, 128, 0),
    (0, 128, 255),  (128, 0, 255),
]


def draw_box(frame, box, label, conf, color):
    """Draw a bounding box with label and confidence."""
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = f"{label} {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    # Label background
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def draw_hud(frame, model_name, conf_thresh, fps, detections):
    """Draw HUD overlay at top of frame."""
    h, w = frame.shape[:2]
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(frame, f"Model: {model_name}", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 100), 2)
    cv2.putText(frame, f"Conf: {conf_thresh:.0%}  |  +/- to adjust",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    cv2.putText(frame, f"Objects: {detections}", (w - 120, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    # Bottom instructions
    cv2.putText(frame, "Q=Quit  S=Screenshot  +/-=Confidence", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


def run(model_name, conf_thresh, cam_id):
    from ultralytics import YOLO

    weights = MODELS[model_name]
    if not Path(weights).exists():
        print(f"[ERR] Weights not found: {weights}")
        print(f"      Train the model first with: python train_ensemble.py --mode single --model {model_name}")
        sys.exit(1)

    print(f"\n[OK] Loading model: {model_name}")
    print(f"     Weights: {weights}")
    model = YOLO(weights)
    names = model.names
    print(f"[OK] Classes ({len(names)}): {list(names.values())}")

    print(f"\n[OK] Trying cameras...")
    cap = None
    # Auto-detect: try the requested cam_id first, then others
    for try_id in [cam_id] + [i for i in range(4) if i != cam_id]:
        print(f"     Trying camera {try_id}...", end=" ", flush=True)
        c = cv2.VideoCapture(try_id, cv2.CAP_DSHOW)  # CAP_DSHOW = faster on Windows
        if c.isOpened():
            ret, _ = c.read()
            if ret:
                print(f"OK!")
                cap = c
                cam_id = try_id
                break
            c.release()
        print("failed")
    if cap is None:
        print(f"[ERR] No working camera found (tried 0-3). Check your webcam is connected.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print(f"[OK] Camera opened! Resolution: {int(cap.get(3))}x{int(cap.get(4))}")
    print(f"\n  Controls: Q=Quit  S=Screenshot  +/-=Confidence threshold\n")

    os_screenshots = Path("screenshots")
    os_screenshots.mkdir(exist_ok=True)

    fps_history = []
    t_prev = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERR] Frame read failed")
            break

        # Run inference
        results = model(frame, conf=conf_thresh, verbose=False)[0]

        # Draw detections
        det_count = 0
        if results.boxes is not None:
            for box in results.boxes:
                cls_id  = int(box.cls[0])
                conf    = float(box.conf[0])
                label   = names[cls_id]
                color   = COLORS[cls_id % len(COLORS)]
                draw_box(frame, box.xyxy[0], label, conf, color)
                det_count += 1

        # FPS
        t_now = time.time()
        fps_history.append(1.0 / max(t_now - t_prev, 1e-6))
        if len(fps_history) > 15:
            fps_history.pop(0)
        fps = sum(fps_history) / len(fps_history)
        t_prev = t_now

        draw_hud(frame, model_name, conf_thresh, fps, det_count)
        cv2.imshow(f"YOLO11 — {model_name}", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:   # Q or ESC
            print("\n[OK] Quit.")
            break
        elif key == ord('s'):               # S = screenshot
            fname = os_screenshots / f"{model_name}_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(str(fname), frame)
            print(f"[OK] Screenshot saved: {fname}")
        elif key == ord('+') or key == ord('='):
            conf_thresh = min(conf_thresh + 0.05, 0.95)
            print(f"     Confidence → {conf_thresh:.0%}")
        elif key == ord('-'):
            conf_thresh = max(conf_thresh - 0.05, 0.05)
            print(f"     Confidence → {conf_thresh:.0%}")

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Live webcam YOLO11 inference")
    parser.add_argument("--model", choices=list(MODELS.keys()),
                        default="irregular_specialized",
                        help="Which model to run (default: irregular_specialized)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold 0–1 (default: 0.35)")
    parser.add_argument("--cam", type=int, default=1,
                        help="Camera device ID (default: 1)")
    args = parser.parse_args()

    run(args.model, args.conf, args.cam)


if __name__ == "__main__":
    main()
