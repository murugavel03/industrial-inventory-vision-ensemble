"""
=============================================================================
  YOLO11 ENSEMBLE TRAINING SCRIPT
  Trains all 5 specialized models sequentially with optimal hyperparameters.
  Uses transfer learning from yolo11n/s/m pretrained weights.
=============================================================================
USAGE:
    # Train all models with available data
    python train_ensemble.py --mode all

    # Train a single model
    python train_ensemble.py --mode single --model small_parts

    # Resume interrupted training
    python train_ensemble.py --mode resume --model irregular_specialized

REQUIREMENTS:
    pip install ultralytics torch torchvision
=============================================================================
"""
import sys, os, json, time, shutil, argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── MODEL TRAINING CONFIGS ──────────────────────────────────────────────────
# Tuned per model based on dataset size and object characteristics.
# yolo11n = nano (fastest), yolo11s = small, yolo11m = medium
TRAIN_CONFIGS = {
    "small_parts": {
        "data":        "datasets/small_parts_merged/data.yaml",
        "model":       "yolo11s.pt",        # small model for tiny objects
        "epochs":      150,
        "imgsz":       640,                # reduced from 1280 — RTX 2050 only has 4GB VRAM
        "batch":       4,                  # reduced from 8 to avoid OOM on 4GB GPU
        "patience":    30,
        "lr0":         0.01,
        "lrf":         0.001,
        "mosaic":      1.0,
        "mixup":       0.1,
        "degrees":     45.0,               # fasteners can appear at any angle
        "flipud":      0.5,
        "fliplr":      0.5,
        "hsv_h":       0.015,
        "hsv_s":       0.7,
        "hsv_v":       0.4,
        "project":     "runs/train",
        "name":        "small_parts_v1",
        "note":        "5/10 classes populated. Needs more washer/bolt_m10/m16 data.",
    },
    "mro_tools": {
        "data":        "datasets/mro_tools_merged/data.yaml",
        "model":       "yolo11s.pt",
        "epochs":      100,
        "imgsz":       640,
        "batch":       8,                  # reduced from 16 to avoid OOM on 4GB VRAM
        "patience":    20,
        "lr0":         0.01,
        "lrf":         0.001,
        "mosaic":      1.0,
        "mixup":       0.0,
        "degrees":     180.0,              # tools appear at all rotations
        "flipud":      0.5,
        "fliplr":      0.5,
        "hsv_h":       0.015,
        "hsv_s":       0.7,
        "hsv_v":       0.4,
        "project":     "runs/train",
        "name":        "mro_tools_v1",
        "note":        "Only 1/10 classes (wrench_open_end). Needs manual data for all other tool types.",
    },
    "heavy_materials": {
        "data":        "datasets/heavy_materials_merged/data.yaml",
        "model":       "yolo11m.pt",       # medium model for large objects
        "epochs":      100,
        "imgsz":       640,
        "batch":       4,                  # reduced from 16 — yolo11m + 4GB VRAM
        "patience":    20,
        "lr0":         0.008,
        "lrf":         0.001,
        "mosaic":      0.5,
        "mixup":       0.0,
        "degrees":     15.0,               # heavy items rarely appear upside down
        "flipud":      0.0,
        "fliplr":      0.5,
        "hsv_h":       0.015,
        "hsv_s":       0.5,
        "hsv_v":       0.5,
        "project":     "runs/train",
        "name":        "heavy_materials_v1",
        "note":        "No public data downloaded. Requires manual collection.",
    },
    "medical_pharma": {
        "data":        "datasets/medical_pharma_merged/data.yaml",
        "model":       "yolo11s.pt",
        "epochs":      120,
        "imgsz":       640,
        "batch":       8,                  # reduced from 16 to avoid OOM on 4GB VRAM
        "patience":    25,
        "lr0":         0.01,
        "lrf":         0.001,
        "mosaic":      0.0,               # medical items need precise boxes
        "mixup":       0.0,
        "degrees":     5.0,               # minimal rotation for medical items
        "flipud":      0.0,
        "fliplr":      0.5,
        "hsv_h":       0.01,
        "hsv_s":       0.3,
        "hsv_v":       0.3,
        "project":     "runs/train",
        "name":        "medical_pharma_v1",
        "note":        "No public data downloaded. Requires manual collection.",
    },
    "irregular_specialized": {
        "data":        "datasets/irregular_specialized_merged/data.yaml",
        "model":       "yolo11s.pt",
        "epochs":      120,
        "imgsz":       640,
        "batch":       16,
        "patience":    25,
        "lr0":         0.01,
        "lrf":         0.001,
        "mosaic":      1.0,
        "mixup":       0.1,
        "degrees":     90.0,
        "flipud":      0.3,
        "fliplr":      0.5,
        "hsv_h":       0.015,
        "hsv_s":       0.7,
        "hsv_v":       0.4,
        "project":     "runs/train",
        "name":        "irregular_specialized_v1",
        "note":        "6/10 classes populated. STRONGEST dataset: 1642 images.",
    },
}

# Training order — strongest datasets first for early validation
TRAIN_ORDER = ["irregular_specialized", "small_parts", "mro_tools", "heavy_materials", "medical_pharma"]


def check_prerequisites():
    """Validate ultralytics is installed and CUDA is available."""
    try:
        import torch
        from ultralytics import YOLO
        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "CPU"
        print(f"[OK] PyTorch {torch.__version__} | Device: {device}")
        if not cuda:
            print("[WARN] No GPU detected. Training will be very slow on CPU.")
            print("       Consider Google Colab (free T4 GPU) for training.")
        return True
    except ImportError as e:
        print(f"[ERR] Missing package: {e}")
        print("      Run: pip install ultralytics")
        return False


def check_dataset(model_key: str, cfg: dict) -> bool:
    """Check if the merged dataset exists and has images."""
    data_yaml = Path(cfg["data"])
    if not data_yaml.exists():
        print(f"  [SKIP] {model_key}: data.yaml not found at {data_yaml}")
        print(f"         Run merge first or collect manual data.")
        return False
    # Count images
    data_dir = data_yaml.parent
    train_imgs = list((data_dir / "train" / "images").glob("*.*"))
    print(f"  [OK] {model_key}: {len(train_imgs)} training images found")
    return len(train_imgs) > 0


def train_model(model_key: str, cfg: dict, device: str = "0") -> dict:
    """Train a single YOLO11 model with the given config."""
    from ultralytics import YOLO

    print(f"\n{'='*60}")
    print(f"  TRAINING: {model_key}")
    print(f"  Note: {cfg['note']}")
    print(f"  Data: {cfg['data']}")
    print(f"  Model: {cfg['model']} | Epochs: {cfg['epochs']} | imgsz: {cfg['imgsz']}")
    print(f"{'='*60}\n")

    start = time.time()
    try:
        model = YOLO(cfg["model"])  # loads pretrained weights
        results = model.train(
            data=cfg["data"],
            epochs=cfg["epochs"],
            imgsz=cfg["imgsz"],
            batch=cfg["batch"],
            patience=cfg["patience"],
            lr0=cfg["lr0"],
            lrf=cfg["lrf"],
            mosaic=cfg["mosaic"],
            mixup=cfg["mixup"],
            degrees=cfg["degrees"],
            flipud=cfg["flipud"],
            fliplr=cfg["fliplr"],
            hsv_h=cfg["hsv_h"],
            hsv_s=cfg["hsv_s"],
            hsv_v=cfg["hsv_v"],
            project=cfg["project"],
            name=cfg["name"],
            device=device,
            exist_ok=True,
            verbose=True,
            # Advanced settings
            cos_lr=True,               # cosine LR schedule
            label_smoothing=0.05,      # reduce overconfidence
            warmup_epochs=3,
            weight_decay=0.0005,
            workers=4,
        )
        elapsed = time.time() - start
        best_model = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
        return {
            "status": "success",
            "model_key": model_key,
            "best_weights": str(best_model),
            "elapsed_min": round(elapsed / 60, 1),
            "map50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "map50_95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
        }
    except Exception as e:
        return {"status": "error", "model_key": model_key, "error": str(e)}


def validate_model(model_key: str, cfg: dict):
    """Run validation on the best saved weights."""
    from ultralytics import YOLO
    best = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    if not best.exists():
        print(f"  [SKIP] No weights found at {best}")
        return
    print(f"\n  Validating {model_key}...")
    model = YOLO(str(best))
    metrics = model.val(data=cfg["data"], imgsz=cfg["imgsz"])
    print(f"  mAP50:    {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")


def resume_training(model_key: str, cfg: dict, device: str = "0"):
    """Resume from last checkpoint."""
    from ultralytics import YOLO
    last = Path(cfg["project"]) / cfg["name"] / "weights" / "last.pt"
    if not last.exists():
        print(f"  [ERR] No checkpoint at {last}. Starting fresh.")
        return train_model(model_key, cfg, device)
    print(f"  Resuming from {last}")
    model = YOLO(str(last))
    model.train(resume=True)


def print_summary(results: list):
    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if r["status"] == "success":
            print(f"  [OK]  {r['model_key']:<25} mAP50={r['map50']:.3f}  ({r['elapsed_min']}min)")
            print(f"        Weights: {r['best_weights']}")
        else:
            print(f"  [ERR] {r['model_key']:<25} {r.get('error','unknown error')}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="YOLO11 Ensemble Training Script")
    parser.add_argument("--mode", choices=["all", "single", "resume", "validate", "check"],
                        default="check", help="Operation to perform")
    parser.add_argument("--model", choices=list(TRAIN_CONFIGS.keys()),
                        help="Target model (required for single/resume/validate)")
    parser.add_argument("--device", default="0",
                        help="CUDA device id (0=first GPU) or 'cpu'")
    args = parser.parse_args()

    if args.mode == "check":
        print("\n=== DATASET STATUS CHECK ===\n")
        if not check_prerequisites():
            return
        print()
        for key, cfg in TRAIN_CONFIGS.items():
            check_dataset(key, cfg)
        print("\nRun with --mode all to start training.")
        return

    if not check_prerequisites():
        sys.exit(1)

    if args.mode == "all":
        results = []
        for model_key in TRAIN_ORDER:
            cfg = TRAIN_CONFIGS[model_key]
            if not check_dataset(model_key, cfg):
                results.append({"status": "skipped", "model_key": model_key})
                continue
            r = train_model(model_key, cfg, args.device)
            results.append(r)
            # Save intermediate results
            with open("training_results.json", "w") as f:
                json.dump(results, f, indent=2)
            # Graceful stop: if STOP_TRAINING file exists, exit after current model
            if Path("STOP_TRAINING").exists():
                print(f"\n[STOP] STOP_TRAINING file detected — stopping after {model_key}.")
                print(f"       Delete STOP_TRAINING and re-run to continue remaining models.")
                break
        print_summary(results)

    elif args.mode == "single":
        if not args.model:
            print("[ERR] --model required for single mode")
            sys.exit(1)
        cfg = TRAIN_CONFIGS[args.model]
        if not check_dataset(args.model, cfg):
            sys.exit(1)
        r = train_model(args.model, cfg, args.device)
        print_summary([r])

    elif args.mode == "resume":
        if not args.model:
            print("[ERR] --model required for resume mode")
            sys.exit(1)
        resume_training(args.model, TRAIN_CONFIGS[args.model], args.device)

    elif args.mode == "validate":
        if not args.model:
            print("[ERR] --model required for validate mode")
            sys.exit(1)
        validate_model(args.model, TRAIN_CONFIGS[args.model])


if __name__ == "__main__":
    main()
