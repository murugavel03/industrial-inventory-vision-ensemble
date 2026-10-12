"""
heavy_materials_openimages.py
==============================
Downloads heavy_materials images from Google Open Images v7 (FREE, no API key needed)
and saves them in YOLO format directly into datasets/heavy_materials_merged/

Classes targeted (matches data.yaml exactly):
  0  drum_barrel
  1  forklift
  2  gas_cylinder
  3  heavy_equipment
  4  pallet
  5  pipe

Run:
    pip install fiftyone
    python heavy_materials_openimages.py --max 300
"""
import sys, shutil, argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Open Images label → our YOLO class ──────────────────────────────────────
# Only use classes confirmed valid via fiftyone.utils.openimages.get_classes()
OI_TO_CLASS = {
    "Barrel":           "drum_barrel",
    "Drum":             "drum_barrel",
    "Gas stove":        "gas_cylinder",
    "Pressure cooker":  "gas_cylinder",
    "Truck":            "forklift",        # closest available to forklift/heavy vehicle
    "Land vehicle":     "heavy_equipment", # covers excavators/bulldozers broadly
}
OI_CLASSES  = list(OI_TO_CLASS.keys())
TARGET_NAMES = ["drum_barrel", "forklift", "gas_cylinder", "heavy_equipment", "pallet", "pipe"]
CLASS_TO_ID  = {c: i for i, c in enumerate(TARGET_NAMES)}

MERGED_DIR = Path("datasets/heavy_materials_merged")


def download_and_export(max_samples: int):
    try:
        import fiftyone.zoo as foz
        import fiftyone as fo
        import yaml
    except ImportError:
        print("[ERR] fiftyone not installed. Run: pip install fiftyone")
        sys.exit(1)

    # Prepare output dirs
    for split in ("train", "valid", "test"):
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    total = 0

    for oi_split, out_split in [("train", "train"), ("validation", "valid")]:
        print(f"\n{'='*55}")
        print(f"  Downloading Open Images [{oi_split}] split...")
        print(f"  Classes: {OI_CLASSES}")
        print(f"  Max samples: {max_samples}")
        print(f"{'='*55}")

        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=oi_split,
                label_types=["detections"],
                classes=OI_CLASSES,
                max_samples=max_samples,
                only_matching=True,
            )
        except Exception as e:
            print(f"  [WARN] Could not load {oi_split}: {e}")
            continue

        print(f"  Got {len(dataset)} images")

        for sample in dataset:
            img_path = Path(sample.filepath)
            if not img_path.exists():
                continue

            detections = sample.get_field("ground_truth")
            if detections is None:
                continue

            yolo_lines = []
            for det in detections.detections:
                target = OI_TO_CLASS.get(det.label)
                if target is None:
                    continue
                cls_id = CLASS_TO_ID[target]
                x, y, w, h = det.bounding_box
                cx = x + w / 2
                cy = y + h / 2
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if not yolo_lines:
                continue

            uid = f"oi_{img_path.stem}"
            dst_img = MERGED_DIR / out_split / "images" / f"{uid}{img_path.suffix}"
            shutil.copy2(img_path, dst_img)
            lbl_file = MERGED_DIR / out_split / "labels" / f"{uid}.txt"
            lbl_file.write_text("\n".join(yolo_lines))
            total += 1

        dataset.delete()
        print(f"  Exported {total} images so far...")

    # Write / update data.yaml
    data_yaml = MERGED_DIR / "data.yaml"
    with open(data_yaml, "w") as f:
        import yaml
        yaml.dump({
            "path":  str(MERGED_DIR.resolve()),
            "train": "train/images",
            "val":   "valid/images",
            "test":  "test/images",
            "nc":    len(TARGET_NAMES),
            "names": TARGET_NAMES,
        }, f, default_flow_style=False)

    # Summary
    print(f"\n{'='*55}")
    print(f"  DONE — {total} images saved to {MERGED_DIR}")
    print(f"  data.yaml updated with {len(TARGET_NAMES)} classes: {TARGET_NAMES}")

    # Class distribution
    class_counts = {c: 0 for c in TARGET_NAMES}
    for split in ("train", "valid"):
        for lf in (MERGED_DIR / split / "labels").glob("*.txt"):
            for line in lf.read_text().splitlines():
                parts = line.strip().split()
                if parts:
                    cls_id = int(parts[0])
                    if cls_id < len(TARGET_NAMES):
                        class_counts[TARGET_NAMES[cls_id]] += 1

    print(f"\n  Class distribution:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(40, cnt // 2)
        print(f"    {cls:<25} {cnt:>5}  {bar}")
    print(f"{'='*55}")

    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=300,
                        help="Max images per split (default: 300)")
    args = parser.parse_args()

    n = download_and_export(args.max)

    if n > 0:
        print(f"\n[NEXT] Starting Model 4 training...")
        import subprocess, sys
        subprocess.run([sys.executable, "train_ensemble.py",
                        "--mode", "single", "--model", "heavy_materials", "--device", "0"])
    else:
        print("\n[ERR] No images downloaded. Check fiftyone install and internet connection.")


if __name__ == "__main__":
    main()
