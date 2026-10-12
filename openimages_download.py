"""
openimages_download.py
Downloads labeled images from Google Open Images v7 via FiftyOne.
This is FREE, unrestricted, and has exactly the classes we need.

Classes available in Open Images v7 relevant to our models:
  heavy_materials:  Barrel, Pallet, Gas cylinder, Pipe
  medical_pharma:   Syringe, Medical equipment, Glove, Face mask

Run:
    pip install fiftyone
    python openimages_download.py --model heavy_materials --max 500
    python openimages_download.py --model medical_pharma  --max 500
    python openimages_download.py --model all             --max 300

Outputs YOLO-format dataset to raw_downloads/<model>/openimages__<model>/
"""
import sys, os, json, shutil, argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ─── OPEN IMAGES CLASS MAPPING TO OUR TARGET CLASSES ─────────────────────────
# Open Images uses its own label names — map them to our YOLO class IDs
OI_CONFIG = {
    "heavy_materials": {
        "oi_classes": [
            "Barrel",          # -> drum_200L
            "Pallet",          # -> pallet_standard
            "Gas stove",       # -> gas_cylinder_industrial (closest available)
            "Pressure cooker", # -> gas_cylinder_industrial
            "Pipe",            # -> pipe_pvc_50mm
            "Wheel",           # -> cable_reel_small (round spool-like)
            "Ladder",          # -> angle_iron (structural metal)
        ],
        "oi_to_target": {
            "Barrel":          "drum_200L",
            "Pallet":          "pallet_standard",
            "Gas stove":       "gas_cylinder_industrial",
            "Pressure cooker": "gas_cylinder_industrial",
            "Pipe":            "pipe_pvc_50mm",
            "Wheel":           "cable_reel_small",
            "Ladder":          "angle_iron",
        },
        "target_classes": [
            "pipe_pvc_50mm", "pipe_steel_100mm", "cable_reel_small",
            "cable_reel_large", "sheet_metal_roll", "drum_200L",
            "pallet_standard", "gas_cylinder_industrial", "angle_iron", "i_beam_segment"
        ],
        "dest": "raw_downloads/heavy_materials/openimages__heavy_materials",
    },
    "medical_pharma": {
        "oi_classes": [
            "Syringe",
            "Medical equipment",
            "Glove",
            "Face mask",
            "Bottle",
            "Box",
        ],
        "oi_to_target": {
            "Syringe":           "syringe_5ml",
            "Medical equipment": "syringe_5ml",
            "Glove":             "surgical_glove_box",
            "Face mask":         "mask_n95",
            "Bottle":            "medicine_bottle_amber",
            "Box":               "pill_blister_pack",
        },
        "target_classes": [
            "syringe_5ml", "syringe_20ml", "iv_bag_500ml", "iv_bag_1000ml",
            "pill_blister_pack", "medicine_bottle_amber", "surgical_glove_box",
            "mask_n95", "bandage_roll", "specimen_container"
        ],
        "dest": "raw_downloads/medical_pharma/openimages__medical_pharma",
    },
}


def check_fiftyone():
    try:
        import fiftyone
        import fiftyone.zoo as foz
        return True
    except ImportError:
        print("\n[ERR] fiftyone not installed.")
        print("      Run: pip install fiftyone")
        print("      Then re-run this script.")
        return False


def download_open_images(model_key: str, max_samples: int, splits=("train", "validation")):
    import fiftyone.zoo as foz
    import fiftyone as fo

    cfg = OI_CONFIG[model_key]
    dest = Path(cfg["dest"])
    dest.mkdir(parents=True, exist_ok=True)

    target_classes = cfg["target_classes"]
    class_to_id = {c: i for i, c in enumerate(target_classes)}
    oi_to_target = cfg["oi_to_target"]
    oi_classes   = cfg["oi_classes"]

    print(f"\n{'='*60}")
    print(f"  Open Images v7 Download: {model_key}")
    print(f"  Searching for: {oi_classes}")
    print(f"  Max samples: {max_samples} per split")
    print(f"  Output: {dest}")
    print(f"{'='*60}")

    total_exported = 0

    for split in splits:
        oi_split = "validation" if split == "valid" else split
        print(f"\n  Downloading split: {oi_split} ...")

        try:
            dataset = foz.load_zoo_dataset(
                "open-images-v7",
                split=oi_split,
                label_types=["detections"],
                classes=oi_classes,
                max_samples=max_samples,
                only_matching=True,   # only images that have our classes
            )
        except Exception as e:
            print(f"  [WARN] Could not load {oi_split}: {e}")
            continue

        print(f"  Downloaded {len(dataset)} images for split={oi_split}")

        out_split = "valid" if oi_split == "validation" else oi_split
        img_dir = dest / out_split / "images"
        lbl_dir = dest / out_split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for sample in dataset:
            img_path = Path(sample.filepath)
            if not img_path.exists():
                continue

            # Build YOLO label lines
            yolo_lines = []
            detections = sample.get_field("detections")
            if detections is None:
                continue

            for det in detections.detections:
                label = det.label
                target = oi_to_target.get(label)
                if target is None or target not in class_to_id:
                    continue
                cls_id = class_to_id[target]
                # Open Images bbox: [x1, y1, w, h] normalized
                x, y, w, h = det.bounding_box
                cx = x + w / 2
                cy = y + h / 2
                yolo_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if not yolo_lines:
                continue

            # Copy image + write label
            dest_img = img_dir / img_path.name
            shutil.copy2(img_path, dest_img)
            lbl_file = lbl_dir / (img_path.stem + ".txt")
            with open(lbl_file, "w") as f:
                f.write("\n".join(yolo_lines))
            total_exported += 1

        # Clean up fiftyone dataset from memory
        dataset.delete()

    # Write data.yaml
    data_yaml = dest / "data.yaml"
    import yaml
    with open(data_yaml, "w") as f:
        yaml.dump({
            "path":  str(dest.absolute()),
            "train": "train/images",
            "val":   "valid/images",
            "test":  "test/images",
            "nc":    len(target_classes),
            "names": target_classes,
        }, f, default_flow_style=False)

    print(f"\n  [OK] {total_exported} images exported to YOLO format")
    print(f"  data.yaml: {data_yaml}")

    # Class distribution
    all_labels = list((dest / "train" / "labels").glob("*.txt")) + \
                 list((dest / "valid" / "labels").glob("*.txt"))
    class_counts = {c: 0 for c in target_classes}
    for lf in all_labels:
        for line in open(lf):
            parts = line.strip().split()
            if parts:
                cls_id = int(parts[0])
                if cls_id < len(target_classes):
                    class_counts[target_classes[cls_id]] += 1

    print(f"\n  Class distribution:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        if cnt > 0:
            bar = "#" * min(40, cnt // 2)
            print(f"    {cls:<35} {cnt:>5}  {bar}")

    return total_exported


def main():
    parser = argparse.ArgumentParser(description="Download Open Images v7 for missing models")
    parser.add_argument("--model", choices=["heavy_materials", "medical_pharma", "all"],
                        default="all", help="Which model to download for")
    parser.add_argument("--max", type=int, default=400,
                        help="Max images per split to download (default: 400)")
    args = parser.parse_args()

    if not check_fiftyone():
        sys.exit(1)

    models = list(OI_CONFIG.keys()) if args.model == "all" else [args.model]

    print(f"\n=== OPEN IMAGES v7 DOWNLOADER ===")
    print(f"Target models: {models}")
    print(f"Max per split: {args.max}")

    results = {}
    for model_key in models:
        n = download_open_images(model_key, args.max)
        results[model_key] = n

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for mk, n in results.items():
        print(f"  {mk}: {n} images downloaded")

    print(f"\nNEXT STEPS:")
    print(f"  python rf_workspace_setup.py --api_key KEY --mode merge \\")
    print(f"      --model heavy_materials \\")
    print(f"      --sources raw_downloads/heavy_materials/openimages__heavy_materials \\")
    print(f"      --output datasets/heavy_materials_merged")
    print(f"\n  python train_ensemble.py --mode all")


if __name__ == "__main__":
    main()
