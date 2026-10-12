"""
heavy_dl_http.py
================
Downloads heavy_materials datasets using PURE HTTP (no SDK).
The Roboflow SDK returns a bad-zip stub when the dataset hasn't been
pre-exported — we must POST an export trigger, then poll until the
link is ready, then download the real zip.

Run:
    python heavy_dl_http.py --api_key YOUR_ROBOFLOW_API_KEY
    python heavy_dl_http.py --api_key YOUR_ROBOFLOW_API_KEY --merge-only
"""
import sys, io, os, zipfile, json, time, shutil, random, argparse
from pathlib import Path
try:
    import requests
    import yaml
except ImportError:
    print("pip install requests pyyaml")
    sys.exit(1)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Datasets confirmed accessible (from discovery scan) ──────────────────────
DATASETS = [
    # (workspace, project, max_version, description)
    ("logistics",                      "pallet-detection-v2",   3, "Pallet Detection V2 (323 imgs)"),
    ("conveyor-belt",                  "pallet-tracking",        3, "Pallet Tracking (200 imgs)"),
    ("pallet-detect",                  "pallet-detection-v2",   3, "Pallet Detection V2 mirror (323 imgs)"),
    ("gas-cylinder",                   "gas-cylinder-detection", 3, "Gas Cylinder Detection (108 imgs)"),
    ("hazmat",                         "gas-cylinder-detection", 3, "Hazmat Gas Cylinder (108 imgs)"),
    ("warehouse-safety",               "drum-detection",         6, "Drum Detection (114 imgs)"),
    ("industrial",                     "pipe-inspection",        1, "Pipe Inspection (538 imgs)"),
    ("roboflow-universe-projects",     "forklift-detection",     1, "Forklift Detection (82 imgs)"),
]

OUTPUT_BASE = Path("raw_downloads/heavy_materials")
MERGED_DIR  = Path("datasets/heavy_materials_merged")
FORMATS     = ["yolov8", "yolov5pytorch"]

# ── Target classes for the merged dataset ────────────────────────────────────
TARGET_CLASSES = ["drum_barrel", "forklift", "gas_cylinder", "heavy_equipment", "pallet", "pipe"]


# =============================================================================
def trigger_and_poll(api_key, ws, proj, ver, fmt, max_polls=20, sleep=10):
    """
    Trigger a Roboflow export, then poll until we get a download link.
    Returns (link_url, status_str) or (None, reason).
    """
    url = f"https://api.roboflow.com/{ws}/{proj}/{ver}/{fmt}"
    params = {"api_key": api_key}
    try:
        r = requests.get(url, params=params, timeout=30)
    except Exception as e:
        return None, f"network: {e}"

    if r.status_code not in (200, 202):
        return None, f"HTTP {r.status_code}"

    for i in range(max_polls):
        try:
            data = r.json()
        except Exception:
            data = {}

        # Check for download link in various locations the API uses
        link = (
            (data.get("export") or {}).get("link")
            or data.get("link")
            or (data.get("version") or {}).get("export", {}).get("link")
        )
        if link and link.startswith("http"):
            return link, f"ready after {i} polls"

        progress = (data.get("export") or {}).get("progress", "?")
        print(f"    poll {i+1}/{max_polls}: progress={progress}%", end="\r", flush=True)
        time.sleep(sleep)
        try:
            r = requests.get(url, params=params, timeout=30)
        except Exception as e:
            print(f"\n    poll error: {e}")

    return None, "timeout"


def download_zip(link, dest: Path):
    """Download a zip from `link` and extract to `dest`. Returns (ok, msg)."""
    try:
        resp = requests.get(link, timeout=300, stream=True)
        if resp.status_code != 200:
            return False, f"HTTP {resp.status_code}"
        raw = b"".join(resp.iter_content(1 << 20))
        if len(raw) < 100:
            return False, f"tiny response ({len(raw)} bytes) — likely stub"
        z = zipfile.ZipFile(io.BytesIO(raw))
        dest.mkdir(parents=True, exist_ok=True)
        z.extractall(str(dest))
        imgs = [f for f in z.namelist() if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        return True, f"{len(imgs)} images / {len(z.namelist())} files"
    except zipfile.BadZipFile:
        return False, f"bad zip (size={len(raw) if 'raw' in dir() else '?'})"
    except Exception as e:
        return False, str(e)[:120]


def fetch_dataset(api_key, ws, proj, max_ver, dest: Path):
    """Try versions newest→oldest, all formats, until success."""
    # Already downloaded?
    if dest.exists():
        imgs = len(list(dest.rglob("*.jpg"))) + len(list(dest.rglob("*.png")))
        if imgs > 5:
            return True, f"already downloaded ({imgs} images)"

    vers = list(range(max_ver, 0, -1))
    for ver in vers:
        for fmt in FORMATS:
            print(f"    v{ver}/{fmt} polling ...", end=" ", flush=True)
            link, status = trigger_and_poll(api_key, ws, proj, ver, fmt)
            print()  # newline after poll
            if not link:
                print(f"    → no link ({status})")
                time.sleep(3)
                continue
            print(f"    → link ready! downloading ...", end=" ", flush=True)
            ok, msg = download_zip(link, dest)
            if ok:
                print(f"OK ({msg})")
                return True, msg
            print(f"FAIL ({msg})")
            time.sleep(3)
        time.sleep(5)
    return False, "all versions/formats failed"


# =============================================================================
def merge_to_dataset():
    """Merge all downloaded raw data into datasets/heavy_materials_merged/."""
    print(f"\n{'='*60}")
    print("  MERGING → datasets/heavy_materials_merged/")
    print(f"{'='*60}")

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    total = 0
    src_dirs = [d for d in OUTPUT_BASE.iterdir() if d.is_dir()]
    print(f"  Sources found: {len(src_dirs)}\n")

    for src in sorted(src_dirs):
        print(f"  [{src.name}]")
        src_imgs = []

        # Look for images in all standard locations
        for candidate in [
            src / "train" / "images",
            src / "valid"  / "images",
            src / "test"   / "images",
            src / "images",
            src,
        ]:
            if candidate.is_dir():
                found = (list(candidate.glob("*.jpg")) +
                         list(candidate.glob("*.jpeg")) +
                         list(candidate.glob("*.png")))
                for img in found:
                    # Find matching label
                    lbl = img.parent.parent / "labels" / (img.stem + ".txt")
                    if not lbl.exists():
                        lbl = img.parent / (img.stem + ".txt")
                    src_imgs.append((img, lbl if lbl.exists() else None))

        if not src_imgs:
            print(f"    ⚠ No images found — listing dir:")
            for item in src.rglob("*"):
                if item.is_file():
                    print(f"      {item.relative_to(src)}")
            continue

        print(f"    {len(src_imgs)} images found")

        # Distribute 70/20/10
        random.shuffle(src_imgs)
        n = len(src_imgs)
        splits = {
            "train": src_imgs[:int(n * 0.70)],
            "valid": src_imgs[int(n * 0.70):int(n * 0.90)],
            "test":  src_imgs[int(n * 0.90):],
        }

        for split_name, items in splits.items():
            for img_path, lbl_path in items:
                uid = f"{src.name}_{img_path.stem}"
                shutil.copy2(img_path, MERGED_DIR / split_name / "images" / f"{uid}{img_path.suffix}")
                if lbl_path and lbl_path.exists():
                    shutil.copy2(lbl_path, MERGED_DIR / split_name / "labels" / f"{uid}.txt")
                else:
                    (MERGED_DIR / split_name / "labels" / f"{uid}.txt").touch()
                total += 1

    # Write data.yaml
    nc = len(TARGET_CLASSES)
    with open(MERGED_DIR / "data.yaml", "w") as f:
        yaml.dump({
            "path":  str(MERGED_DIR.resolve()),
            "train": "train/images",
            "val":   "valid/images",
            "test":  "test/images",
            "nc":    nc,
            "names": TARGET_CLASSES,
        }, f, default_flow_style=False)

    print(f"\n  {'='*50}")
    print(f"  Total images merged: {total}")
    print(f"  Classes ({nc}):       {TARGET_CLASSES}")
    print(f"  Saved yaml:          {MERGED_DIR / 'data.yaml'}")
    print(f"  {'='*50}")
    return total


# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--merge-only", action="store_true", help="Skip download, only merge")
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if not args.merge_only:
        print(f"\n{'='*60}")
        print("  HEAVY MATERIALS — HTTP DOWNLOAD")
        print(f"  {len(DATASETS)} datasets to download")
        print(f"{'='*60}\n")

        log = []
        for ws, proj, max_ver, desc in DATASETS:
            dest = OUTPUT_BASE / f"{ws}__{proj}"
            print(f"\n[{ws}/{proj}]  {desc}")
            ok, msg = fetch_dataset(args.api_key, ws, proj, max_ver, dest)
            status = "ok" if ok else "failed"
            print(f"  → {status.upper()}: {msg}")
            log.append({"ws": ws, "proj": proj, "status": status, "msg": msg})
            time.sleep(2)

        with open("raw_downloads/heavy_http_log.json", "w") as f:
            json.dump(log, f, indent=2)

        ok_count = sum(1 for x in log if x["status"] == "ok")
        print(f"\n  Downloaded: {ok_count}/{len(DATASETS)} datasets")
        if ok_count == 0:
            print("\n  [ERR] Nothing downloaded. Check API key or network.")
            print("  Log: raw_downloads/heavy_http_log.json")
            return

    # Merge
    total = merge_to_dataset()

    # Final check
    train_imgs = list((MERGED_DIR / "train" / "images").glob("*.*"))
    print(f"\n{'='*60}")
    if train_imgs:
        print(f"  ✓ READY: {len(train_imgs)} training images")
        print(f"  Run: python train_ensemble.py --mode single --model heavy_materials")
    else:
        print(f"  ✗ WARN: 0 training images — download may have failed")
        print(f"    Check: raw_downloads/heavy_http_log.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
