"""
heavy_materials_fetch.py
========================
Step 1: Discover available heavy_materials datasets on Roboflow Universe
Step 2: Download all valid ones into raw_downloads/heavy_materials/
Step 3: Merge all downloaded data into datasets/heavy_materials_merged/
         with a proper data.yaml ready for YOLO training.

Run:
    python heavy_materials_fetch.py --api_key YOUR_ROBOFLOW_API_KEY

Usage with flags:
    --discover-only     Only check what's available, no download
    --merge-only        Skip download, only (re)merge already-downloaded data
"""
import sys, io, os, zipfile, json, time, shutil, yaml, requests, argparse, random
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# =============================================================================
# CANDIDATES — broad net of pallets, pipes, drums, cylinders, heavy industrial
# =============================================================================
HEAVY_CANDIDATES = [
    # ── Pallets ──────────────────────────────────────────────────────────────
    ("rmv-sys-eng",            "pallet-detection",             "Pallet detection"),
    ("logistics",              "pallet-detection-v2",          "Logistics pallet v2"),
    ("object-detect",          "pallet",                       "Pallet object detect"),
    ("palletdetect",           "pallet-detection",             "Pallet detect"),
    ("warehouse-ai",           "pallet-detection",             "Warehouse AI pallets"),
    ("roboflow-universe-projects", "pallet-detection",         "Roboflow pallet"),
    ("conveyor-belt",          "pallet-tracking",              "Pallet tracking"),
    ("palletsdetect",          "wooden-pallet-detection",      "Wooden pallet"),
    ("pallet-detect",          "pallet-detection-v2",          "Pallet v2"),
    ("warehouse-vision",       "pallet-detection",             "Warehouse vision pallet"),
    ("forklift-ai",            "pallet-pocket-detection",      "Forklift pallet pocket"),
    ("material-handling",      "pallet-detection",             "Material handling"),
    ("siminnovations",         "wooden-pallet-detection",      "SIM Innovations pallet"),

    # ── Gas Cylinders / LPG ──────────────────────────────────────────────────
    ("gas-cylinder",           "gas-cylinder-detection",       "Gas cylinder detect"),
    ("gas-detect",             "gas-cylinder",                 "Gas cylinder v2"),
    ("cylinder-detect",        "lpg-cylinder-detection",       "LPG cylinder"),
    ("lpg-detection",          "lpg-cylinder",                 "LPG detect"),
    ("industrial-gas",         "gas-cylinder-detection",       "Industrial gas cylinder"),
    ("safetyscan",             "gas-cylinder-detection",       "Safety scan cylinders"),
    ("hazmat",                 "gas-cylinder-detection",       "Hazmat cylinders"),
    ("oil-gas",                "gas-cylinder-detection",       "Oil gas cylinders"),
    ("compressed-gas",         "cylinder-detection",           "Compressed gas"),

    # ── Drums / Barrels ──────────────────────────────────────────────────────
    ("barrel-detect",          "barrel-drum-detection",        "Barrel drum"),
    ("industrial-obj",         "industrial-drum",              "Industrial drum"),
    ("warehouse-safety",       "drum-detection",               "Warehouse drum"),
    ("drum-detect",            "chemical-drum-detection",      "Chemical drum"),
    ("hazardous-materials",    "drum-barrel-detection",        "Hazardous drum"),

    # ── Pipes ────────────────────────────────────────────────────────────────
    ("pipe-detect",            "pipe-detection",               "Pipe detect"),
    ("industrial",             "pipe-inspection",              "Pipe inspection"),
    ("pipeline-ai",            "pipeline-detection",           "Pipeline AI"),
    ("construction-vision",    "pipe-detection",               "Construction pipe"),
    ("plumbing-detect",        "pipe-fitting-detection",       "Plumbing pipe"),

    # ── Forklifts / Heavy Equipment ──────────────────────────────────────────
    ("roboflow-universe-projects", "forklift-detection",       "RF forklift"),
    ("forklift-assist",        "forklift-detection",           "Forklift assist"),
    ("industrial-ai",          "heavy-equipment-detection",    "Heavy equipment"),
    ("construction",           "heavy-machinery-detection",    "Heavy machinery"),
    ("site-safety",            "forklift-safety",              "Forklift safety"),

    # ── General Warehouse / Industrial Objects ────────────────────────────────
    ("warehouse-safety",       "warehouse-objects-v2",         "Warehouse objects v2"),
    ("construction",           "construction-materials",       "Construction materials"),
    ("safety-vest",            "industrial-objects-detection", "Industrial objects"),
    ("warehouse-detect",       "warehouse-item-detection",     "Warehouse items"),
    ("yolo-warehouse",         "warehouse-detection",          "YOLO warehouse"),
    ("industrial-detect",      "industrial-item-detection",    "Industrial items"),
    ("inventory-detect",       "warehouse-inventory-detection","Warehouse inventory"),
    ("factory-vision",         "factory-floor-objects",        "Factory objects"),
    ("logistics-ai",           "logistics-object-detection",   "Logistics objects"),
    ("scrap-metal",            "scrap-metal-detection",        "Scrap metal"),
    ("metal-detect",           "metal-object-detection",       "Metal objects"),
]

OUTPUT_BASE   = Path("raw_downloads/heavy_materials")
MERGED_DIR    = Path("datasets/heavy_materials_merged")
CLASS_MAPPING = {
    # Pallets → pallet
    "pallet": "pallet", "wooden pallet": "pallet", "pallet_pocket": "pallet",
    # Gas cylinders
    "gas cylinder": "gas_cylinder", "gas cylinder head": "gas_cylinder",
    "Recognising-Indian-Indian-gas": "gas_cylinder", "lpg cylinder": "gas_cylinder",
    "lpg-cylinder": "gas_cylinder", "cylinder": "gas_cylinder",
    # Drums / barrels
    "drum": "drum_barrel", "barrel": "drum_barrel", "chemical drum": "drum_barrel",
    # Pipes
    "pipe": "pipe", "pipeline": "pipe", "pipe fitting": "pipe",
    # Forklift
    "forklift": "forklift",
    # Heavy equipment
    "heavy equipment": "heavy_equipment", "machinery": "heavy_equipment",
    "excavator": "heavy_equipment", "crane": "heavy_equipment",
}
TARGET_CLASSES = sorted(set(CLASS_MAPPING.values()))


# =============================================================================
# HELPERS
# =============================================================================
def check_project(api_key, ws, proj):
    """Return project metadata dict or None."""
    try:
        r = requests.get(
            f"https://api.roboflow.com/{ws}/{proj}",
            params={"api_key": api_key}, timeout=10
        )
        if r.status_code == 200:
            d = r.json().get("project", {})
            return {
                "ok": True,
                "images": d.get("images", 0),
                "versions": d.get("versions", 0),
                "name": d.get("name", proj),
            }
        return {"ok": False, "code": r.status_code}
    except Exception as e:
        return {"ok": False, "err": str(e)[:80]}


def poll_export(api_key, ws, proj, ver, fmt, max_polls=12, interval=8):
    """Trigger export and poll for download link. Returns (link|None, status)."""
    url = f"https://api.roboflow.com/{ws}/{proj}/{ver}/{fmt}"
    try:
        r = requests.get(url, params={"api_key": api_key}, timeout=20)
    except Exception as e:
        return None, f"request error: {e}"

    if r.status_code not in (200, 202):
        return None, f"HTTP {r.status_code}"

    for attempt in range(max_polls):
        if r.status_code == 200:
            try:
                data = r.json()
                link = (data.get("export") or {}).get("link") or data.get("link")
                if link:
                    return link, "ready"
                prog = (data.get("export") or {}).get("progress", "?")
                print(f"    poll {attempt+1}/{max_polls}: {prog}%", end="\r", flush=True)
            except Exception:
                pass
        time.sleep(interval)
        try:
            r = requests.get(url, params={"api_key": api_key}, timeout=20)
        except Exception as e:
            print(f"    poll error: {e}")
    return None, "timeout"


def download_zip(link, dest: Path):
    """Download zip from link and extract. Returns (success, msg)."""
    try:
        resp = requests.get(link, timeout=300, stream=True)
        if resp.status_code != 200:
            return False, f"zip HTTP {resp.status_code}"
        content = b"".join(resp.iter_content(chunk_size=1 << 20))
        z = zipfile.ZipFile(io.BytesIO(content))
        dest.mkdir(parents=True, exist_ok=True)
        z.extractall(str(dest))
        n_img = sum(1 for f in z.namelist() if f.lower().endswith((".jpg", ".jpeg", ".png")))
        return True, f"{n_img} images / {len(z.namelist())} files"
    except zipfile.BadZipFile:
        return False, "bad zip"
    except Exception as e:
        return False, str(e)[:120]


def try_sdk_download(api_key, ws, proj, dest: Path):
    """Try Roboflow SDK download. Returns (success, msg)."""
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace(ws).project(proj)
        versions = project.versions()
        if not versions:
            return False, "no versions"
        dest.mkdir(parents=True, exist_ok=True)
        versions[-1].download("yolov8", location=str(dest))
        n_img = len(list(dest.rglob("*.jpg"))) + len(list(dest.rglob("*.png")))
        if n_img == 0:
            return False, "sdk: 0 images downloaded — falling back to HTTP"
        return True, f"sdk: {n_img} images"
    except Exception as e:
        return False, f"sdk: {str(e)[:80]}"


def download_dataset(api_key, ws, proj, n_versions, dest: Path):
    """Try SDK then HTTP export for all versions. Returns (success, msg)."""
    # Skip already downloaded
    if dest.exists():
        n = len(list(dest.rglob("*.jpg"))) + len(list(dest.rglob("*.png")))
        if n > 5:
            return True, f"already_exists ({n} images)"

    # SDK first
    ok, msg = try_sdk_download(api_key, ws, proj, dest)
    if ok:
        return True, msg

    # HTTP export — try newest 5 versions, both formats
    vers = list(range(n_versions, max(0, n_versions - 5), -1))
    for ver in vers:
        for fmt in ["yolov8", "yolov5pytorch"]:
            print(f"    v{ver}/{fmt} ...", end=" ", flush=True)
            link, status = poll_export(api_key, ws, proj, ver, fmt)
            if not link:
                print(f"no link ({status})")
                time.sleep(2)
                continue
            print("link ready! downloading...", end=" ", flush=True)
            ok, dl_msg = download_zip(link, dest)
            if ok:
                print(f"OK ({dl_msg})")
                return True, dl_msg
            print(f"FAIL ({dl_msg})")
            time.sleep(2)
        time.sleep(3)
    return False, "all methods failed"


# =============================================================================
# MERGE
# =============================================================================
def merge_heavy_materials():
    """Merge all downloaded raw data into datasets/heavy_materials_merged/."""
    print(f"\n{'='*60}")
    print("  MERGING → datasets/heavy_materials_merged/")
    print(f"{'='*60}")

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        (MERGED_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (MERGED_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    total_images = 0
    class_counter = {c: 0 for c in TARGET_CLASSES}

    src_datasets = [d for d in OUTPUT_BASE.iterdir() if d.is_dir()]
    if not src_datasets:
        print("  [ERR] No downloaded datasets found to merge!")
        return False

    for src in src_datasets:
        print(f"\n  Source: {src.name}")
        # Find images in any train/valid/test split
        for split in ("train", "valid", "test"):
            img_dir = src / split / "images"
            lbl_dir = src / split / "labels"
            if not img_dir.exists():
                # Try flat structure
                img_dir = src / "images"
                lbl_dir = src / "labels"
            if not img_dir.exists():
                continue

            images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
            if not images:
                continue

            # Assign to splits: 70% train, 20% valid, 10% test
            random.shuffle(images)
            n = len(images)
            splits = {
                "train": images[:int(n * 0.70)],
                "valid": images[int(n * 0.70):int(n * 0.90)],
                "test":  images[int(n * 0.90):],
            }

            for dest_split, imgs in splits.items():
                for img_path in imgs:
                    uid = f"{src.name}_{img_path.stem}"
                    dst_img = MERGED_DIR / dest_split / "images" / f"{uid}{img_path.suffix}"
                    shutil.copy2(img_path, dst_img)

                    # Copy matching label if it exists
                    lbl_path = lbl_dir / (img_path.stem + ".txt")
                    if lbl_path.exists():
                        shutil.copy2(lbl_path, MERGED_DIR / dest_split / "labels" / f"{uid}.txt")
                    else:
                        # Create empty label file
                        (MERGED_DIR / dest_split / "labels" / f"{uid}.txt").touch()
                    total_images += 1

        print(f"    copied to merged split")

    # Write data.yaml
    nc = len(TARGET_CLASSES)
    data_yaml = {
        "path":  str(MERGED_DIR.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "test":  "test/images",
        "nc":    nc,
        "names": TARGET_CLASSES,
    }
    with open(MERGED_DIR / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\n  [OK] Merged {total_images} images total")
    print(f"  [OK] Classes ({nc}): {TARGET_CLASSES}")
    print(f"  [OK] Saved: {MERGED_DIR / 'data.yaml'}")
    return total_images > 0


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Fetch and merge heavy_materials dataset")
    parser.add_argument("--api_key", required=True, help="Roboflow API key")
    parser.add_argument("--discover-only", action="store_true", help="Only scan, don't download")
    parser.add_argument("--merge-only",    action="store_true", help="Skip download, only merge")
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    if not args.merge_only:
        # ── DISCOVERY ────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("  HEAVY MATERIALS — DISCOVERY SCAN")
        print(f"  Checking {len(HEAVY_CANDIDATES)} candidate datasets...")
        print(f"{'='*60}\n")

        found = []
        for ws, proj, desc in HEAVY_CANDIDATES:
            print(f"  {ws}/{proj:35s} ... ", end="", flush=True)
            res = check_project(args.api_key, ws, proj)
            if res["ok"] and res["versions"] > 0 and res["images"] > 0:
                print(f"✓  [{res['images']:4d} imgs, {res['versions']} ver]  {res['name']}")
                found.append({
                    "workspace": ws, "project": proj, "desc": desc,
                    "images": res["images"], "versions": res["versions"],
                })
            elif res["ok"] and res["versions"] == 0:
                print("SKIP (0 versions)")
            elif res["ok"]:
                print(f"SKIP (0 images)")
            else:
                print(f"MISS [{res.get('code', res.get('err', '?'))}]")
            time.sleep(0.4)

        print(f"\n  Found {len(found)} downloadable datasets:")
        for d in found:
            print(f"    • {d['workspace']}/{d['project']}  ({d['images']} imgs)")

        with open("raw_downloads/heavy_materials_discovery.json", "w") as f:
            json.dump(found, f, indent=2)
        print(f"\n  Saved: raw_downloads/heavy_materials_discovery.json")

        if args.discover_only or not found:
            return

        # ── DOWNLOAD ─────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("  DOWNLOADING")
        print(f"{'='*60}")

        dl_log = []
        for d in found:
            ws, proj = d["workspace"], d["project"]
            dest = OUTPUT_BASE / f"{ws}__{proj}"
            print(f"\n  [{ws}/{proj}]  ({d['images']} imgs)")
            ok, msg = download_dataset(args.api_key, ws, proj, d["versions"], dest)
            status = "ok" if ok else "failed"
            print(f"  → {status.upper()}: {msg}")
            dl_log.append({**d, "dest": str(dest), "status": status, "msg": msg})
            time.sleep(2)

        with open("raw_downloads/heavy_materials_download_log.json", "w") as f:
            json.dump(dl_log, f, indent=2)

        ok_count = sum(1 for d in dl_log if d["status"] == "ok")
        print(f"\n  Downloaded: {ok_count}/{len(found)} datasets OK")

    # ── MERGE ─────────────────────────────────────────────────────────────────
    merge_heavy_materials()

    # ── FINAL STATUS ──────────────────────────────────────────────────────────
    train_imgs = list((MERGED_DIR / "train" / "images").glob("*.*"))
    print(f"\n{'='*60}")
    print(f"  RESULT: {len(train_imgs)} training images in heavy_materials_merged")
    if len(train_imgs) > 0:
        print(f"  READY: Run 'python train_ensemble.py --mode single --model heavy_materials'")
    else:
        print(f"  WARN: No images found — check download logs for errors")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
