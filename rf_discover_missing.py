"""
rf_discover_missing.py
Expanded discovery scan specifically for the two blocked models:
  - heavy_materials  (pallets, pipes, drums, gas cylinders)
  - medical_pharma   (syringes, pills, IV bags, medical supplies)

Tests ~20 new candidates per model via the Roboflow REST API.
Run: python rf_discover_missing.py --api_key YOUR_ROBOFLOW_API_KEY
"""
import sys, io, zipfile, json, time, requests, argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANDIDATES = {
    "heavy_materials": [
        # Pallet datasets
        ("rmv-sys-eng",      "pallet-detection",            "Pallet detection"),
        ("object-detect",    "pallet",                      "Pallet object detection"),
        ("yolo-warehouse",   "pallet-v2",                   "Warehouse pallet v2"),
        ("objectdetection",  "pallets",                     "Pallets dataset"),
        ("forklift-assist",  "pallet-pockets",              "Pallet pockets for forklifts"),
        ("warehouse",        "pallet-detection",            "Warehouse pallets"),
        ("logistics",        "pallet-detection-v2",         "Logistics pallet"),
        # Gas cylinder / drum
        ("gas-detect",       "gas-cylinder",                "Gas cylinder detect"),
        ("cylinder-detect",  "lpg-cylinder-detection",      "LPG cylinder detection"),
        ("barrel-detect",    "barrel-drum-detection",       "Barrel drum detection"),
        ("industrial-obj",   "industrial-drum",             "Industrial drum objects"),
        # Pipe
        ("pipe-detect",      "pipe-detection",              "Pipe detection"),
        ("industrial",       "pipe-inspection",             "Industrial pipe inspection"),
        # General industrial
        ("roboflow-universe-projects", "forklift-detection","Forklift & pallet"),
        ("safety-vest",      "industrial-objects-detection","Industrial objects"),
        ("construction",     "construction-materials",      "Construction materials"),
        ("warehouse-safety", "warehouse-objects-v2",        "Warehouse objects v2"),
        ("industrial-ai",    "heavy-equipment-detection",   "Heavy equipment"),
    ],
    "medical_pharma": [
        # Syringe
        ("shirley-chia",     "syringe-dataset",             "Syringe public domain"),
        ("dataset-t7hz7",    "syringes",                    "Syringes CC-BY-4.0"),
        ("syringe-16nka",    "syringe-vyhfk",               "Syringe 2025"),
        ("medical-detection","syringe-detection-v2",        "Syringe detection v2"),
        ("health-ai",        "medical-syringe",             "Medical syringe"),
        # Medical waste (multi-class)
        ("med-waste-proj",   "medical-waste-v2",            "Medical waste v2"),
        ("biomedical",       "biomedical-waste-detection",  "Biomedical waste"),
        ("hospital-waste",   "hospital-waste-detection",    "Hospital waste detection"),
        ("medical-ai",       "medical-supplies-detection",  "Medical supplies"),
        # Pill / blister pack
        ("pill-detect",      "pill-detection-v2",           "Pill detection v2"),
        ("pharma-detect",    "tablet-detection",            "Tablet/pill detection"),
        ("medicine-detect",  "medicine-detection",          "Medicine detection"),
        ("drug-detection",   "drug-detection-dataset",      "Drug/pill detection"),
        # IV bags / hospital
        ("iv-bag-detect",    "iv-bag-detection",            "IV bag detection"),
        ("hospital-equip",   "hospital-equipment",          "Hospital equipment"),
        ("medical-equip",    "medical-equipment-v2",        "Medical equipment v2"),
        # Masks / PPE
        ("ppe-detection",    "medical-ppe-detection",       "Medical PPE detection"),
        ("mask-detection",   "n95-mask-detection",          "N95 mask detection"),
    ],
}


def check(api_key, ws, proj):
    try:
        r = requests.get(f"https://api.roboflow.com/{ws}/{proj}",
                         params={"api_key": api_key}, timeout=10)
        if r.status_code == 200:
            d = r.json().get("project", {})
            return {"ok": True, "imgs": d.get("images", 0), "vers": d.get("versions", 0), "name": d.get("name", proj)}
        return {"ok": False, "code": r.status_code}
    except Exception as e:
        return {"ok": False, "err": str(e)[:60]}


def download_dataset(api_key, ws, proj, vers, dest):
    """Try SDK first, then direct HTTP export with polling."""
    from roboflow import Roboflow
    rf = Roboflow(api_key=api_key)
    dest_path = Path(dest)

    # Stage 1: SDK
    try:
        project = rf.workspace(ws).project(proj)
        versions = project.versions()
        if versions:
            versions[-1].download("yolov8", location=str(dest_path))
            return True, "sdk"
    except Exception as e:
        if "zip" not in str(e).lower():
            pass  # non-zip error, still try HTTP

    # Stage 2: HTTP export with polling
    for v in range(vers, 0, -1):
        url = f"https://api.roboflow.com/{ws}/{proj}/{v}/yolov8?api_key={api_key}"
        r = requests.get(url, timeout=20)
        if r.status_code not in (200, 202):
            continue
        # Poll up to 10x if 202
        for _ in range(10):
            if r.status_code == 200:
                link = (r.json().get("export") or {}).get("link") or r.json().get("link")
                if link:
                    zr = requests.get(link, timeout=180)
                    if zr.status_code == 200:
                        try:
                            z = zipfile.ZipFile(io.BytesIO(zr.content))
                            dest_path.mkdir(parents=True, exist_ok=True)
                            z.extractall(str(dest_path))
                            return True, f"http-v{v} ({len(z.namelist())} files)"
                        except zipfile.BadZipFile:
                            break
            time.sleep(8)
            r = requests.get(url, timeout=20)
        time.sleep(1)

    return False, "all methods failed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--output", default="./raw_downloads")
    args = parser.parse_args()

    found = {}
    print("\n" + "="*60)
    print("  EXPANDED DISCOVERY — heavy_materials + medical_pharma")
    print("="*60)

    for model_key, candidates in CANDIDATES.items():
        found[model_key] = []
        print(f"\n--- {model_key.upper()} ---")
        for ws, proj, desc in candidates:
            print(f"  {ws}/{proj} ... ", end="", flush=True)
            res = check(args.api_key, ws, proj)
            if res["ok"] and res["vers"] > 0:
                print(f"OK  [{res['imgs']} imgs, {res['vers']} ver] — {res['name']}")
                found[model_key].append({
                    "workspace": ws, "project": proj,
                    "images": res["imgs"], "versions": res["vers"],
                    "url": f"https://universe.roboflow.com/{ws}/{proj}"
                })
            elif res["ok"] and res["vers"] == 0:
                print(f"SKIP (0 versions)")
            else:
                print(f"FAIL [{res.get('code', res.get('err', '?'))}]")
            time.sleep(0.4)

    # Summary
    print(f"\n{'='*60}")
    print("  RESULTS")
    print(f"{'='*60}")
    total = 0
    for mk, ds in found.items():
        print(f"\n  {mk}: {len(ds)} usable datasets")
        for d in ds:
            print(f"    * {d['workspace']}/{d['project']}  ({d['images']} imgs)")
            total += 1
    print(f"\n  Total: {total} new datasets found")

    with open("raw_downloads/expanded_discovery.json", "w") as f:
        json.dump(found, f, indent=2)
    print("  Saved: raw_downloads/expanded_discovery.json")

    if args.discover_only or total == 0:
        return

    # Download valid ones
    print(f"\n{'='*60}")
    print("  DOWNLOADING")
    print(f"{'='*60}")
    dl_log = {}
    for mk, datasets in found.items():
        dl_log[mk] = []
        for d in datasets:
            dest = f"{args.output}/{mk}/{d['workspace']}__{d['project']}"
            print(f"\n  [{d['workspace']}/{d['project']}]")
            ok, msg = download_dataset(args.api_key, d["workspace"], d["project"], d["versions"], dest)
            status = "ok" if ok else "failed"
            print(f"    {status.upper()}: {msg}")
            dl_log[mk].append({**d, "dest": dest, "status": status, "msg": msg})
            time.sleep(2)

    with open("raw_downloads/expanded_download_log.json", "w") as f:
        json.dump(dl_log, f, indent=2)
    print(f"\n  Download log: raw_downloads/expanded_download_log.json")
    print("\n  NEXT: Run merge for each model that has data, then train_ensemble.py")


if __name__ == "__main__":
    main()
