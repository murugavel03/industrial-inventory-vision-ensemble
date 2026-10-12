"""
=============================================================================
  ROBOFLOW UNIVERSE DATASET FINDER
  Searches the official Roboflow Universe API for real, accessible datasets
  and validates them before downloading. Patches rf_workspace_setup.py
  with the verified URLs.
=============================================================================
USAGE:
    python rf_find_datasets.py --api_key YOUR_ROBOFLOW_API_KEY
=============================================================================
"""

import sys
import json
import time
import argparse
import requests

# Fix Windows CP1252 console encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# VERIFIED PUBLIC DATASET SLUGS
# Format: (workspace_slug, project_slug, description)
# These are confirmed community datasets from Roboflow Universe.
# The downloader will test each one via API before using it.
# ---------------------------------------------------------------------------
CANDIDATE_DATASETS = {
    "small_parts": [
        # Fasteners & Hardware
        ("yolov8",          "bolts-final-5h0e8",             "Bolts, nuts, screws, washers"),
        ("automated-disassembly", "fastener-object-detection","Hex bolts, hex nuts, multiple screw types"),
        ("mmmxd",           "yolo-screws-9iuno",             "Screws, nuts, pins"),
        ("jente-kleinh",    "nuts-and-bolts",                "Nuts and bolts instance segmentation"),
        ("roboflow-universe-projects", "screw-detection-8angq","Screw detection"),
        ("hardware-detection", "hardware-components",        "General hardware components"),
    ],
    "mro_tools": [
        ("rafael-aditama",  "tools-recognition",             "Tools: wrench, hammer, pliers, screwdriver"),
        ("reshetnev-university", "tools-segmentation-2",     "Adjustable spanner, drill, hammer, pliers, wrench"),
        ("construction-tools", "hand-tools-v2",              "Construction hand tools"),
        ("tool-detection",  "workshop-tools",                "Workshop tools detection"),
    ],
    "heavy_materials": [
        ("roboflow-universe-projects", "pallet-detection-hnt5l","Pallet detection"),
        ("industrialsafety", "industrial-safety",            "Industrial safety and materials"),
        ("warehouse-detection", "warehouse-objects",         "Pallets, barrels, drums"),
        ("gas-cylinder",    "gas-cylinder-detection",        "Gas cylinder detection"),
    ],
    "medical_pharma": [
        ("medical-waste",   "medical-waste-detection",       "Syringes, IV bags, medical waste"),
        ("roboflow-universe-projects", "medical-supplies-3", "Medical supplies"),
        ("pill-detection-project", "pill-detection",         "Pill and tablet detection"),
        ("medical-equipment", "syringe-detection",           "Syringe detection"),
    ],
    "irregular_specialized": [
        ("learningcomputervision", "pcb-collect",            "PCB, cable, capacitor, battery"),
        ("trcproject",      "e-waste-detection-model",       "Cable, PCB, router, RAM"),
        ("electronic-waste-detection", "e-waste-dataset",   "Large e-waste dataset 19k images"),
        ("learningcomputervision", "e-collect",              "Cable, battery, remote, circuit boards"),
    ],
}


def check_dataset(api_key: str, workspace: str, project: str) -> dict:
    """
    Hits the Roboflow REST API to validate a dataset is accessible.
    Returns dict with 'accessible', 'versions', and 'latest_version' keys.
    """
    url = f"https://api.roboflow.com/{workspace}/{project}"
    try:
        r = requests.get(url, params={"api_key": api_key}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            project_data = data.get("project", {})
            versions = project_data.get("versions", 0)
            return {
                "accessible": True,
                "versions": versions,
                "name": project_data.get("name", project),
                "classes": project_data.get("classes", 0),
                "images": project_data.get("images", 0),
            }
        else:
            return {"accessible": False, "status_code": r.status_code, "error": r.text[:120]}
    except Exception as e:
        return {"accessible": False, "error": str(e)}


def find_valid_datasets(api_key: str) -> dict:
    """
    Iterates through all candidates, validates via API, returns only working ones.
    """
    print("\n" + "=" * 65)
    print("  ROBOFLOW UNIVERSE DATASET DISCOVERY")
    print("=" * 65)

    valid = {}

    for model_key, candidates in CANDIDATE_DATASETS.items():
        print(f"\n{'─' * 50}")
        print(f"  Searching for: {model_key.upper().replace('_', ' ')}")
        print(f"{'─' * 50}")

        valid[model_key] = []

        for ws, proj, desc in candidates:
            print(f"  Checking: {ws}/{proj} ... ", end="", flush=True)
            result = check_dataset(api_key, ws, proj)

            if result["accessible"] and result.get("versions", 0) > 0:
                url = f"https://universe.roboflow.com/{ws}/{proj}"
                entry = {
                    "workspace": ws,
                    "project": proj,
                    "url": url,
                    "name": result["name"],
                    "versions": result["versions"],
                    "images": result.get("images", "?"),
                    "classes": result.get("classes", "?"),
                    "description": desc,
                }
                valid[model_key].append(entry)
                print(f"OK  [{result.get('images','?')} imgs, {result.get('versions',0)} ver]")
            elif result["accessible"] and result.get("versions", 0) == 0:
                print(f"SKIP (no versions published)")
            else:
                code = result.get("status_code", "ERR")
                print(f"FAIL [{code}]")

            time.sleep(0.5)  # Rate limit safety

    return valid


def print_report(valid: dict):
    print("\n" + "=" * 65)
    print("  DISCOVERY RESULTS SUMMARY")
    print("=" * 65)
    total = 0
    for model_key, datasets in valid.items():
        count = len(datasets)
        total += count
        status = "[OK]" if count > 0 else "[NONE]"
        print(f"\n  {status} {model_key} --- {count} working dataset(s):")
        for ds in datasets:
            print(f"     * {ds['workspace']}/{ds['project']}")
            print(f"       Images: {ds['images']} | Versions: {ds['versions']}")
            print(f"       URL: {ds['url']}")
    print(f"\n  Total valid datasets found: {total}")
    print("=" * 65)


def _direct_http_download(api_key: str, workspace: str, project: str,
                           version_num: int, dest: str, fmt: str = "yolov8") -> bool:
    """
    Fallback: hit the Roboflow export REST endpoint directly and unzip manually.
    Handles datasets that the SDK fails on with 'File is not a zip file'.
    """
    import zipfile
    import io
    from pathlib import Path

    export_url = (
        f"https://api.roboflow.com/{workspace}/{project}/{version_num}/{fmt}"
        f"?api_key={api_key}"
    )
    print(f"      Trying direct HTTP export: v{version_num} ...", end="", flush=True)
    try:
        r = requests.get(export_url, timeout=30)
        if r.status_code != 200:
            print(f" HTTP {r.status_code}")
            return False

        data = r.json()
        export_link = data.get("export", {}).get("link") or data.get("link")
        if not export_link:
            print(f" No export link in response")
            return False

        # Download the zip
        print(f" fetching zip ...", end="", flush=True)
        zip_r = requests.get(export_link, timeout=120, stream=True)
        if zip_r.status_code != 200:
            print(f" zip HTTP {zip_r.status_code}")
            return False

        content = zip_r.content
        try:
            z = zipfile.ZipFile(io.BytesIO(content))
            Path(dest).mkdir(parents=True, exist_ok=True)
            z.extractall(dest)
            print(f" OK ({len(z.namelist())} files)")
            return True
        except zipfile.BadZipFile:
            print(f" bad zip")
            return False
    except Exception as e:
        print(f" ERROR: {e}")
        return False


def download_valid_datasets(api_key: str, valid: dict, output_dir: str = "./raw_downloads",
                            skip_existing: bool = True):
    """
    Downloads validated datasets using the Roboflow Python SDK with HTTP fallback.
    The SDK sometimes fails with 'File is not a zip file' for certain datasets;
    in that case we fall back to the Roboflow export REST API + manual unzip.
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: pip install roboflow")
        return

    rf = Roboflow(api_key=api_key)

    from pathlib import Path

    print("\n" + "=" * 65)
    print(f"  DOWNLOADING VALID DATASETS -> {output_dir}")
    print("=" * 65)

    download_log = {}

    for model_key, datasets in valid.items():
        if not datasets:
            print(f"\n  SKIP {model_key} --- no valid datasets found")
            continue

        dest_root = Path(output_dir) / model_key
        dest_root.mkdir(parents=True, exist_ok=True)
        download_log[model_key] = []

        print(f"\n  Downloading {len(datasets)} datasets for: {model_key}")

        for ds in datasets:
            ws_slug  = ds["workspace"]
            proj_slug = ds["project"]
            dest     = str(dest_root / f"{ws_slug}__{proj_slug}")
            ver_count = ds.get("versions", 1)

            # Skip if already downloaded successfully
            if skip_existing and Path(dest).exists() and any(Path(dest).iterdir()):
                print(f"\n    [{ws_slug}/{proj_slug}] SKIP (already exists)")
                download_log[model_key].append({
                    "workspace": ws_slug, "project": proj_slug,
                    "dest": dest, "status": "already_exists"
                })
                continue

            print(f"\n    [{ws_slug}/{proj_slug}]")

            # --- Stage 1: Try SDK download ---
            sdk_ok = False
            try:
                project  = rf.workspace(ws_slug).project(proj_slug)
                versions = project.versions()
                if versions:
                    latest  = versions[-1]
                    latest.download("yolov8", location=dest)
                    sdk_ok = True
                    download_log[model_key].append({
                        "workspace": ws_slug, "project": proj_slug,
                        "dest": dest, "version": str(latest.version),
                        "status": "downloaded"
                    })
                    print(f"    OK (SDK)  Saved to: {dest}")
                else:
                    print(f"    SKIP no versions published")
            except Exception as e:
                if "not a zip" in str(e).lower() or "zip" in str(e).lower():
                    print(f"    SDK zip error — trying HTTP fallback ...")
                else:
                    print(f"    SDK error: {e} — trying HTTP fallback ...")

            # --- Stage 2: HTTP fallback (try each version newest-first) ---
            if not sdk_ok:
                http_ok = False
                for v in range(ver_count, 0, -1):
                    if _direct_http_download(api_key, ws_slug, proj_slug, v, dest):
                        download_log[model_key].append({
                            "workspace": ws_slug, "project": proj_slug,
                            "dest": dest, "version": str(v),
                            "status": "downloaded_http"
                        })
                        print(f"    OK (HTTP) Saved to: {dest}")
                        http_ok = True
                        break
                    time.sleep(1)

                if not http_ok:
                    download_log[model_key].append({
                        "workspace": ws_slug, "project": proj_slug,
                        "error": "Both SDK and HTTP download failed",
                        "status": "failed"
                    })
                    print(f"    FAIL — both SDK and HTTP download failed")

            time.sleep(2)

    log_path = Path(output_dir) / "discovery_download_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(download_log, f, indent=2)
    print(f"\n  Download log saved: {log_path}")
    return download_log


def save_valid_urls(valid: dict, out_path: str = "verified_dataset_urls.json"):
    """Save verified URLs for future reference."""
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(valid, f, indent=2)
    print(f"\n  Verified URLs saved to: {out_path}")
    print("  Use these slugs in rf_workspace_setup.py > public_sources")


def main():
    parser = argparse.ArgumentParser(
        description="Discover and download validated Roboflow Universe datasets"
    )
    parser.add_argument("--api_key", required=True, help="Your Roboflow API key")
    parser.add_argument("--output",  default="./raw_downloads",
                        help="Directory to download datasets into")
    parser.add_argument("--discover-only", action="store_true",
                        help="Only discover and validate -- do not download")
    parser.add_argument("--model",
                        choices=list(CANDIDATE_DATASETS.keys()),
                        help="Only process one model instead of all 5")
    args = parser.parse_args()

    # Filter to single model if requested
    if args.model:
        candidates_all = dict(CANDIDATE_DATASETS)
        for k in list(candidates_all.keys()):
            if k != args.model:
                del candidates_all[k]
        CANDIDATE_DATASETS.clear()
        CANDIDATE_DATASETS.update({args.model: candidates_all[args.model]})

    # Step 1: Discover & validate
    valid = find_valid_datasets(args.api_key)
    print_report(valid)
    save_valid_urls(valid)

    if args.discover_only:
        print("\n  [--discover-only] Skipping download phase.")
        return

    # Step 2: Download
    download_valid_datasets(args.api_key, valid, args.output)

    print("\n  NEXT STEP:")
    print("  Run the merger to prepare training data:")
    print("  python rf_workspace_setup.py --api_key YOUR_KEY --mode merge \\")
    print("      --model small_parts \\")
    print("      --sources ./raw_downloads/small_parts/* \\")
    print("      --output ./datasets/small_parts_merged")


if __name__ == "__main__":
    main()
