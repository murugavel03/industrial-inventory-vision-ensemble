"""
rf_final_download.py
Downloads the remaining 5 blocked datasets using the Roboflow export API with proper polling.
Uses version-by-version retry with yolov8/yolov5pytorch formats.

Run: python rf_final_download.py --api_key YOUR_ROBOFLOW_API_KEY
"""
import sys, io, zipfile, json, time, requests, argparse
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# These all returned HTTP 202 (async export) or 1-file zip stub from SDK
# Format: (workspace, project, max_version, dest)
TARGETS = [
    # Heavy materials — all returned 1-file zip
    ("logistics",                      "pallet-detection-v2",          3, "raw_downloads/heavy_materials/logistics__pallet-detection-v2"),
    ("industrial",                     "pipe-inspection",               1, "raw_downloads/heavy_materials/industrial__pipe-inspection"),
    ("roboflow-universe-projects",     "forklift-detection",            1, "raw_downloads/heavy_materials/roboflow-universe-projects__forklift-detection"),
    # Medical — SDK gave zip error
    ("biomedical",                     "biomedical-waste-detection",    9, "raw_downloads/medical_pharma/biomedical__biomedical-waste-detection"),
    ("ppe-detection",                  "medical-ppe-detection",        969, "raw_downloads/medical_pharma/ppe-detection__medical-ppe-detection"),
]

FORMATS = ["yolov8", "yolov5pytorch"]


def poll_export(api_key, ws, proj, ver, fmt, max_polls=15, interval=10):
    """Trigger export and poll until link is ready. Returns (link_url or None, status)."""
    url = f"https://api.roboflow.com/{ws}/{proj}/{ver}/{fmt}"
    params = {"api_key": api_key}

    r = requests.get(url, params=params, timeout=20)
    if r.status_code not in (200, 202):
        return None, f"HTTP {r.status_code}"

    for attempt in range(max_polls):
        if r.status_code == 200:
            try:
                data = r.json()
                link = (data.get("export") or {}).get("link") or data.get("link")
                if link:
                    return link, "ready"
                # Export still processing
                prog = (data.get("export") or {}).get("progress", "?")
                print(f"    poll {attempt+1}: progress={prog}%", end="\r", flush=True)
            except Exception:
                pass
        time.sleep(interval)
        try:
            r = requests.get(url, params=params, timeout=20)
        except Exception as e:
            print(f"    poll error: {e}")

    return None, "timeout"


def download_zip(link, dest):
    """Download zip from link and extract to dest. Returns (success, msg)."""
    try:
        resp = requests.get(link, timeout=300)
        if resp.status_code != 200:
            return False, f"zip HTTP {resp.status_code}"
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        Path(dest).mkdir(parents=True, exist_ok=True)
        z.extractall(dest)
        return True, f"{len(z.namelist())} files"
    except zipfile.BadZipFile:
        return False, "bad zip"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", required=True)
    args = parser.parse_args()

    print("\n=== FINAL DOWNLOAD — HTTP EXPORT POLLING ===\n")
    results = []

    for ws, proj, max_ver, dest in TARGETS:
        dest_path = Path(dest)

        # Skip already downloaded
        if dest_path.exists() and sum(1 for _ in dest_path.rglob("*.jpg")) + sum(1 for _ in dest_path.rglob("*.png")) > 5:
            n = sum(1 for _ in dest_path.rglob("*.*"))
            print(f"[SKIP] {ws}/{proj} — already has {n} files")
            results.append({"ws": ws, "proj": proj, "status": "exists"})
            continue

        print(f"\n[{ws}/{proj}] max_ver={max_ver}")
        success = False

        # Try versions newest-first, both formats
        versions_to_try = list(range(max_ver, max(0, max_ver - 5), -1))  # try last 5 versions
        if max_ver > 5:
            versions_to_try = [max_ver, max_ver-1, max_ver//2, 1]  # smart sampling

        for ver in versions_to_try:
            for fmt in FORMATS:
                print(f"  v{ver}/{fmt} polling...", end=" ", flush=True)
                link, status = poll_export(args.api_key, ws, proj, ver, fmt)
                if not link:
                    print(f"no link ({status})")
                    time.sleep(2)
                    continue

                print(f"link ready! downloading...", end=" ", flush=True)
                ok, msg = download_zip(link, dest)
                if ok:
                    print(f"OK ({msg})")
                    results.append({"ws": ws, "proj": proj, "ver": ver, "fmt": fmt, "dest": dest, "status": "ok"})
                    success = True
                    break
                else:
                    print(f"FAIL ({msg})")
                time.sleep(2)

            if success:
                break
            time.sleep(3)

        if not success:
            print(f"  !! FAILED — {ws}/{proj}")
            results.append({"ws": ws, "proj": proj, "status": "failed"})

    # Summary
    print(f"\n{'='*50}")
    ok = sum(1 for r in results if r["status"] in ("ok", "exists"))
    print(f"  DONE: {ok}/{len(TARGETS)} datasets ready")
    for r in results:
        icon = "OK  " if r["status"] in ("ok","exists") else "FAIL"
        print(f"  [{icon}] {r.get('ws','')}/{r.get('proj','')}")

    with open("raw_downloads/final_download_log.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Log: raw_downloads/final_download_log.json")
    print("="*50)


if __name__ == "__main__":
    main()
