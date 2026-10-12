"""
Targeted retry for datasets that failed with 'File is not a zip file'.
Uses direct Roboflow export REST API + manual HTTP download instead of SDK.
Run: python rf_retry_failed.py --api_key YOUR_ROBOFLOW_API_KEY
"""
import sys, io, zipfile, json, time, requests
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

API_KEY = sys.argv[sys.argv.index("--api_key") + 1] if "--api_key" in sys.argv else None
if not API_KEY:
    print("Usage: python rf_retry_failed.py --api_key YOUR_KEY")
    sys.exit(1)

# Failed datasets to retry: (workspace, project, max_versions, output_path)
FAILED = [
    ("tool-detection",               "workshop-tools",            1, "raw_downloads/mro_tools/tool-detection__workshop-tools"),
    ("gas-cylinder",                 "gas-cylinder-detection",    3, "raw_downloads/heavy_materials/gas-cylinder__gas-cylinder-detection"),
    ("medical-waste",                "medical-waste-detection",   1, "raw_downloads/medical_pharma/medical-waste__medical-waste-detection"),
    ("medical-equipment",            "syringe-detection",         9, "raw_downloads/medical_pharma/medical-equipment__syringe-detection"),
    ("electronic-waste-detection",   "e-waste-dataset",           5, "raw_downloads/irregular_specialized/electronic-waste-detection__e-waste-dataset"),
]

FORMATS = ["yolov8", "yolov5pytorch", "yolov5"]  # try multiple formats

def try_download(ws, proj, ver, fmt, dest):
    url = f"https://api.roboflow.com/{ws}/{proj}/{ver}/{fmt}?api_key={API_KEY}"
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    try:
        data = r.json()
    except Exception:
        return False, "non-JSON response"

    link = (data.get("export") or {}).get("link") or data.get("link")
    if not link:
        # Sometimes the export isn't ready — check for async export
        export_info = data.get("export", {})
        if export_info.get("progress", 100) < 100:
            return False, "export not ready (async)"
        return False, f"no link in: {list(data.keys())}"

    # Download the zip
    zr = requests.get(link, timeout=180)
    if zr.status_code != 200:
        return False, f"zip fetch HTTP {zr.status_code}"
    try:
        z = zipfile.ZipFile(io.BytesIO(zr.content))
        Path(dest).mkdir(parents=True, exist_ok=True)
        z.extractall(dest)
        return True, f"{len(z.namelist())} files"
    except zipfile.BadZipFile as e:
        return False, f"bad zip: {e}"

print("\n=== RETRY FAILED DOWNLOADS (HTTP export method) ===\n")
results = []

for ws, proj, max_ver, dest in FAILED:
    print(f"[{ws}/{proj}]")
    success = False
    for ver in range(max_ver, 0, -1):
        for fmt in FORMATS:
            print(f"  v{ver} / {fmt} ...", end=" ", flush=True)
            ok, msg = try_download(ws, proj, ver, fmt, dest)
            print(msg)
            if ok:
                success = True
                results.append({"ws": ws, "proj": proj, "ver": ver, "fmt": fmt, "status": "ok", "dest": dest})
                break
            time.sleep(1)
        if success:
            break
        time.sleep(1)
    if not success:
        print(f"  !! ALL versions/formats failed for {ws}/{proj}")
        results.append({"ws": ws, "proj": proj, "status": "failed"})
    print()

with open("raw_downloads/retry_log.json", "w") as f:
    json.dump(results, f, indent=2)

ok_count = sum(1 for r in results if r["status"] == "ok")
print(f"=== DONE: {ok_count}/{len(FAILED)} recovered ===")
print(f"Log saved: raw_downloads/retry_log.json")
