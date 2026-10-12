"""
rf_async_download.py — Handles Roboflow async (HTTP 202) export jobs.

When Roboflow returns 202 on an export request, the dataset is being generated.
We poll every 10s until the link is ready, then download and unzip.

Run: python rf_async_download.py --api_key YOUR_ROBOFLOW_API_KEY

Targets only the 5 datasets that couldn't be downloaded via the SDK.
"""
import sys, io, zipfile, json, time, requests
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

def get_api_key():
    if "--api_key" in sys.argv:
        return sys.argv[sys.argv.index("--api_key") + 1]
    print("Usage: python rf_async_download.py --api_key YOUR_KEY")
    sys.exit(1)

API_KEY = get_api_key()

# Datasets that returned HTTP 202 (async export capable)
# (workspace, project, version, format, destination)
TARGETS = [
    ("medical-equipment",          "syringe-detection",          8, "yolov8",     "raw_downloads/medical_pharma/medical-equipment__syringe-detection"),
    ("electronic-waste-detection", "e-waste-dataset",            4, "yolov5pytorch","raw_downloads/irregular_specialized/electronic-waste-detection__e-waste-dataset"),
    ("tool-detection",             "workshop-tools",             1, "yolov8",     "raw_downloads/mro_tools/tool-detection__workshop-tools"),
    ("gas-cylinder",               "gas-cylinder-detection",     3, "yolov8",     "raw_downloads/heavy_materials/gas-cylinder__gas-cylinder-detection"),
    ("medical-waste",              "medical-waste-detection",    1, "yolov8",     "raw_downloads/medical_pharma/medical-waste__medical-waste-detection"),
]

def trigger_and_wait(ws, proj, ver, fmt, max_polls=20, poll_interval=12) -> tuple[bool, str, str]:
    """
    Triggers an export job and polls until the link is ready.
    Returns (success, download_link_or_error, final_format).
    """
    url = f"https://api.roboflow.com/{ws}/{proj}/{ver}/{fmt}"
    params = {"api_key": API_KEY}

    # Try triggering the export
    r = requests.get(url, params=params, timeout=20)
    if r.status_code not in (200, 202):
        return False, f"Initial request: HTTP {r.status_code}", fmt

    # If immediately ready
    if r.status_code == 200:
        try:
            data = r.json()
            link = (data.get("export") or {}).get("link") or data.get("link")
            if link:
                return True, link, fmt
        except Exception:
            pass

    # Poll for async completion
    print(f"    Export queued (202) — polling every {poll_interval}s ...", flush=True)
    for attempt in range(1, max_polls + 1):
        time.sleep(poll_interval)
        print(f"    Poll {attempt}/{max_polls} ...", end=" ", flush=True)
        try:
            r2 = requests.get(url, params=params, timeout=20)
            if r2.status_code == 200:
                data = r2.json()
                link = (data.get("export") or {}).get("link") or data.get("link")
                if link:
                    print(f"READY")
                    return True, link, fmt
                else:
                    # Check export progress
                    prog = (data.get("export") or {}).get("progress", "?")
                    print(f"progress={prog}%")
            elif r2.status_code == 202:
                print(f"still generating...")
            else:
                print(f"HTTP {r2.status_code}")
        except Exception as e:
            print(f"error: {e}")

    return False, f"Timed out after {max_polls} polls", fmt


def download_zip(link, dest) -> tuple[bool, str]:
    """Downloads a zip from a URL and extracts to dest."""
    print(f"    Downloading zip ...", end=" ", flush=True)
    try:
        r = requests.get(link, timeout=300, stream=True)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}")
            return False, f"zip HTTP {r.status_code}"
        content = r.content
        print(f"{len(content)//1024} KB", end=" ", flush=True)
        z = zipfile.ZipFile(io.BytesIO(content))
        Path(dest).mkdir(parents=True, exist_ok=True)
        z.extractall(dest)
        print(f"-> {len(z.namelist())} files extracted OK")
        return True, f"{len(z.namelist())} files"
    except zipfile.BadZipFile as e:
        print(f"bad zip: {e}")
        return False, str(e)
    except Exception as e:
        print(f"error: {e}")
        return False, str(e)


# ─── MAIN ────────────────────────────────────────────────────────────────────
print("\n=== ASYNC EXPORT DOWNLOADER ===\n")
results = []

for ws, proj, ver, fmt, dest in TARGETS:
    print(f"\n[{ws}/{proj}]  v{ver} / {fmt}")

    # Skip if already downloaded
    dest_path = Path(dest)
    if dest_path.exists() and sum(1 for _ in dest_path.rglob("*")) > 5:
        print(f"    SKIP — already downloaded ({sum(1 for _ in dest_path.rglob('*'))} files)")
        results.append({"ws": ws, "proj": proj, "status": "already_exists"})
        continue

    # Step 1: Trigger export and wait for link
    ok, result, used_fmt = trigger_and_wait(ws, proj, ver, fmt)

    if not ok:
        print(f"    FAIL — {result}")
        results.append({"ws": ws, "proj": proj, "status": "failed", "error": result})
        continue

    # Step 2: Download and extract zip
    dl_ok, dl_msg = download_zip(result, dest)
    if dl_ok:
        results.append({"ws": ws, "proj": proj, "ver": ver, "fmt": used_fmt, "dest": dest, "status": "ok", "files": dl_msg})
        print(f"    Saved to: {dest}")
    else:
        results.append({"ws": ws, "proj": proj, "status": "failed", "error": dl_msg})

# ─── SUMMARY ────────────────────────────────────────────────────────────────
print("\n" + "="*50)
ok_count = sum(1 for r in results if r.get("status") in ("ok", "already_exists"))
print(f"  RESULT: {ok_count}/{len(TARGETS)} datasets available")
for r in results:
    icon = "[OK]" if r.get("status") in ("ok","already_exists") else "[FAIL]"
    print(f"  {icon}  {r['ws']}/{r['proj']}")

out_log = Path("raw_downloads/async_download_log.json")
out_log.parent.mkdir(exist_ok=True)
with open(out_log, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Log saved: {out_log}")
print("="*50)
