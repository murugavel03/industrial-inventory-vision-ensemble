"""
=============================================================================
  ROBOFLOW WORKSPACE AUTO-SETUP — Ensemble Object Detection Pipeline
  Author  : AI/ML Engineering Consultant
  Project : Industrial + Medical + Specialized Inventory Detection
  Models  : 5 Specialized YOLO11 Ensemble Models
=============================================================================

USAGE:
    python rf_workspace_setup.py --api_key YOUR_API_KEY --mode setup
    python rf_workspace_setup.py --api_key YOUR_KEY --mode upload --model small_parts --dataset ./datasets/small_parts_final
    python rf_workspace_setup.py --api_key YOUR_KEY --mode status
    python rf_workspace_setup.py --api_key YOUR_KEY --mode export_all

REQUIREMENTS:
    pip install roboflow requests tqdm pyyaml
=============================================================================
"""
import os
import sys
import json
import yaml
import shutil
import argparse
import time
import hashlib
from pathlib import Path
from typing import Optional
from tqdm import tqdm

try:
    from roboflow import Roboflow
    import requests
except ImportError:
    print("[ERR] Missing dependencies. Run: pip install roboflow requests tqdm pyyaml")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Fix Windows CP1252 console — reconfigure stdout/stderr to UTF-8 so that
# emoji / Unicode characters in print() do not raise UnicodeEncodeError.
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# =============================================================================
# MASTER CONFIGURATION — All 5 Models
# =============================================================================

ENSEMBLE_CONFIG = {
    "small_parts": {
        "display_name": "Model 1 - Small Parts and Fasteners",
        "project_id":   "ensemble-small-parts",
        "description":  "Detection of bolts, nuts, screws, washers and small hardware components",
        "annotation_type": "object-detection",
        "license":      "MIT",
        "classes": [
            "bolt_hex_m6",
            "bolt_hex_m10",
            "bolt_hex_m16",
            "nut_hex_standard",
            "nut_lock",
            "washer_flat",
            "washer_spring",
            "screw_phillips_m4",
            "screw_allen_m5",
            "rivet_pop"
        ],
        "target_images": 3000,
        "imgsz": 1280,
        "use_sahi": True,
        "augmentations": {
            "flip_horizontal": True,
            "flip_vertical": False,
            "rotation": [-45, 45],
            "brightness": [-25, 25],
            "blur": [0, 1.5],
            "noise": [0, 1.5],
            "mosaic": True
        },
        "public_sources": [
            "https://universe.roboflow.com/automated-disassembly/fastener-object-detection",
            "https://universe.roboflow.com/mmmxd/yolo-screws-9iuno",
            "https://universe.roboflow.com/jente-kleinh/nuts-and-bolts"
        ]
    },

    "mro_tools": {
        "display_name": "Model 2 - MRO Hand Tools",
        "project_id":   "ensemble-mro-tools",
        "description":  "Detection of wrenches, screwdrivers, pliers, hammers and workshop tools",
        "annotation_type": "object-detection",
        "license":      "MIT",
        "classes": [
            "wrench_open_end",
            "wrench_ring",
            "screwdriver_flat",
            "screwdriver_phillips",
            "pliers_standard",
            "pliers_needle_nose",
            "hammer_claw",
            "tape_measure",
            "caliper_vernier",
            "drill_bit_set"
        ],
        "target_images": 2500,
        "imgsz": 640,
        "use_sahi": False,
        "augmentations": {
            "flip_horizontal": True,
            "flip_vertical": True,
            "rotation": [-180, 180],
            "brightness": [-30, 30],
            "blur": [0, 1.0],
            "noise": [0, 2.0],
            "mosaic": True
        },
        "public_sources": [
            "https://universe.roboflow.com/rafael-aditama/tools-recognition",
            "https://universe.roboflow.com/tool-detection/workshop-tools"
        ]
    },

    "heavy_materials": {
        "display_name": "Model 3 - Heavy Industrial Materials",
        "project_id":   "ensemble-heavy-materials",
        "description":  "Detection of pipes, drums, pallets, cable reels and bulk industrial materials",
        "annotation_type": "object-detection",
        "license":      "MIT",
        "classes": [
            "pipe_pvc_50mm",
            "pipe_steel_100mm",
            "cable_reel_small",
            "cable_reel_large",
            "sheet_metal_roll",
            "drum_200L",
            "pallet_standard",
            "gas_cylinder_industrial",
            "angle_iron",
            "i_beam_segment"
        ],
        "target_images": 1500,
        "imgsz": 640,
        "use_sahi": False,
        "augmentations": {
            "flip_horizontal": True,
            "flip_vertical": False,
            "rotation": [-15, 15],
            "brightness": [-40, 40],
            "blur": [0, 2.0],
            "noise": [0, 1.0],
            "mosaic": False
        },
        "public_sources": [
            "https://universe.roboflow.com/gas-cylinder/gas-cylinder-detection"
            # NOTE: Heavy materials have sparse public data.
            # Supplement with manual collection (see README for strategy).
        ]
    },

    "medical_pharma": {
        "display_name": "Model 4 - Medical and Pharma Supplies",
        "project_id":   "ensemble-medical-pharma",
        "description":  "Detection of syringes, IV bags, pills, surgical supplies and medical equipment",
        "annotation_type": "object-detection",
        "license":      "MIT",
        "classes": [
            "syringe_5ml",
            "syringe_20ml",
            "iv_bag_500ml",
            "iv_bag_1000ml",
            "pill_blister_pack",
            "medicine_bottle_amber",
            "surgical_glove_box",
            "mask_n95",
            "bandage_roll",
            "specimen_container"
        ],
        "target_images": 3000,
        "imgsz": 640,
        "use_sahi": False,
        "augmentations": {
            "flip_horizontal": True,
            "flip_vertical": False,
            "rotation": [-5, 5],
            "brightness": [-20, 20],
            "blur": [0, 0.5],
            "noise": [0, 1.0],
            "mosaic": False
        },
        "public_sources": [
            "https://universe.roboflow.com/medical-equipment/syringe-detection"
            # NOTE: medical-waste/medical-waste-detection only has 2 images — skip.
            # Supplement with manual collection from hospital/pharma public image banks.
        ]
    },

    "irregular_specialized": {
        "display_name": "Model 5 - Irregular and Specialized Goods",
        "project_id":   "ensemble-irregular-specialized",
        "description":  "Detection of cables, hoses, circuit boards, filters and irregular industrial items",
        "annotation_type": "object-detection",
        "license":      "MIT",
        "classes": [
            "cable_bundle_loose",
            "hose_coiled",
            "filter_cartridge",
            "gasket_rubber",
            "seal_o_ring_large",
            "insulation_foam_sheet",
            "circuit_board_populated",
            "transformer_small",
            "pump_unit_centrifugal",
            "valve_ball_2inch"
        ],
        "target_images": 2000,
        "imgsz": 640,
        "use_sahi": False,
        "augmentations": {
            "flip_horizontal": True,
            "flip_vertical": False,
            "rotation": [-90, 90],
            "brightness": [-30, 30],
            "blur": [0, 1.5],
            "noise": [0, 2.0],
            "mosaic": True
        },
        "public_sources": [
            "https://universe.roboflow.com/learningcomputervision/pcb-collect",
            "https://universe.roboflow.com/trcproject/e-waste-detection-model",
            "https://universe.roboflow.com/electronic-waste-detection/e-waste-dataset",
            "https://universe.roboflow.com/learningcomputervision/e-collect"
        ]
    }
}

# Label normalization map — maps ACTUAL source class names to unified target classes
# Built from inspecting data.yaml files in the downloaded datasets
GLOBAL_LABEL_MAP = {
    # ─── SMALL PARTS (fastener-object-detection classes) ───────────────────
    # automated-disassembly/fastener-object-detection
    "combo-screw":       "screw_phillips_m4",
    "flat-screw":        "screw_phillips_m4",
    "hex-bolt":          "bolt_hex_m6",
    "hex-screw":         "screw_allen_m5",
    "phillips-screw":    "screw_phillips_m4",
    "square-screw":      "screw_phillips_m4",
    "undefined":         None,                # drop unmapped/unknown
    # jente-kleinh/nuts-and-bolts
    "Bolt":              "bolt_hex_m6",
    "bolt":              "bolt_hex_m6",
    "Nut":               "nut_hex_standard",
    "nut":               "nut_hex_standard",
    # mmmxd/yolo-screws-9iuno
    "screw":             "screw_phillips_m4",
    "Screw":             "screw_phillips_m4",
    "pin":               "rivet_pop",
    "Pin":               "rivet_pop",
    # Generic aliases
    "Washer":            "washer_flat",
    "washer":            "washer_flat",
    "Rivet":             "rivet_pop",
    "rivet":             "rivet_pop",
    "Allen":             "screw_allen_m5",
    "Lock nut":          "nut_lock",
    "lock_nut":          "nut_lock",

    # ─── MRO TOOLS (tools-recognition classes) ─────────────────────────────
    # rafael-aditama/tools-recognition — only has generic "tools" class
    "tools":             "wrench_open_end",   # best-effort generic mapping
    "Tools":             "wrench_open_end",
    # Generic tool aliases
    "Wrench":            "wrench_open_end",
    "wrench":            "wrench_open_end",
    "Hammer":            "hammer_claw",
    "hammer":            "hammer_claw",
    "Pliers":            "pliers_standard",
    "pliers":            "pliers_standard",
    "Screwdriver":       "screwdriver_flat",
    "screwdriver":       "screwdriver_flat",
    "Tape measure":      "tape_measure",
    "tape_measure":      "tape_measure",
    "Caliper":           "caliper_vernier",
    "caliper":           "caliper_vernier",
    "Drill bit":         "drill_bit_set",
    "drill_bit":         "drill_bit_set",

    # ─── MEDICAL/PHARMA ─────────────────────────────────────────────────────
    "Syringe":           "syringe_5ml",
    "syringe":           "syringe_5ml",
    "syringes":          "syringe_5ml",     # dataset-t7hz7/syringes
    "Syringes":          "syringe_5ml",
    "syringe-needle":    "syringe_5ml",     # syringe-16nka/syringe-vyhfk
    "0":                 "syringe_5ml",     # syringe-16nka/syringe-vyhfk class '0'
    "dry":               "syringe_5ml",     # shirley-chia: dry-tip syringe state
    "wet":               "syringe_5ml",     # shirley-chia: wet-tip syringe state
    "injection":         "syringe_5ml",
    "needle":            "syringe_5ml",
    "Needle":            "syringe_5ml",
    "IV Bag":            "iv_bag_500ml",
    "iv_bag":            "iv_bag_500ml",
    "drip_bag":          "iv_bag_500ml",
    "N95":               "mask_n95",
    "mask":              "mask_n95",
    "Mask":              "mask_n95",
    "mask_n95":          "mask_n95",
    "Pill":              "pill_blister_pack",
    "pill":              "pill_blister_pack",
    "blister":           "pill_blister_pack",
    "medicine_bottle":   "medicine_bottle_amber",
    "bandage":           "bandage_roll",
    "Bandage":           "bandage_roll",
    "glove":             "surgical_glove_box",
    "Gloves":            "surgical_glove_box",
    "Glove":             "surgical_glove_box",
    "gloves":            "surgical_glove_box",
    "container":         "specimen_container",
    "scissors":          None,
    "cotton":            None,
    "vial":              "medicine_bottle_amber",
    "Vial":              "medicine_bottle_amber",
    # biomedical/biomedical-waste-detection (common classes in this type of dataset)
    "biomedical-waste":  "syringe_5ml",
    "cotton-ball":       None,
    "iv-tube":           "iv_bag_500ml",
    "IV-tube":           "iv_bag_500ml",
    "medical-waste":     "syringe_5ml",
    "sharps":            "syringe_5ml",
    "Sharps":            "syringe_5ml",
    "ampoule":           "medicine_bottle_amber",
    "Ampoule":           "medicine_bottle_amber",
    "bottle":            "medicine_bottle_amber",
    "Bottle":            "medicine_bottle_amber",
    "medicine":          "medicine_bottle_amber",
    "Medicine":          "medicine_bottle_amber",
    "tablet":            "pill_blister_pack",
    "Tablet":            "pill_blister_pack",
    "capsule":           "pill_blister_pack",
    "Capsule":           "pill_blister_pack",
    "bandage-roll":      "bandage_roll",
    "surgical-gloves":   "surgical_glove_box",
    "specimen":          "specimen_container",
    "Specimen":          "specimen_container",

    # ─── HEAVY MATERIALS ────────────────────────────────────────────────────
    "Pallet":            "pallet_standard",
    "pallet":            "pallet_standard",
    "Drum":              "drum_200L",
    "drum":              "drum_200L",
    "Barrel":            "drum_200L",
    "barrel":            "drum_200L",
    "Pipe":              "pipe_pvc_50mm",
    "pipe":              "pipe_pvc_50mm",
    "gas_cylinder":      "gas_cylinder_industrial",
    "Gas Cylinder":      "gas_cylinder_industrial",
    "cylinder":          "gas_cylinder_industrial",
    "Cable Reel":        "cable_reel_small",
    "cable_reel":        "cable_reel_small",
    "Sheet":             "sheet_metal_roll",
    "I-beam":            "i_beam_segment",
    "Angle iron":        "angle_iron",
    # logistics/pallet-detection-v2
    "pallet-detection-v2": "pallet_standard",
    "Pallet Detection":  "pallet_standard",
    "pallets":           "pallet_standard",
    "Pallets":           "pallet_standard",
    "euro-pallet":       "pallet_standard",
    "EUR-pallet":        "pallet_standard",
    "block-pallet":      "pallet_standard",
    # industrial/pipe-inspection
    "pipe-defect":       "pipe_pvc_50mm",
    "pipe-crack":        "pipe_pvc_50mm",
    "pipe-corrosion":    "pipe_pvc_50mm",
    "pipe-joint":        "pipe_pvc_50mm",
    "pipe-weld":         "pipe_steel_100mm",
    "steel-pipe":        "pipe_steel_100mm",
    "pvc-pipe":          "pipe_pvc_50mm",
    # roboflow-universe-projects/forklift-detection
    "forklift":          "pallet_standard",   # co-occurs with pallets
    "Forklift":          "pallet_standard",
    "pallet-on-forklift": "pallet_standard",

    # ─── IRREGULAR / SPECIALIZED (e-waste / PCB datasets) ──────────────────
    # learningcomputervision/pcb-collect
    "Capacitor":         "circuit_board_populated",
    "Diode":             "circuit_board_populated",
    "Inductor":          "circuit_board_populated",
    "LED":               "circuit_board_populated",
    "Resistor":          "circuit_board_populated",
    "battery":           "transformer_small",
    "Battery":           "transformer_small",
    "buck":              "circuit_board_populated",
    "resistor":          "circuit_board_populated",
    "transistor":        "circuit_board_populated",
    # trcproject/e-waste-detection-model
    "Printed Circuit Board PCB": "circuit_board_populated",
    "HDD":               "circuit_board_populated",
    "NetworkSwitch":     "valve_ball_2inch",
    "Router":            "pump_unit_centrifugal",
    "Smart Phone":       None,
    "USB Flash Drive":   "filter_cartridge",
    "Remote control":    None,
    "computer mouse":    None,
    "Keyboard":          None,
    # learningcomputervision/e-collect
    "cable":             "cable_bundle_loose",
    "Cable":             "cable_bundle_loose",
    "pcb":               "circuit_board_populated",
    "Motherboard":       "circuit_board_populated",
    "Apple computer motherboard": "circuit_board_populated",
    "Arduino microcontroller":    "circuit_board_populated",
    "ESP32-microcontroller-pcb":  "circuit_board_populated",
    "Motor-driver-pcb":           "circuit_board_populated",
    "Raspberry PI":               "circuit_board_populated",
    "Keyboard motherboard":       "circuit_board_populated",
    "AC-to-DC-power-adapter":     "transformer_small",
    "Power Adapter":              "transformer_small",
    "adapter":                    "transformer_small",
    "Computer power supply unit": "transformer_small",
    "Hard drive":                 "circuit_board_populated",
    "memory":                     "circuit_board_populated",
    "Charger":                    "cable_bundle_loose",
    "Charger-WEEE Recycling":     "cable_bundle_loose",
    "Top-plug":                   "cable_bundle_loose",
    # Drop items not mappable to industrial classes
    "Cellphone":         None,
    "Desktop Monitor":   None,
    "Laptop":            None,
    "TV":                None,
    "Speaker":           None,
    "Mouse":             None,
    "Printer":           None,
    "Radio":             None,
    "remote":            None,
    "phone":             None,
    "Laptop Keyboard":   None,
    "Desktop Keyboard":  None,
    "D-Link Router":     None,
    "Logitech mouse":    None,
    "mobile phone":      None,
    "Mobile phone":      None,
    "Nokia 1200":        None,
    "Nokia 5130":        None,
    "Samsung Duos":      None,
    "Feature phone":     None,
    "Gaming device-Shredding and Seperation": None,
    "Electronic":        None,
    "dvd player":        None,
    "DVD-TV Remote":     None,
    "Blow dry machine":  None,
    "Hair dryer":        None,
    "Water heater":      None,
    "Multimeter-probe":  None,
    "Pendrive-Secure data erasure and WEEE Recycling": None,
    "Printer-Toner Catridge Recycling": None,
    "Speaker-Speaker Driver Recycling": None,
    "Remote-Disassembly and Material Recovery": None,
    "Router-Disassembly and Material Recovery": None,
    "Keyboard-Shredding and Seperation": None,
    "CPU-Shredding and Seperation": None,
    "Camera-WEEE Recycling": None,
    "CPU":               "circuit_board_populated",
    "Camera-WEEE Recycling": None,
    "Desktop-Donation and Refurbishment": None,
    "Laptop-Donation and Refurbishment": None,
    "Phone-Refurbishment and Recycling": None,
    "Motherboard-Recycling and Material Reclamation": "circuit_board_populated",
}


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

class RoboflowWorkspaceManager:
    """
    Full lifecycle manager for the 5-model ensemble Roboflow workspace.
    Handles creation, upload, augmentation config, and export.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.rf = Roboflow(api_key=api_key)
        self.workspace = self.rf.workspace()
        self.log = []
        print(f"\n✅ Connected to Roboflow workspace: {self.workspace.name}\n")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        Strips characters that the Roboflow API rejects in project names.
        Keeps only alphanumeric characters, spaces, and hyphens.
        Truncates to 50 characters (API limit).
        """
        import re
        # Replace em-dash, en-dash with hyphen; ampersand with 'and'
        name = name.replace("\u2014", "-").replace("\u2013", "-").replace("&", "and")
        # Strip any remaining non-alphanumeric (except space and hyphen)
        name = re.sub(r"[^A-Za-z0-9 \-]", "", name)
        # Collapse multiple spaces/hyphens
        name = re.sub(r"[ \-]+", " ", name).strip()
        return name[:50]

    def setup_all_projects(self):
        """
        Creates all 5 projects in your Roboflow workspace.
        Safe to re-run — skips if project already exists.
        """
        print("=" * 60)
        print("  SETTING UP 5 ENSEMBLE PROJECTS")
        print("=" * 60)

        results = {}
        for model_key, config in ENSEMBLE_CONFIG.items():
            safe_name = self._sanitize_name(config["display_name"])
            print(f"\n📁 Creating: {config['display_name']}")
            print(f"   API name : '{safe_name}'")
            try:
                project = self.workspace.create_project(
                    project_name=safe_name,
                    project_type=config["annotation_type"],
                    project_license=config["license"],
                    annotation="box"
                )
                results[model_key] = {
                    "status": "created",
                    "project_id": project.id,
                    "url": f"https://app.roboflow.com/{self.workspace.url}/{project.id}"
                }
                print(f"   ✅ Created → {results[model_key]['url']}")
            except Exception as e:
                err_str = str(e)
                if "already exists" in err_str.lower():
                    results[model_key] = {"status": "exists", "note": err_str}
                    print(f"   ℹ️  Already exists — skipping")
                elif "422" in err_str:
                    results[model_key] = {"status": "error", "error": err_str}
                    print(f"   ❌ 422 Error — API rejected the request.")
                    print(f"      Likely cause: project name or license invalid.")
                    print(f"      Safe name used: '{safe_name}'")
                    print(f"      Full error: {err_str}")
                else:
                    results[model_key] = {"status": "error", "error": err_str}
                    print(f"   ❌ Error: {e}")

            time.sleep(1)  # Rate limit safety

        self._save_setup_log(results)
        self._print_summary(results)
        return results

    def upload_dataset(
        self,
        model_key: str,
        dataset_dir: str,
        batch_name: Optional[str] = None,
        split: str = "train"
    ):
        """
        Uploads a YOLO-format dataset directory to the specified model project.

        Args:
            model_key    : one of the 5 model keys
            dataset_dir  : path to directory with images/ and labels/ subdirs
            batch_name   : optional batch tag for tracking
            split        : 'train', 'valid', or 'test'
        """
        config = ENSEMBLE_CONFIG[model_key]
        dataset_path = Path(dataset_dir)

        print(f"\n{'='*60}")
        print(f"  UPLOADING TO: {config['display_name']}")
        print(f"  Source: {dataset_path}")
        print(f"{'='*60}")

        # Validate directory structure
        images_dir = dataset_path / "images" / split
        labels_dir = dataset_path / "labels" / split

        if not images_dir.exists():
            images_dir = dataset_path / split / "images"
            labels_dir = dataset_path / split / "labels"

        if not images_dir.exists():
            print(f"❌ Cannot find images directory. Expected: {images_dir}")
            return False

        image_files = list(images_dir.glob("*.jpg")) + \
                      list(images_dir.glob("*.png")) + \
                      list(images_dir.glob("*.jpeg"))

        print(f"📸 Found {len(image_files)} images to upload")

        # Load project
        try:
            project = self.workspace.project(config["project_id"])
        except Exception as e:
            print(f"❌ Project not found: {config['project_id']}. Run --mode setup first.")
            return False

        # Upload with progress bar
        batch = batch_name or f"batch_{time.strftime('%Y%m%d_%H%M')}"
        uploaded, failed = 0, 0

        for img_path in tqdm(image_files, desc=f"Uploading {model_key}"):
            label_path = labels_dir / (img_path.stem + ".txt")

            try:
                if label_path.exists():
                    project.upload(
                        image_path=str(img_path),
                        annotation_path=str(label_path),
                        split=split,
                        num_retry_uploads=3,
                        batch_name=batch,
                        tag_names=[model_key, split],
                        is_prediction=False
                    )
                else:
                    # Upload image only (unannotated)
                    project.upload(
                        image_path=str(img_path),
                        split=split,
                        num_retry_uploads=3,
                        batch_name=batch,
                        tag_names=[model_key, split, "unannotated"]
                    )
                uploaded += 1
            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"\n   ⚠️  Failed: {img_path.name} — {e}")

            time.sleep(0.1)  # Respect rate limits

        print(f"\n✅ Upload complete: {uploaded} succeeded, {failed} failed")
        print(f"   Batch tag: {batch}")
        return uploaded > 0

    def generate_dataset_version(
        self,
        model_key: str,
        version_name: Optional[str] = None
    ):
        """
        Generates a versioned dataset with augmentations applied.
        This locks the dataset for training.
        """
        config = ENSEMBLE_CONFIG[model_key]
        aug = config["augmentations"]

        print(f"\n🔄 Generating dataset version for: {config['display_name']}")

        try:
            project = self.workspace.project(config["project_id"])

            augmentation_config = {
                "flip": {
                    "horizontal": aug["flip_horizontal"],
                    "vertical":   aug["flip_vertical"]
                },
                "rotation": {
                    "enable": True,
                    "degrees": max(abs(aug["rotation"][0]), abs(aug["rotation"][1]))
                },
                "brightness": {
                    "enable": True,
                    "brighten": aug["brightness"][1],
                    "darken":   abs(aug["brightness"][0])
                },
                "blur": {
                    "enable": True,
                    "pixels": aug["blur"][1]
                },
                "noise": {
                    "enable": True,
                    "percent": aug["noise"][1]
                }
            }

            version = project.generate_version(settings=augmentation_config)
            print(f"✅ Version generated: v{version.version}")
            return version

        except Exception as e:
            print(f"❌ Version generation failed: {e}")
            return None

    def export_all_datasets(self, format: str = "yolov11", output_dir: str = "./exported_datasets"):
        """
        Exports all 5 model datasets in YOLO11 format for training.
        """
        out = Path(output_dir)
        out.mkdir(exist_ok=True)

        print(f"\n{'='*60}")
        print(f"  EXPORTING ALL 5 DATASETS → {output_dir}")
        print(f"{'='*60}")

        export_paths = {}
        for model_key, config in ENSEMBLE_CONFIG.items():
            print(f"\n📦 Exporting: {config['display_name']}")
            try:
                project = self.workspace.project(config["project_id"])
                versions = project.versions()

                if not versions:
                    print(f"   ⚠️  No versions found — skipping")
                    continue

                # Export latest version
                latest = versions[-1]
                dataset = latest.download(
                    model_format=format,
                    location=str(out / model_key)
                )
                export_paths[model_key] = str(out / model_key)
                print(f"   ✅ Exported to: {out / model_key}")

            except Exception as e:
                print(f"   ❌ Export failed: {e}")

        # Auto-generate training config YAML
        self._write_training_manifest(export_paths, output_dir)
        return export_paths

    def check_status(self):
        """
        Prints current status of all 5 projects — image count, class count, versions.
        """
        print(f"\n{'='*60}")
        print(f"  ENSEMBLE PROJECT STATUS REPORT")
        print(f"  Workspace: {self.workspace.name}")
        print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        for model_key, config in ENSEMBLE_CONFIG.items():
            print(f"\n{'─'*40}")
            print(f"  {config['display_name']}")
            print(f"  Project ID : {config['project_id']}")
            print(f"  Target     : {config['target_images']} images")
            print(f"  Classes    : {len(config['classes'])}")
            print(f"  SAHI       : {'Yes' if config['use_sahi'] else 'No'}")
            print(f"  Input Size : {config['imgsz']}px")
            print(f"  Classes    :")
            for i, cls in enumerate(config["classes"], 1):
                print(f"    {i:2}. {cls}")

            try:
                project = self.workspace.project(config["project_id"])
                print(f"\n  Roboflow Stats:")
                print(f"    Images   : {project.images}")
                print(f"    Versions : {len(project.versions())}")
                print(f"    URL      : https://app.roboflow.com/{self.workspace.url}/{config['project_id']}")
            except Exception as e:
                print(f"\n  ⚠️  Not yet created on Roboflow (run --mode setup)")

    def download_public_datasets(self, model_key: str, output_dir: str = "./raw_downloads"):
        """
        Downloads public Roboflow Universe datasets for a given model.
        These can then be merged using dataset_merger.py
        """
        config = ENSEMBLE_CONFIG[model_key]
        out = Path(output_dir) / model_key
        out.mkdir(parents=True, exist_ok=True)

        print(f"\n📥 Downloading public datasets for: {config['display_name']}")
        print(f"   Output dir: {out}")
        print(f"   Sources: {len(config['public_sources'])} datasets")

        download_log = []
        for i, url in enumerate(config["public_sources"], 1):
            print(f"\n   [{i}/{len(config['public_sources'])}] {url}")
            # Parse workspace/project from URL
            parts = url.rstrip("/").split("/")
            if len(parts) >= 2:
                ws_name = parts[-2]
                proj_name = parts[-1]
                try:
                    ext_project = self.rf.workspace(ws_name).project(proj_name)
                    versions = ext_project.versions()
                    if versions:
                        dest = str(out / f"source_{i}_{proj_name}")
                        versions[-1].download("yolov11", location=dest)
                        download_log.append({"source": url, "dest": dest, "status": "ok"})
                        print(f"   ✅ Downloaded to: {dest}")
                    else:
                        print(f"   ⚠️  No versions available")
                except Exception as e:
                    download_log.append({"source": url, "error": str(e), "status": "failed"})
                    print(f"   ❌ Failed: {e}")
            time.sleep(2)

        log_file = out / "download_log.json"
        with open(log_file, "w") as f:
            json.dump(download_log, f, indent=2)
        print(f"\n📋 Download log saved: {log_file}")

    def _write_training_manifest(self, export_paths: dict, output_dir: str):
        """Writes a master training manifest pointing to all 5 datasets."""
        manifest = {
            "project": "Industrial Ensemble Object Detection",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": {}
        }

        for model_key, path in export_paths.items():
            config = ENSEMBLE_CONFIG[model_key]
            data_yaml = Path(path) / "data.yaml"
            manifest["models"][model_key] = {
                "display_name": config["display_name"],
                "data_yaml":    str(data_yaml),
                "imgsz":        config["imgsz"],
                "use_sahi":     config["use_sahi"],
                "classes":      config["classes"],
                "target_images": config["target_images"]
            }

        out = Path(output_dir) / "training_manifest.yaml"
        with open(out, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        print(f"\n📋 Training manifest saved: {out}")

    def _save_setup_log(self, results: dict):
        log_path = Path("roboflow_setup_log.json")
        with open(log_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n📋 Setup log saved: {log_path}")

    def _print_summary(self, results: dict):
        print(f"\n{'='*60}")
        print("  SETUP SUMMARY")
        print(f"{'='*60}")
        created = sum(1 for r in results.values() if r.get("status") == "created")
        exists  = sum(1 for r in results.values() if r.get("status") == "exists")
        errors  = sum(1 for r in results.values() if r.get("status") == "error")
        print(f"  ✅ Created : {created}")
        print(f"  ℹ️  Exists  : {exists}")
        print(f"  ❌ Errors  : {errors}")
        print(f"{'='*60}\n")


# =============================================================================
# DATASET MERGER (standalone utility)
# =============================================================================

class DatasetMerger:
    """
    Merges multiple YOLO-format datasets with label normalization.
    Handles class remapping, split stratification, and deduplication.
    """

    def __init__(self, output_dir: str, model_key: str):
        self.output_dir = Path(output_dir)
        self.model_key  = model_key
        self.config     = ENSEMBLE_CONFIG[model_key]
        self.final_classes = self.config["classes"]
        self.class_to_id   = {c: i for i, c in enumerate(self.final_classes)}
        self.stats = {c: 0 for c in self.final_classes}
        self.seen_hashes   = set()
        self.total_images  = 0

        for split in ["train", "valid", "test"]:
            (self.output_dir / split / "images").mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    def add_source(self, source_dir: str, split_ratio=(0.80, 0.10, 0.10)):
        source = Path(source_dir)
        images = (
            list((source / "train" / "images").glob("*.*")) +
            list((source / "valid" / "images").glob("*.*")) +
            list((source / "images").glob("*.*"))
        )
        images = [i for i in images if i.suffix.lower() in {".jpg", ".jpeg", ".png"}]

        # Load source class list
        yaml_file = source / "data.yaml"
        src_classes = []
        if yaml_file.exists():
            with open(yaml_file) as f:
                src_data = yaml.safe_load(f)
            src_classes = src_data.get("names", [])

        print(f"   📁 Source: {source.name} — {len(images)} images, {len(src_classes)} classes")

        import random
        random.shuffle(images)
        n = len(images)
        splits = {
            "train": images[:int(n * split_ratio[0])],
            "valid": images[int(n * split_ratio[0]):int(n * (split_ratio[0]+split_ratio[1]))],
            "test":  images[int(n * (split_ratio[0]+split_ratio[1])):]
        }

        for split, imgs in splits.items():
            for img_path in imgs:
                # Deduplication by file hash
                with open(img_path, "rb") as f:
                    h = hashlib.md5(f.read()).hexdigest()
                if h in self.seen_hashes:
                    continue
                self.seen_hashes.add(h)

                # Find label
                for lbl_search in [
                    img_path.parent.parent / "labels" / img_path.name.replace(img_path.suffix, ".txt"),
                    img_path.parent.parent.parent / "labels" / img_path.parent.name / img_path.name.replace(img_path.suffix, ".txt"),
                    img_path.with_suffix(".txt")
                ]:
                    if lbl_search.exists():
                        label_path = lbl_search
                        break
                else:
                    continue  # No label found

                # Remap labels
                new_lines = []
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        class_id = int(parts[0])
                        orig_name = src_classes[class_id] if class_id < len(src_classes) else None
                        if orig_name:
                            # 1. Try exact match in label map
                            unified = GLOBAL_LABEL_MAP.get(orig_name)
                            # 2. Try lowercased underscore version if not in map
                            if unified is None and orig_name not in GLOBAL_LABEL_MAP:
                                fallback = orig_name.lower().replace(" ", "_").replace("-", "_")
                                unified = GLOBAL_LABEL_MAP.get(fallback, fallback)
                            # 3. None means explicitly drop this class
                            if unified is None:
                                continue
                            if unified in self.class_to_id:
                                new_lines.append(f"{self.class_to_id[unified]} " + " ".join(parts[1:]))
                                self.stats[unified] += 1

                if new_lines:
                    dest_img = self.output_dir / split / "images" / img_path.name
                    dest_lbl = self.output_dir / split / "labels" / (img_path.stem + ".txt")
                    shutil.copy2(img_path, dest_img)
                    with open(dest_lbl, "w") as f:
                        f.write("\n".join(new_lines))
                    self.total_images += 1

    def finalize(self):
        """Write data.yaml and print class distribution report."""
        data_yaml = {
            "path":  str(self.output_dir.absolute()),
            "train": "train/images",
            "val":   "valid/images",
            "test":  "test/images",
            "nc":    len(self.final_classes),
            "names": self.final_classes
        }
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"\n{'='*50}")
        print(f"  ✅ DATASET READY: {self.model_key}")
        print(f"  Total images : {self.total_images}")
        print(f"  data.yaml    : {yaml_path}")
        print(f"\n  Class Distribution:")
        for cls, count in sorted(self.stats.items(), key=lambda x: -x[1]):
            bar = "█" * min(40, count // 5)
            print(f"    {cls:<35} {count:>5}  {bar}")
        print(f"{'='*50}\n")

        # Quality check
        empty = [c for c, n in self.stats.items() if n == 0]
        low   = [c for c, n in self.stats.items() if 0 < n < 300]
        if empty:
            print(f"⚠️  EMPTY CLASSES (0 annotations): {empty}")
            print(f"   → These classes will be dropped during training!")
        if low:
            print(f"⚠️  LOW-DATA CLASSES (<300 annotations): {low}")
            print(f"   → Consider adding more data or merging similar classes")

        return yaml_path


# =============================================================================
# CLI ENTRYPOINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Roboflow Workspace Setup for 5-Model Ensemble",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initial setup — create all 5 projects
  python rf_workspace_setup.py --api_key YOUR_KEY --mode setup

  # Check status of all projects
  python rf_workspace_setup.py --api_key YOUR_KEY --mode status

  # Download public datasets for a model
  python rf_workspace_setup.py --api_key YOUR_KEY --mode download --model small_parts

  # Upload your merged dataset
  python rf_workspace_setup.py --api_key YOUR_KEY --mode upload --model medical_pharma --dataset ./datasets/medical_final

  # Generate versioned dataset (locks it for training)
  python rf_workspace_setup.py --api_key YOUR_KEY --mode version --model mro_tools

  # Export all datasets for training
  python rf_workspace_setup.py --api_key YOUR_KEY --mode export_all --output ./training_datasets

  # Merge multiple raw sources for a model
  python rf_workspace_setup.py --api_key YOUR_KEY --mode merge --model small_parts --sources ./raw/source1 ./raw/source2 --output ./datasets/small_parts_merged
        """
    )
    parser.add_argument("--api_key", required=True, help="Your Roboflow API key")
    parser.add_argument("--mode", required=True,
                        choices=["setup", "status", "upload", "download", "version", "export_all", "merge"],
                        help="Operation to perform")
    parser.add_argument("--model", choices=list(ENSEMBLE_CONFIG.keys()),
                        help="Target model key (required for upload/download/version/merge)")
    parser.add_argument("--dataset", help="Path to YOLO dataset directory (for upload)")
    parser.add_argument("--sources", nargs="+", help="Source dataset dirs (for merge)")
    parser.add_argument("--output", default="./merged_output", help="Output directory")
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"],
                        help="Dataset split to upload")
    parser.add_argument("--batch", help="Custom batch name for upload tracking")
    parser.add_argument("--format", default="yolov11", help="Export format")

    args = parser.parse_args()

    manager = RoboflowWorkspaceManager(api_key=args.api_key)

    if args.mode == "setup":
        manager.setup_all_projects()

    elif args.mode == "status":
        manager.check_status()

    elif args.mode == "upload":
        if not args.model or not args.dataset:
            print("❌ --model and --dataset required for upload mode")
            sys.exit(1)
        manager.upload_dataset(args.model, args.dataset, args.batch, args.split)

    elif args.mode == "download":
        if not args.model:
            print("❌ --model required for download mode")
            sys.exit(1)
        manager.download_public_datasets(args.model, args.output)

    elif args.mode == "version":
        if not args.model:
            print("❌ --model required for version mode")
            sys.exit(1)
        manager.generate_dataset_version(args.model)

    elif args.mode == "export_all":
        manager.export_all_datasets(args.format, args.output)

    elif args.mode == "merge":
        if not args.model or not args.sources:
            print("❌ --model and --sources required for merge mode")
            sys.exit(1)
        merger = DatasetMerger(args.output, args.model)
        for src in args.sources:
            merger.add_source(src)
        merger.finalize()


if __name__ == "__main__":
    main()
