#!/bin/bash
# =============================================================================
#  ENSEMBLE PIPELINE QUICK-START
#  Replace YOUR_ROBOFLOW_API_KEY before running
# =============================================================================

API_KEY="YOUR_ROBOFLOW_API_KEY"
SCRIPT="python rf_workspace_setup.py --api_key $API_KEY"

echo "============================================"
echo "  ENSEMBLE ROBOFLOW PIPELINE"
echo "============================================"

# Step 1: Create all 5 projects
echo -e "\n[STEP 1] Creating workspace projects..."
$SCRIPT --mode setup

# Step 2: Download public datasets for each model
echo -e "\n[STEP 2] Downloading public datasets..."
for MODEL in small_parts mro_tools heavy_materials medical_pharma irregular_specialized; do
    echo "  → Downloading for: $MODEL"
    $SCRIPT --mode download --model $MODEL --output ./raw_downloads
done

# Step 3: Merge datasets per model
echo -e "\n[STEP 3] Merging datasets..."
for MODEL in small_parts mro_tools heavy_materials medical_pharma irregular_specialized; do
    echo "  → Merging: $MODEL"
    $SCRIPT --mode merge \
        --model $MODEL \
        --sources ./raw_downloads/$MODEL/source_1_* \
        --output ./datasets/${MODEL}_final
done

# Step 4: Upload merged datasets
echo -e "\n[STEP 4] Uploading to Roboflow..."
for MODEL in small_parts mro_tools heavy_materials medical_pharma irregular_specialized; do
    echo "  → Uploading: $MODEL"
    $SCRIPT --mode upload \
        --model $MODEL \
        --dataset ./datasets/${MODEL}_final \
        --split train \
        --batch "batch_$(date +%Y%m%d)"
done

# Step 5: Check final status
echo -e "\n[STEP 5] Final status check..."
$SCRIPT --mode status

echo -e "\n✅ Pipeline complete!"
echo "   Next: Generate dataset versions on Roboflow UI, then run training."
