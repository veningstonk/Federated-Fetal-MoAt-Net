#!/usr/bin/env bash
# scripts/download_data.sh
# ─────────────────────────────────────────────────────────────────────────────
# Instructions for downloading the two public fetal ultrasound datasets.
# Both datasets are publicly available under open licenses.
#
# FPUS23:          https://github.com/bharathprabakaran/FPUS23
# FETAL_PLANES_DB: https://doi.org/10.5281/zenodo.3904280
# ─────────────────────────────────────────────────────────────────────────────

set -e

DATA_DIR="data"
mkdir -p "$DATA_DIR"

echo "================================================================"
echo "  Fetal Ultrasound Dataset Download Instructions"
echo "================================================================"
echo ""
echo "DATASET 1: FPUS23"
echo "-----------------"
echo "Source  : https://github.com/bharathprabakaran/FPUS23"
echo "License : IEEE DataPort open access"
echo ""
echo "Steps:"
echo "  1. Visit https://github.com/bharathprabakaran/FPUS23"
echo "  2. Download the dataset archive (FPUS23.zip or via the IEEE DataPort link)"
echo "  3. Extract and organise as follows:"
echo ""
echo "     data/FPUS23/"
echo "       AC_PLANE/    ← Abdominal Circumference images"
echo "       BPD_PLANE/   ← Biparietal Diameter images"
echo "       FL_PLANE/    ← Femur Length images"
echo "       NO_PLANE/    ← Non-diagnostic frames"
echo ""
echo "  Alternatively, if the repository provides per-class folders already,"
echo "  simply rename them to match the above structure."
echo ""
echo "  Expected total images: ~5,265 (subset of 15,728 total)"
echo ""

echo "================================================================"
echo "DATASET 2: FETAL_PLANES_DB"
echo "--------------------------"
echo "Source  : https://zenodo.org/record/3904280"
echo "DOI     : 10.5281/zenodo.3904280"
echo "License : Creative Commons Attribution 4.0"
echo ""
echo "Steps:"
echo "  Option A — wget/curl:"
echo '    wget "https://zenodo.org/record/3904280/files/FETAL_PLANES_DB.zip" -O data/FETAL_PLANES_DB.zip'
echo '    unzip data/FETAL_PLANES_DB.zip -d data/'
echo ""
echo "  Option B — Manual download:"
echo "    Visit https://zenodo.org/record/3904280"
echo "    Download FETAL_PLANES_DB.zip and extract to data/"
echo ""
echo "  Organise the extracted files as:"
echo ""
echo "     data/FETAL_PLANES_DB/"
echo "       Fetal_Abdomen/"
echo "       Fetal_Brain/"
echo "       Fetal_Femur/"
echo "       Fetal_Thorax/"
echo "       Maternal_Cervix/"
echo "       Other/"
echo ""
echo "  Expected total images: 12,400"
echo ""
echo "================================================================"
echo "Verification"
echo "────────────"
echo "After downloading, run:"
echo ""
echo "  python -c \""
echo "  from data.dataset import FetalUltrasoundDataset"
echo "  ds = FetalUltrasoundDataset('data', 'fpus23', 'train')"
echo "  print('FPUS23 train samples:', len(ds))"
echo "  ds2 = FetalUltrasoundDataset('data', 'fetal_planes_db', 'train')"
echo "  print('FETAL_PLANES_DB train samples:', len(ds2))"
echo "  \""
echo ""
echo "================================================================"

# Optional: attempt automatic download of FETAL_PLANES_DB if wget is available
if command -v wget &> /dev/null; then
    echo ""
    read -p "Attempt automatic download of FETAL_PLANES_DB via wget? [y/N] " choice
    if [[ "$choice" == "y" || "$choice" == "Y" ]]; then
        mkdir -p "$DATA_DIR"
        echo "Downloading FETAL_PLANES_DB ..."
        wget -q --show-progress \
            "https://zenodo.org/record/3904280/files/FETAL_PLANES_DB.zip" \
            -O "$DATA_DIR/FETAL_PLANES_DB.zip"
        echo "Extracting ..."
        unzip -q "$DATA_DIR/FETAL_PLANES_DB.zip" -d "$DATA_DIR/"
        echo "Done. Data saved to $DATA_DIR/FETAL_PLANES_DB/"
    fi
fi
