"""
scripts/01_download_data.py
============================
ONE-SHOT script: download and organise all required datasets.

Run once. Everything saves to  data/raw/.  No re-running needed.

    python scripts/01_download_data.py

What it downloads
-----------------
  1. DLPFC (Human Brain Dorsolateral Prefrontal Cortex)
     12 Visium slices from 3 donors.  Most-used ST benchmark.
     Source: http://spatial.libd.org/spatialLIBD/

  2. 10x Visium Mouse Brain Coronal Section
     Single section, well-annotated, free from 10x website.

  3. Mouse Olfactory Bulb (MOB)
     Stereo-seq based spatial dataset; 6-layer structure.

  4. Human Breast Cancer (HBC)
     10x Visium, invasive ductal carcinoma, 20 annotated clusters.

  5. STARmap (Mouse Visual Cortex)
     In-situ sequencing, subcellular resolution, ~1000 genes.

NOTE: DLPFC requires anndata / squidpy helpers to fetch.
      The script will print manual instructions if auto-download fails.
"""

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"


def _progress_hook(blocknum, blocksize, totalsize):
    """Simple download progress bar."""
    downloaded = blocknum * blocksize
    if totalsize > 0:
        pct = min(downloaded / totalsize * 100, 100)
        bar = int(pct / 2)
        sys.stdout.write(f"\r  [{'#' * bar}{' ' * (50-bar)}] {pct:.1f}%")
        sys.stdout.flush()
        if pct >= 100:
            print()


def download_file(url: str, dest: Path) -> bool:
    """Download url → dest, skip if already exists."""
    if dest.exists():
        print(f"  [skip] Already exists: {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        print(f"  Downloading: {dest.name}")
        print(f"  URL: {url}")
        urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
        print(f"  Saved: {dest}")
        return True
    except Exception as exc:
        print(f"  [FAILED] {exc}")
        return False


def download_dlpfc():
    """
    Download DLPFC 10x Visium slices.
    These are hosted via the spatialLIBD Bioconductor package.
    We use pre-packaged .h5ad versions from a public mirror.
    """
    print("\n--- DLPFC (Human Brain) ---")
    dlpfc_dir = DATA_RAW / "DLPFC"
    dlpfc_dir.mkdir(parents=True, exist_ok=True)

    # 12 slice IDs used in the benchmark
    slice_ids = [
        "151507", "151508", "151509", "151510",
        "151669", "151670", "151671", "151672",
        "151673", "151674", "151675", "151676",
    ]

    base_url = (
        "https://raw.githubusercontent.com/QiaoZS/STAGATE_paper/main/"
        "DLPFC/{sid}/{sid}_filtered_feature_bc_matrix.h5"
    )

    success_count = 0
    for sid in slice_ids:
        url  = base_url.format(sid=sid)
        dest = dlpfc_dir / f"{sid}.h5"
        if download_file(url, dest):
            success_count += 1

    if success_count == 0:
        print(
            "\n  [INFO] Auto-download failed (CDN may be unavailable).\n"
            "  Manual download steps:\n"
            "  1. Open R and run:\n"
            "       BiocManager::install('spatialLIBD')\n"
            "       spatialLIBD::fetch_data(type = 'spe') |> sce_to_adata() |> "
            "write_h5ad('data/raw/DLPFC/')\n"
            "  2. Or download from:\n"
            "       https://research.libd.org/spatialLIBD/\n"
            "  3. Place 12 .h5ad files in:  data/raw/DLPFC/\n"
        )
    else:
        print(f"\n  DLPFC: {success_count}/12 slices ready.")


def download_mouse_brain():
    """
    Download 10x Visium Mouse Brain Coronal section.
    Freely available from 10x Genomics website.
    """
    print("\n--- 10x Visium Mouse Brain ---")
    mb_dir = DATA_RAW / "MouseBrain"
    mb_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "filtered_feature_bc_matrix.h5":
            "https://cf.10xgenomics.com/samples/spatial-exp/1.0.0/"
            "V1_Adult_Mouse_Brain/V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5",
        "spatial.tar.gz":
            "https://cf.10xgenomics.com/samples/spatial-exp/1.0.0/"
            "V1_Adult_Mouse_Brain/V1_Adult_Mouse_Brain_spatial.tar.gz",
    }

    for fname, url in files.items():
        download_file(url, mb_dir / fname)

    # Unpack spatial folder
    tarball = mb_dir / "spatial.tar.gz"
    if tarball.exists() and not (mb_dir / "spatial").exists():
        import tarfile
        print("  Extracting spatial.tar.gz …")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(mb_dir)
        print("  Extracted.")

    print(f"\n  Mouse Brain data in: {mb_dir}")


def convert_h5_to_h5ad():
    """
    Convert raw 10x .h5 files to AnnData .h5ad format using scanpy.
    Adds spatial coordinates from the spatial/ folder.
    """
    print("\n--- Converting .h5 → .h5ad ---")
    try:
        import scanpy as sc
        import squidpy as sq
    except ImportError:
        print("  [SKIP] scanpy/squidpy not installed. Run 01_install_env first.")
        return

    PROCESSED = ROOT / "data" / "processed"

    # DLPFC
    dlpfc_raw = DATA_RAW / "DLPFC"
    if dlpfc_raw.exists():
        for h5_file in sorted(dlpfc_raw.glob("*.h5")):
            sid      = h5_file.stem
            out_path = PROCESSED / "DLPFC" / f"{sid}.h5ad"
            if out_path.exists():
                print(f"  [skip] {sid}.h5ad already done")
                continue
            try:
                adata = sc.read_10x_h5(h5_file)
                adata.var_names_make_unique()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                adata.write_h5ad(out_path)
                print(f"  {sid} → {out_path}")
            except Exception as exc:
                print(f"  [WARN] Could not convert {sid}: {exc}")

    # Mouse Brain
    mb_raw = DATA_RAW / "MouseBrain"
    h5_mb  = mb_raw / "filtered_feature_bc_matrix.h5"
    if h5_mb.exists():
        out_path = PROCESSED / "MouseBrain" / "mouse_brain.h5ad"
        if not out_path.exists():
            try:
                adata = sc.read_10x_h5(h5_mb)
                adata.var_names_make_unique()
                sq.read.visium(
                    path         = mb_raw,
                    count_file   = "filtered_feature_bc_matrix.h5",
                    library_id   = "mouse_brain",
                )
                out_path.parent.mkdir(parents=True, exist_ok=True)
                adata.write_h5ad(out_path)
                print(f"  Mouse Brain → {out_path}")
            except Exception as exc:
                print(f"  [WARN] Mouse Brain conversion: {exc}")

    # Human Breast Cancer
    hbc_raw = DATA_RAW / "HumanBreastCancer"
    h5_hbc  = hbc_raw / "filtered_feature_bc_matrix.h5"
    if h5_hbc.exists():
        out_path = PROCESSED / "HumanBreastCancer" / "breast_cancer.h5ad"
        if not out_path.exists():
            try:
                adata = sc.read_10x_h5(h5_hbc)
                adata.var_names_make_unique()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                adata.write_h5ad(out_path)
                print(f"  Human Breast Cancer → {out_path}")
            except Exception as exc:
                print(f"  [WARN] HBC conversion: {exc}")

    # Mouse Olfactory Bulb
    mob_raw = DATA_RAW / "MouseOB"
    h5_mob  = mob_raw / "MOB_filtered_feature_bc_matrix.h5"
    if h5_mob.exists():
        out_path = PROCESSED / "MouseOB" / "mouse_ob.h5ad"
        if not out_path.exists():
            try:
                adata = sc.read_10x_h5(h5_mob)
                adata.var_names_make_unique()
                out_path.parent.mkdir(parents=True, exist_ok=True)
                adata.write_h5ad(out_path)
                print(f"  Mouse OB → {out_path}")
            except Exception as exc:
                print(f"  [WARN] MOB conversion: {exc}")

    # STARmap (CSV-based)
    star_raw    = DATA_RAW / "STARmap"
    star_counts = star_raw / "cell_barcode_count.csv"
    star_coords = star_raw / "centroids.tsv"
    if star_counts.exists():
        out_path = PROCESSED / "STARmap" / "starmap.h5ad"
        if not out_path.exists():
            try:
                import pandas as pd
                import numpy as np
                counts_df = pd.read_csv(star_counts, index_col=0)
                adata = sc.AnnData(X=counts_df.values.astype(np.float32))
                adata.obs_names = [str(x) for x in counts_df.index]
                adata.var_names = [str(x) for x in counts_df.columns]
                if star_coords.exists():
                    coords = pd.read_csv(star_coords, sep="\t", header=None)
                    if coords.shape[1] >= 2:
                        adata.obsm["spatial"] = coords.iloc[:adata.n_obs, :2].values.astype(np.float64)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                adata.write_h5ad(out_path)
                print(f"  STARmap → {out_path}")
            except Exception as exc:
                print(f"  [WARN] STARmap conversion: {exc}")


def download_mouse_olfactory_bulb():
    """
    Download Mouse Olfactory Bulb (MOB) Stereo-seq data.
    Published by Chen et al. (Cell, 2022).
    Pre-processed .h5ad available from public mirrors.
    """
    print("\n--- Mouse Olfactory Bulb (MOB) ---")
    mob_dir = DATA_RAW / "MouseOB"
    mob_dir.mkdir(parents=True, exist_ok=True)

    url = (
        "https://github.com/QIFEIDKN/STAGATE_pyG/raw/main/"
        "Data/MOB/MOB_filtered_feature_bc_matrix.h5"
    )
    dest = mob_dir / "MOB_filtered_feature_bc_matrix.h5"
    if not download_file(url, dest):
        print(
            "\n  [INFO] Auto-download failed.\n"
            "  Manual: download MOB Stereo-seq data and place .h5 file in:\n"
            f"    {mob_dir}\n"
        )
    else:
        print(f"\n  MOB data in: {mob_dir}")


def download_human_breast_cancer():
    """
    Download 10x Visium Human Breast Cancer (invasive ductal carcinoma).
    Freely available from 10x Genomics website.
    """
    print("\n--- Human Breast Cancer (10x Visium) ---")
    hbc_dir = DATA_RAW / "HumanBreastCancer"
    hbc_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "filtered_feature_bc_matrix.h5":
            "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/"
            "V1_Breast_Cancer_Block_A_Section_1/"
            "V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5",
        "spatial.tar.gz":
            "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/"
            "V1_Breast_Cancer_Block_A_Section_1/"
            "V1_Breast_Cancer_Block_A_Section_1_spatial.tar.gz",
    }

    for fname, url in files.items():
        download_file(url, hbc_dir / fname)

    tarball = hbc_dir / "spatial.tar.gz"
    if tarball.exists() and not (hbc_dir / "spatial").exists():
        import tarfile
        print("  Extracting spatial.tar.gz …")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(hbc_dir)
        print("  Extracted.")

    print(f"\n  Human Breast Cancer data in: {hbc_dir}")


def download_starmap():
    """
    Download STARmap mouse visual cortex in-situ sequencing dataset.
    Wang et al. (Science, 2018).
    """
    print("\n--- STARmap (Mouse Visual Cortex) ---")
    star_dir = DATA_RAW / "STARmap"
    star_dir.mkdir(parents=True, exist_ok=True)

    # Expression matrix and metadata from original STARmap publication
    files = {
        "cell_barcode_count.csv":
            "https://raw.githubusercontent.com/weallen/STARmap/master/"
            "data/visual_1020/cell_barcode_count.csv",
        "centroids.tsv":
            "https://raw.githubusercontent.com/weallen/STARmap/master/"
            "data/visual_1020/centroids.tsv",
    }

    for fname, url in files.items():
        download_file(url, star_dir / fname)

    print(f"\n  STARmap data in: {star_dir}")


if __name__ == "__main__":
    print("=" * 55)
    print("  CauST – Data Download Script")
    print("=" * 55)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    download_dlpfc()
    download_mouse_brain()
    download_mouse_olfactory_bulb()
    download_human_breast_cancer()
    download_starmap()
    convert_h5_to_h5ad()

    print("\nDone. Check data/raw/ and data/processed/.")
