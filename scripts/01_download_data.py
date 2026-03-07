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
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"

try:
    import squidpy as sq
    _HAS_SQ = True
except ImportError:
    _HAS_SQ = False

try:
    import scanpy as sc
    _HAS_SC = True
except ImportError:
    _HAS_SC = False


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


def _dlpfc_file_is_valid(path) -> bool:
    """Return True if an h5ad file exists and has >= 1000 spots (full section)."""
    if not path.exists():
        return False
    try:
        import anndata as ad
        tmp = ad.read_h5ad(path, backed="r")
        n_obs = tmp.n_obs
        tmp.file.close()
        return n_obs >= 1000
    except Exception:
        return False


def _geo_series_ftp_url(accession: str, filename: str) -> str:
    """Build the FTP-over-HTTPS URL for a series-level supplementary file."""
    digits = accession.replace("GSE", "")
    prefix = "GSE" + digits[:-3] + "nnn"          # GSE144nnn
    return (
        f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/"
        f"{accession}/suppl/{filename}"
    )


def _download_dlpfc_geo_files(dl_dir: Path) -> bool:
    """
    Download the three series-level supplementary files from GSE144136:
      - GSE144136_GeneBarcodeMatrix_Annotated.mtx.gz  (413 MB, count matrix)
      - GSE144136_CellNames.csv.gz                    (barcodes + slice IDs)
      - GSE144136_GeneNames.csv.gz                    (gene symbols)
    Returns True if all three exist on disk afterwards.
    """
    needed = [
        "GSE144136_GeneBarcodeMatrix_Annotated.mtx.gz",
        "GSE144136_CellNames.csv.gz",
        "GSE144136_GeneNames.csv.gz",
    ]
    for fname in needed:
        dest = dl_dir / fname
        if dest.exists():
            print(f"  [skip] {fname} already downloaded")
            continue
        url = _geo_series_ftp_url("GSE144136", fname)
        if not download_file(url, dest):
            return False
    return all((dl_dir / f).exists() for f in needed)


def _split_mtx_to_h5ad(dl_dir: Path, out_dir: Path, slice_ids: list) -> int:
    """
    Load the combined MTX matrix + CSV metadata, split by slice ID
    embedded in cell names, and write one h5ad per slice.
    Returns number of slices written.
    """
    import gzip
    import pandas as pd
    import scipy.io as sio
    import anndata as ad
    import numpy as np

    mtx_path   = dl_dir / "GSE144136_GeneBarcodeMatrix_Annotated.mtx.gz"
    cells_path = dl_dir / "GSE144136_CellNames.csv.gz"
    genes_path = dl_dir / "GSE144136_GeneNames.csv.gz"

    print("  Loading GeneBarcodeMatrix_Annotated.mtx.gz …")
    mat = sio.mmread(mtx_path).T.tocsr()          # genes × cells → cells × genes

    print("  Loading CellNames.csv.gz …")
    cells_df = pd.read_csv(cells_path, header=None, compression="gzip")
    cell_names = cells_df.iloc[:, 0].astype(str).values

    print("  Loading GeneNames.csv.gz …")
    genes_df = pd.read_csv(genes_path, header=None, compression="gzip")
    gene_names = genes_df.iloc[:, 0].astype(str).values

    print(f"  Combined matrix: {mat.shape[0]} cells × {mat.shape[1]} genes")

    # ── Extract slice ID from cell names ──────────────────────────────────
    # Names typically look like: "151507_AAACAAGTATCTCCCA-1" or "AAACAAGTATCTCCCA-1_151507"
    # Try prefix first, then suffix
    slice_labels = np.array([""] * len(cell_names), dtype=object)
    for sid in slice_ids:
        mask = np.array([sid in cn for cn in cell_names])
        slice_labels[mask] = sid

    assigned = np.sum(slice_labels != "")
    print(f"  Assigned {assigned}/{len(cell_names)} cells to known slice IDs")

    if assigned == 0:
        # Show a few cell names to help debug
        print(f"  First 5 cell names: {cell_names[:5].tolist()}")
        print("  [ERROR] Could not match any cell names to slice IDs.")
        return 0

    converted = 0
    for sid in slice_ids:
        out_path = out_dir / f"{sid}.h5ad"
        if _dlpfc_file_is_valid(out_path):
            print(f"  [skip] {sid}.h5ad already valid")
            converted += 1
            continue

        mask = slice_labels == sid
        n_spots = int(mask.sum())
        if n_spots == 0:
            print(f"  [WARN] No cells found for slice {sid}")
            continue

        sub_mat   = mat[mask]
        sub_cells = cell_names[mask]

        # Strip slice prefix/suffix from barcode to get clean barcode
        barcodes = []
        for cn in sub_cells:
            bc = cn.replace(f"{sid}_", "").replace(f"_{sid}", "").replace(sid, "")
            if bc.startswith("_"):
                bc = bc[1:]
            if bc.endswith("_"):
                bc = bc[:-1]
            barcodes.append(bc if bc else cn)

        adata = ad.AnnData(
            X=sub_mat,
            obs=pd.DataFrame(index=barcodes),
            var=pd.DataFrame(index=gene_names),
        )
        adata.var_names_make_unique()
        adata.obs_names_make_unique()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(out_path)
        print(f"  {sid}: {n_spots} spots × {adata.n_vars} genes → saved")
        converted += 1

    return converted


def download_dlpfc():
    """
    Download DLPFC Visium sections by delegating to scripts/get_dlpfc.py.

    GEO accession GSE144239 (Maynard et al. 2021) provides 4 Visium sections
    (P4_rep1, P4_rep2, P6_rep1, P6_rep2).  The full 12-section dataset
    (IDs 151507-151676) is hosted by LIBD's spatialLIBD R package, not GEO.
    """
    print("\n--- DLPFC (Human Brain) ---")
    dlpfc_dir = DATA_RAW / "DLPFC"
    dlpfc_dir.mkdir(parents=True, exist_ok=True)

    # Check for ANY valid h5ad files (new P4/P6 names or old 151xxx names)
    existing = list(dlpfc_dir.glob("*.h5ad"))
    valid = [p for p in existing if _dlpfc_file_is_valid(p)]
    if len(valid) >= 4:
        names = [p.stem for p in valid]
        print(f"  [skip] {len(valid)} DLPFC sections already valid: {names}")
        return

    # Remove stale 224-spot files
    for p in existing:
        if not _dlpfc_file_is_valid(p):
            obs = _dlpfc_obs(p)
            print(f"  [stale] Removing {p.name} – only {obs} spots")
            p.unlink()

    print("  Delegating to scripts/get_dlpfc.py …")
    # Import and run the download logic
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "get_dlpfc", ROOT / "scripts" / "get_dlpfc.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def _dlpfc_obs(path) -> int:
    """Return spot count of an h5ad without fully loading it."""
    try:
        import anndata as ad
        tmp = ad.read_h5ad(path, backed="r")
        n = tmp.n_obs
        tmp.file.close()
        return n
    except Exception:
        return 0


def _print_dlpfc_manual_instructions(dlpfc_dir: Path):
    print(
        "\n  ──────────────────────────────────────────────────────────────────\n"
        "  Automated DLPFC download failed.\n"
        "\n"
        "  Run the standalone download script:\n"
        "    python scripts/get_dlpfc.py\n"
        "\n"
        "  Data source: GEO GSE144239 (Maynard et al. 2021, Nat Neuroscience)\n"
        "  GEO provides 4 Visium sections (P4_rep1, P4_rep2, P6_rep1, P6_rep2).\n"
        "\n"
        f"  Target directory: {dlpfc_dir}\n"
        "  Then run: python scripts/02_preprocess.py\n"
        "  ──────────────────────────────────────────────────────────────────\n"
    )


def download_mouse_brain():
    """
    Download 10x Visium Mouse Brain via squidpy built-in datasets.
    This bypasses the 10x Genomics CDN (which blocks automated access)
    and downloads from squidpy's own hosted mirror.
    """
    print("\n--- 10x Visium Mouse Brain ---")
    mb_dir = DATA_RAW / "MouseBrain"
    mb_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = ROOT / "data" / "processed" / "MouseBrain"
    proc_dir.mkdir(parents=True, exist_ok=True)

    out_h5ad = proc_dir / "mouse_brain.h5ad"
    if out_h5ad.exists():
        print(f"  [skip] mouse_brain.h5ad already exists")
        print(f"  Mouse Brain data in: {proc_dir}")
        return

    # ── Primary: squidpy built-in (downloads from scverse CDN, not 10x) ─────
    if _HAS_SQ:
        try:
            import warnings
            print("  Downloading via squidpy.datasets …")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                adata = sq.datasets.visium_hne_adata()
            adata.write_h5ad(out_h5ad)
            print(f"  Saved: {out_h5ad}  shape={adata.shape}")
            print(f"  Mouse Brain data in: {proc_dir}")
            return
        except Exception as exc:
            print(f"  [WARN] squidpy.datasets failed: {exc}")

    # ── Fallback: direct squidpy CDN URL ─────────────────────────────────────
    cdn_url = (
        "https://raw.githubusercontent.com/scverse/squidpy_notebooks/"
        "master/_data/tutorial_data/visium_hne.h5ad"
    )
    if download_file(cdn_url, proc_dir / "mouse_brain.h5ad"):
        print(f"  Mouse Brain data in: {proc_dir}")
        return

    print(
        "\n  [INFO] Automated download failed. Manual step:\n"
        "    python -c \"import squidpy as sq; sq.datasets.visium_hne_adata()"
        ".write_h5ad('data/processed/MouseBrain/mouse_brain.h5ad')\"\n"
    )


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
    Chen et al. (Cell, 2022). Tries multiple mirrors.
    """
    print("\n--- Mouse Olfactory Bulb (MOB) ---")
    mob_dir = DATA_RAW / "MouseOB"
    mob_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = ROOT / "data" / "processed" / "MouseOB"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Try multiple known mirrors for the Stereo-seq MOB h5ad
    mirrors = [
        "https://github.com/QIFEIDKN/STAGATE/raw/main/"
        "data/MOB/MOB_filtered_feature_bc_matrix.h5",
        "https://github.com/QIFEIDKN/STAGATE_pyG/raw/main/"
        "Tutorial_data/MOB_filtered_feature_bc_matrix.h5",
    ]
    dest = mob_dir / "MOB_filtered_feature_bc_matrix.h5"
    success = dest.exists()
    for url in mirrors:
        if success:
            break
        success = download_file(url, dest)

    if success and _HAS_SC:
        out_h5ad = proc_dir / "mouse_ob.h5ad"
        if not out_h5ad.exists():
            try:
                adata = sc.read_10x_h5(dest)
                adata.var_names_make_unique()
                adata.write_h5ad(out_h5ad)
                print(f"  Saved: {out_h5ad}")
            except Exception as exc:
                print(f"  [WARN] h5 conversion: {exc}")
    elif not success:
        print(
            "\n  [INFO] MOB auto-download failed (optional dataset).\n"
            "  Manual: download from the STAGATE tutorial and place .h5 file in:\n"
            f"    {mob_dir}/MOB_filtered_feature_bc_matrix.h5\n"
            "  Or use a squidpy Slideseq dataset as alternative:\n"
            "    python -c \"import squidpy as sq; "
            "sq.datasets.slideseqv2().write_h5ad('data/processed/MouseOB/mouse_ob.h5ad')\"\n"
        )
    print(f"  MOB data in: {mob_dir}")


def download_human_breast_cancer():
    """
    Download 10x Visium Human Breast Cancer.
    10x CDN blocks automated access; use squidpy MIBI-TOF as a comparable
    spatial dataset, or provide manual instructions for the real HBC data.
    """
    print("\n--- Human Breast Cancer (10x Visium) ---")
    hbc_dir = DATA_RAW / "HumanBreastCancer"
    hbc_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = ROOT / "data" / "processed" / "HumanBreastCancer"
    proc_dir.mkdir(parents=True, exist_ok=True)

    out_h5ad = proc_dir / "breast_cancer.h5ad"
    if out_h5ad.exists():
        print(f"  [skip] breast_cancer.h5ad already exists")
        print(f"  HBC data in: {proc_dir}")
        return

    # 10x CDN (403 expected); kept for completeness
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
        print("  Extracting spatial.tar.gz …")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(hbc_dir)
        print("  Extracted.")

    h5_file = hbc_dir / "filtered_feature_bc_matrix.h5"
    if h5_file.exists() and _HAS_SC and not out_h5ad.exists():
        try:
            adata = sc.read_10x_h5(h5_file)
            adata.var_names_make_unique()
            adata.write_h5ad(out_h5ad)
            print(f"  Saved: {out_h5ad}")
        except Exception as exc:
            print(f"  [WARN] h5 conversion: {exc}")
    elif not h5_file.exists():
        print(
            "\n  [INFO] 10x CDN blocks automated downloads (optional dataset).\n"
            "  Option A – squidpy alternative (different tissue, same format):\n"
            "    python -c \"import squidpy as sq; sq.datasets.visium_fluo_adata()"
            ".write_h5ad('data/processed/HumanBreastCancer/breast_cancer.h5ad')\"\n"
            "  Option B – manual browser download:\n"
            "    https://www.10xgenomics.com/resources/datasets/"
            "human-breast-cancer-block-a-section-1-1-standard-1-1-0\n"
            "  Download count matrix + spatial → place in:\n"
            f"    {hbc_dir}/\n"
        )

    print(f"  Human Breast Cancer data in: {hbc_dir}")


def download_starmap():
    """
    Download STARmap mouse visual cortex. Wang et al. (Science, 2018).
    Data hosted at starmapresources.com. Tries multiple mirrors.
    """
    print("\n--- STARmap (Mouse Visual Cortex) ---")
    star_dir = DATA_RAW / "STARmap"
    star_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = ROOT / "data" / "processed" / "STARmap"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Multiple mirror attempts for each file
    file_mirrors = {
        "cell_barcode_count.csv": [
            "https://raw.githubusercontent.com/weallen/STARmap/master/"
            "data/visual_1020/cell_barcode_count.csv",
            "https://raw.githubusercontent.com/wanglab-broad/starmap-py/"
            "main/data/visual_1020/cell_barcode_count.csv",
        ],
        "centroids.tsv": [
            "https://raw.githubusercontent.com/weallen/STARmap/master/"
            "data/visual_1020/centroids.tsv",
            "https://raw.githubusercontent.com/wanglab-broad/starmap-py/"
            "main/data/visual_1020/centroids.tsv",
        ],
    }

    for fname, mirrors in file_mirrors.items():
        dest = star_dir / fname
        for url in mirrors:
            if download_file(url, dest):
                break

    counts_file = star_dir / "cell_barcode_count.csv"
    coords_file = star_dir / "centroids.tsv"
    out_h5ad    = proc_dir / "starmap.h5ad"

    if counts_file.exists() and _HAS_SC and not out_h5ad.exists():
        try:
            import pandas as pd, numpy as np
            counts_df = pd.read_csv(counts_file, index_col=0)
            adata = sc.AnnData(X=counts_df.values.astype(np.float32))
            adata.obs_names = [str(x) for x in counts_df.index]
            adata.var_names = [str(x) for x in counts_df.columns]
            if coords_file.exists():
                coords = pd.read_csv(coords_file, sep="\t", header=None)
                adata.obsm["spatial"] = coords.iloc[:adata.n_obs, :2].values.astype(np.float64)
            adata.write_h5ad(out_h5ad)
            print(f"  Saved: {out_h5ad}")
        except Exception as exc:
            print(f"  [WARN] STARmap conversion: {exc}")
    elif not counts_file.exists():
        print(
            "\n  [INFO] STARmap mirrors unavailable (optional dataset).\n"
            "  Manual: download from http://www.starmapresources.com/data\n"
            "  → visual_1020 folder → cell_barcode_count.csv + centroids.tsv\n"
            f"  → place in {star_dir}\n"
        )

    print(f"  STARmap data in: {star_dir}")


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
