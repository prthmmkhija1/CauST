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


def _gsm_ftp_files(gsm_acc: str) -> list:
    """List supplementary files in the NCBI FTP directory for a GSM sample."""
    import ftplib
    digits       = gsm_acc[3:]                           # e.g. "4284316"
    gsm_prefix   = gsm_acc[:3] + digits[:-3] + "nnn"    # "GSM4284nnn"
    ftp_dir      = f"/geo/samples/{gsm_prefix}/{gsm_acc}/suppl/"
    try:
        ftp = ftplib.FTP("ftp.ncbi.nlm.nih.gov", timeout=30)
        ftp.login()
        names = ftp.nlst(ftp_dir)
        ftp.quit()
        return names
    except Exception as exc:
        print(f"  [FTP] Could not list {gsm_acc}/suppl/: {exc}")
        return []


def _geo_gsm_for_slices(series_acc: str, slice_ids: list) -> dict:
    """
    Query the GEO soft text endpoint to map slice IDs → GSM accessions.
    Returns {slice_id: gsm_accession} for each slice found.
    """
    import urllib.request as req
    url = (
        f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
        f"?acc={series_acc}&targ=gsm&form=text&view=brief"
    )
    try:
        with req.urlopen(url, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        print(f"  [GEO] Could not query sample list: {exc}")
        return {}

    result      = {}
    current_gsm = None
    for line in text.splitlines():
        if line.startswith("^SAMPLE = "):
            current_gsm = line.split("= ", 1)[1].strip()
        elif line.startswith("!Sample_title = ") and current_gsm:
            title = line.split("= ", 1)[1].strip()
            for sid in slice_ids:
                if sid in title and sid not in result:
                    result[sid] = current_gsm
                    break
    return result


def _download_gsm_sample(gsm_acc: str, sid: str, dest_dir: Path) -> Path | None:
    """
    Download filtered_feature_bc_matrix.h5[.gz] (and optionally spatial.tar.gz)
    for one GSM sample, decompress if needed, and return the local .h5 path.
    """
    import gzip
    import shutil

    files = _gsm_ftp_files(gsm_acc)
    if not files:
        return None

    bases = {os.path.basename(f): f for f in files}

    # ── Download h5 matrix ────────────────────────────────────────────────
    h5_candidates = [n for n in bases if n.endswith(".h5") or n.endswith(".h5.gz")]
    if not h5_candidates:
        print(f"  [WARN] No .h5 file in {gsm_acc}/suppl/: {list(bases.keys())}")
        return None

    h5_remote = bases[h5_candidates[0]]
    h5_local  = dest_dir / h5_candidates[0]
    if not h5_local.exists():
        url = f"https://ftp.ncbi.nlm.nih.gov{h5_remote}"
        if not download_file(url, h5_local):
            return None

    if str(h5_local).endswith(".gz"):
        decompressed = h5_local.with_suffix("")
        if not decompressed.exists():
            with gzip.open(h5_local, "rb") as fin, open(decompressed, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        h5_local = decompressed

    # ── Download spatial archive (optional) ───────────────────────────────
    spatial_candidates = [n for n in bases if "spatial" in n.lower() and n.endswith(".tar.gz")]
    if spatial_candidates:
        sp_remote = bases[spatial_candidates[0]]
        sp_local  = dest_dir / spatial_candidates[0]
        if not sp_local.exists():
            url = f"https://ftp.ncbi.nlm.nih.gov{sp_remote}"
            download_file(url, sp_local)   # best-effort; non-fatal
        if sp_local.exists():
            with tarfile.open(sp_local, "r:gz") as tar:
                tar.extractall(dest_dir)

    return h5_local


def _convert_sample_to_h5ad(h5_path: Path, sample_dir: Path, sid: str, out_path: Path):
    """Convert a single 10x h5 + optional spatial folder → h5ad."""
    spatial_dir = sample_dir / "spatial"
    if spatial_dir.exists():
        adata = sc.read_visium(
            path=sample_dir,
            count_file=h5_path.name,
            load_images=False,
        )
    else:
        adata = sc.read_10x_h5(h5_path)
        adata.var_names_make_unique()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"  {sid}: {adata.n_obs} spots × {adata.n_vars} genes → saved")
    return adata.n_obs


def download_dlpfc():
    """
    Download the full DLPFC 10x Visium dataset from NCBI GEO (GSE144136).

    The LIBD AWS S3 mirror only contains 224-spot manually-annotated subsets.
    Full tissue sections (~3 400 spots each) are in individual GSM sample
    pages of GSE144136 (Maynard et al. 2021, Nature Neuroscience).

    Strategy:
      1. Query GEO soft text to map slice IDs → GSM accessions.
      2. For each GSM, download filtered_feature_bc_matrix.h5[.gz] from
         its FTP supplementary directory.
      3. Convert to h5ad.
    """
    print("\n--- DLPFC (Human Brain) ---")
    dlpfc_dir = DATA_RAW / "DLPFC"
    dlpfc_dir.mkdir(parents=True, exist_ok=True)
    proc_dir = ROOT / "data" / "processed" / "DLPFC"
    proc_dir.mkdir(parents=True, exist_ok=True)

    slice_ids = [
        "151507", "151508", "151509", "151510",
        "151669", "151670", "151671", "151672",
        "151673", "151674", "151675", "151676",
    ]

    # ── Remove stale 224-spot raw files ──────────────────────────────────────
    for sid in slice_ids:
        raw_path = dlpfc_dir / f"{sid}.h5ad"
        if raw_path.exists() and not _dlpfc_file_is_valid(raw_path):
            print(f"  [stale] Removing {sid}.h5ad – only {_dlpfc_obs(raw_path)} spots")
            raw_path.unlink()

    # ── Short-circuit if all 12 valid slices already present ─────────────────
    valid_raw = sum(1 for s in slice_ids if _dlpfc_file_is_valid(dlpfc_dir / f"{s}.h5ad"))
    if valid_raw == 12:
        print(f"  [skip] All 12 DLPFC slices already valid in {dlpfc_dir}")
        return
    print(f"  {valid_raw}/12 valid slices found; fetching from NCBI GEO …")

    # ── Find which GSM accession corresponds to each slice ───────────────────
    print("  Querying GEO sample list for GSE144136 …")
    gsm_map = _geo_gsm_for_slices("GSE144136", slice_ids)
    if not gsm_map:
        print("  [WARN] Could not retrieve sample list from GEO.")
        _print_dlpfc_manual_instructions(dlpfc_dir)
        return
    print(f"  Found {len(gsm_map)}/12 sample mappings: {gsm_map}")

    # ── Download + convert each sample ───────────────────────────────────────
    converted = 0
    for sid in slice_ids:
        out_path = dlpfc_dir / f"{sid}.h5ad"
        if _dlpfc_file_is_valid(out_path):
            print(f"  [skip] {sid}.h5ad already valid")
            converted += 1
            continue

        if sid not in gsm_map:
            print(f"  [WARN] No GSM found for slice {sid} – skipping")
            continue

        gsm = gsm_map[sid]
        sample_dl_dir = DATA_RAW / f"DLPFC_raw" / sid
        sample_dl_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading {sid} ({gsm}) …")
        h5_path = _download_gsm_sample(gsm, sid, sample_dl_dir)
        if h5_path is None:
            print(f"  [WARN] Download failed for {sid}")
            continue

        if not _HAS_SC:
            print("  [SKIP] scanpy not installed – cannot convert.")
            continue

        try:
            _convert_sample_to_h5ad(h5_path, sample_dl_dir, sid, out_path)
            converted += 1
        except Exception as exc:
            print(f"  [WARN] Conversion failed for {sid}: {exc}")

    if converted > 0:
        print(f"  DLPFC: {converted}/12 slices ready in {dlpfc_dir}")
        print("  Run  python scripts/02_preprocess.py  next.")
    else:
        _print_dlpfc_manual_instructions(dlpfc_dir)


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
        "  The full Visium data (~3 400 spots/slice) is in individual SAMPLE\n"
        "  pages of GEO series GSE144136 (Maynard et al. 2021).\n"
        "  NOTE: the series-level file 'GSE144136_GRCh38-1.2.0_premrna.tar.gz'\n"
        "  is a Cell Ranger genome reference — NOT count data.\n"
        "\n"
        "  Manual approach:\n"
        "  1. Go to https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144136\n"
        "  2. Click a sample link (e.g. GSM4284316 for slice 151507)\n"
        "  3. Scroll down to 'Supplementary file' and download:\n"
        "       *_filtered_feature_bc_matrix.h5.gz\n"
        "       *_spatial.tar.gz   (for spatial coordinates)\n"
        "  4. Repeat for all 12 slices (151507-151510, 151669-151676)\n"
        "  5. Place each h5[.gz] in  data/raw/DLPFC_raw/{slice_id}/\n"
        "     and the extracted spatial/ folder alongside it\n"
        "  6. Re-run: python scripts/01_download_data.py\n"
        "\n"
        "  Each converted file must have ≥1 000 spots.\n"
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
