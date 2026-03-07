#!/usr/bin/env python3
"""
Standalone DLPFC Visium download script.

Downloads all 12 Visium sections for the human DLPFC dataset
(Maynard et al. 2021, Nature Neuroscience) from NCBI GEO.

    python scripts/get_dlpfc.py

Output: data/raw/DLPFC/{151507..151676}.h5ad  (~3 400 spots each)
Then run: python scripts/02_preprocess.py
"""

import ftplib
import gzip
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parents[1]
OUT_DIR  = ROOT / "data" / "raw" / "DLPFC"
DL_DIR   = ROOT / "data" / "raw" / "DLPFC_download"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DL_DIR.mkdir(parents=True, exist_ok=True)

SLICE_IDS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

FTP_HOST = "ftp.ncbi.nlm.nih.gov"
# Maynard et al. 2021 Visium DLPFC – correct accession
ACCESSION    = "GSE144239"
SERIES_DIR   = f"/geo/series/GSE144nnn/{ACCESSION}/suppl/"
BASE_FTP_URL = f"https://{FTP_HOST}"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _progress(blocknum, blocksize, total):
    done = blocknum * blocksize
    if total > 0:
        pct = min(done / total * 100, 100)
        bar = int(pct / 2)
        sys.stdout.write(f"\r  [{'#'*bar}{' '*(50-bar)}] {pct:.1f}%  ")
        sys.stdout.flush()
        if pct >= 100:
            print()


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [skip] {dest.name}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {dest.name} …")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
        return True
    except Exception as exc:
        print(f"  [FAILED] {exc}")
        if dest.exists():
            dest.unlink()
        return False


def ftp_list(ftp_dir: str) -> list[str]:
    try:
        ftp = ftplib.FTP(FTP_HOST, timeout=30)
        ftp.login()
        names = ftp.nlst(ftp_dir)
        ftp.quit()
        return names
    except Exception as exc:
        print(f"  [FTP] {exc}")
        return []


def decompress_gz(src: Path) -> Path:
    dest = src.with_suffix("")
    if not dest.exists():
        print(f"  Decompressing {src.name} …")
        with gzip.open(src, "rb") as fin, open(dest, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return dest


def is_valid_h5ad(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import anndata as ad
        tmp = ad.read_h5ad(path, backed="r")
        n   = tmp.n_obs
        tmp.file.close()
        return n >= 1000
    except Exception:
        return False


# ── Per-sample (GSM) strategy ─────────────────────────────────────────────────

def _list_gsm_suppl(gsm: str) -> dict[str, str]:
    """Return {filename: full_ftp_path} for a GSM's supplementary directory."""
    digits = gsm[3:]
    prefix = gsm[:3] + digits[:-3] + "nnn"
    ftp_dir = f"/geo/samples/{prefix}/{gsm}/suppl/"
    paths = ftp_list(ftp_dir)
    return {os.path.basename(p): p for p in paths}


def _geo_soft_gsm_map(series: str, slice_ids: list[str]) -> dict[str, str]:
    """Query GEO SOFT to get {slice_id: GSM_accession}."""
    import urllib.request as req
    url = (
        f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
        f"?acc={series}&targ=gsm&form=text&view=brief"
    )
    try:
        with req.urlopen(url, timeout=30) as r:
            text = r.read().decode("utf-8", errors="ignore")
    except Exception as exc:
        print(f"  [GEO SOFT] {exc}")
        return {}

    result, cur_gsm = {}, None
    for line in text.splitlines():
        if line.startswith("^SAMPLE = "):
            cur_gsm = line.split("= ", 1)[1].strip()
        elif line.startswith("!Sample_title = ") and cur_gsm:
            title = line.split("= ", 1)[1].strip()
            for sid in slice_ids:
                if sid in title and sid not in result:
                    result[sid] = cur_gsm
    return result


def strategy_per_gsm() -> int:
    """Download per-sample h5 + spatial files from individual GSM pages."""
    import scanpy as sc

    print("  Strategy: per-GSM FTP download …")
    gsm_map = _geo_soft_gsm_map(ACCESSION, SLICE_IDS)
    if not gsm_map:
        print("  Could not retrieve GSM map from GEO SOFT.")
        return 0

    print(f"  Found {len(gsm_map)}/12 GSM mappings.")
    converted = 0
    for sid in SLICE_IDS:
        out = OUT_DIR / f"{sid}.h5ad"
        if is_valid_h5ad(out):
            print(f"  [skip] {sid}.h5ad already valid")
            converted += 1
            continue
        if sid not in gsm_map:
            print(f"  [WARN] No GSM for {sid}")
            continue

        gsm     = gsm_map[sid]
        sdir    = DL_DIR / sid
        sdir.mkdir(exist_ok=True)
        files   = _list_gsm_suppl(gsm)

        h5_names = [n for n in files if n.endswith(".h5") or n.endswith(".h5.gz")]
        if not h5_names:
            print(f"  [WARN] No .h5 in {gsm}: {list(files.keys())}")
            continue

        h5_remote = files[h5_names[0]]
        h5_local  = sdir / h5_names[0]
        if not download(f"{BASE_FTP_URL}{h5_remote}", h5_local):
            continue
        if str(h5_local).endswith(".gz"):
            h5_local = decompress_gz(h5_local)

        # spatial (optional)
        sp_names = [n for n in files if "spatial" in n.lower() and n.endswith(".tar.gz")]
        if sp_names:
            sp_remote = files[sp_names[0]]
            sp_local  = sdir / sp_names[0]
            if download(f"{BASE_FTP_URL}{sp_remote}", sp_local):
                with tarfile.open(sp_local, "r:gz") as tar:
                    tar.extractall(sdir)

        try:
            spatial_dir = sdir / "spatial"
            if spatial_dir.exists():
                adata = sc.read_visium(path=sdir, count_file=h5_local.name, load_images=False)
            else:
                adata = sc.read_10x_h5(h5_local)
                adata.var_names_make_unique()
            adata.write_h5ad(out)
            print(f"  {sid}: {adata.n_obs} spots – saved")
            converted += 1
        except Exception as exc:
            print(f"  [WARN] Conversion failed for {sid}: {exc}")

    return converted


# ── RAW tar strategy ──────────────────────────────────────────────────────────

def strategy_raw_tar(tar_names: list[str]) -> int:
    """Download the series RAW.tar, extract, convert per-slice h5 to h5ad."""
    import scanpy as sc

    print("  Strategy: series-level RAW tar …")
    raw_name  = tar_names[0]
    raw_url   = f"{BASE_FTP_URL}{SERIES_DIR}{raw_name}"
    raw_local = DL_DIR / raw_name

    if not download(raw_url, raw_local):
        return 0

    extract_dir = DL_DIR / raw_name.replace(".tar", "").replace(".gz", "") + "_ext"
    extract_dir.mkdir(exist_ok=True)
    if not any(extract_dir.iterdir()):
        print(f"  Extracting {raw_name} …")
        with tarfile.open(raw_local) as tar:
            tar.extractall(extract_dir)
        # expand inner per-sample tars if present
        for inner in extract_dir.rglob("*.tar.gz"):
            sd = inner.parent / inner.name.replace(".tar.gz", "")
            sd.mkdir(exist_ok=True)
            with tarfile.open(inner, "r:gz") as t:
                t.extractall(sd)
        print("  Extraction done.")

    converted = 0
    for sid in SLICE_IDS:
        out = OUT_DIR / f"{sid}.h5ad"
        if is_valid_h5ad(out):
            print(f"  [skip] {sid}.h5ad")
            converted += 1
            continue

        h5_hits = (
            list(extract_dir.rglob(f"*{sid}*filtered_feature_bc_matrix.h5"))
            + list(extract_dir.rglob(f"*{sid}*.h5"))
        )
        gz_hits = list(extract_dir.rglob(f"*{sid}*.h5.gz"))
        candidates = h5_hits or [decompress_gz(p) for p in gz_hits]

        if not candidates:
            print(f"  [WARN] No h5 found for {sid} in extracted archive")
            continue

        h5   = candidates[0]
        sdir = h5.parent
        try:
            spatial_dir = sdir / "spatial"
            if spatial_dir.exists():
                adata = sc.read_visium(path=sdir, count_file=h5.name, load_images=False)
            else:
                adata = sc.read_10x_h5(h5)
                adata.var_names_make_unique()
            adata.write_h5ad(out)
            print(f"  {sid}: {adata.n_obs} spots – saved")
            converted += 1
        except Exception as exc:
            print(f"  [WARN] Conversion failed for {sid}: {exc}")

    return converted


# ── MTX / combined matrix strategy ───────────────────────────────────────────

def strategy_combined_mtx(files: dict[str, str]) -> int:
    """
    Download combined sparse MTX + barcodes + genes, split by slice ID.
    Used when the series provides one big matrix (like GSE144136 did –
    though that turned out to be the wrong dataset entirely).
    """
    import numpy as np
    import pandas as pd
    import scipy.io as sio
    import anndata as ad

    mtx_name  = next((n for n in files if n.endswith(".mtx.gz") and "barcode" in n.lower()), None)
    cell_name = next((n for n in files if "cellname" in n.lower() or "barcode" in n.lower()
                      and n.endswith(".csv.gz")), None)
    gene_name = next((n for n in files if "gene" in n.lower() and n.endswith(".csv.gz")), None)

    if not (mtx_name and cell_name and gene_name):
        print(f"  Could not identify MTX trio in: {list(files.keys())}")
        return 0

    print("  Strategy: combined MTX matrix …")
    for fname, fpath in [(mtx_name, files[mtx_name]),
                         (cell_name, files[cell_name]),
                         (gene_name, files[gene_name])]:
        if not download(f"{BASE_FTP_URL}{fpath}", DL_DIR / fname):
            return 0

    print("  Loading MTX …")
    mat        = sio.mmread(DL_DIR / mtx_name).T.tocsr()
    cell_names = pd.read_csv(DL_DIR / cell_name, header=None, compression="gzip"
                             ).iloc[:, 0].astype(str).values
    gene_names = pd.read_csv(DL_DIR / gene_name, header=None, compression="gzip"
                             ).iloc[:, 0].astype(str).values
    print(f"  Matrix: {mat.shape[0]} cells × {mat.shape[1]} genes")

    slice_labels = np.array([""] * len(cell_names), dtype=object)
    for sid in SLICE_IDS:
        mask = np.array([sid in cn for cn in cell_names])
        slice_labels[mask] = sid

    assigned = int((slice_labels != "").sum())
    print(f"  Matched {assigned}/{len(cell_names)} cells to slice IDs")
    if assigned == 0:
        print(f"  First 5 cell names: {cell_names[:5].tolist()}")
        return 0

    converted = 0
    for sid in SLICE_IDS:
        out  = OUT_DIR / f"{sid}.h5ad"
        if is_valid_h5ad(out):
            print(f"  [skip] {sid}.h5ad"); converted += 1; continue
        mask = slice_labels == sid
        if not mask.any():
            print(f"  [WARN] No cells for {sid}"); continue

        barcodes = []
        for cn in cell_names[mask]:
            bc = cn.replace(f"{sid}_", "").replace(f"_{sid}", "")
            barcodes.append(bc if bc else cn)

        adata = ad.AnnData(
            X=mat[mask],
            obs=pd.DataFrame(index=barcodes),
            var=pd.DataFrame(index=gene_names),
        )
        adata.var_names_make_unique(); adata.obs_names_make_unique()
        adata.write_h5ad(out)
        print(f"  {sid}: {int(mask.sum())} cells – saved")
        converted += 1

    return converted


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    already = sum(1 for s in SLICE_IDS if is_valid_h5ad(OUT_DIR / f"{s}.h5ad"))
    if already == 12:
        print("All 12 DLPFC slices already present. Nothing to do.")
        print(f"  Location: {OUT_DIR}")
        return

    print(f"DLPFC download  ({already}/12 slices already valid)")
    print(f"GEO accession : {ACCESSION}  (Maynard et al. 2021, Nat Neuroscience)")
    print(f"Output        : {OUT_DIR}")
    print()

    # ── Step 1: list the FTP supplementary directory ─────────────────────────
    print(f"Listing FTP: {SERIES_DIR}")
    paths = ftp_list(SERIES_DIR)
    if not paths:
        print(
            "\n[ERROR] Could not list FTP directory.\n"
            "  Check network or try manually:\n"
            f"  https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={ACCESSION}\n"
        )
        sys.exit(1)

    files   = {os.path.basename(p): p for p in paths}
    basenames = list(files.keys())
    print(f"Found {len(basenames)} files:")
    for n in basenames:
        print(f"  {n}")
    print()

    # ── Step 2: choose strategy based on available files ─────────────────────
    raw_tars = [n for n in basenames
                if "RAW" in n.upper() and n.endswith((".tar", ".tar.gz", ".tgz"))]
    mtx_files = [n for n in basenames if n.endswith(".mtx.gz")]

    if raw_tars:
        converted = strategy_raw_tar(raw_tars)
    elif mtx_files:
        converted = strategy_combined_mtx(files)
    else:
        # Try per-GSM pages as last resort
        converted = strategy_per_gsm()

    print()
    if converted > 0:
        print(f"Done: {converted}/12 slices saved to {OUT_DIR}")
        print("Next step:  python scripts/02_preprocess.py")
    else:
        print(
            "[FAILED] Could not download any slices automatically.\n"
            "\n"
            "The dataset is Maynard et al. 2021 (Nat Neuroscience).\n"
            f"GEO page: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={ACCESSION}\n"
            "\n"
            "The supplementary files listed above show exactly what is available.\n"
            "Paste the output of this script and we can adapt the download.\n"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
