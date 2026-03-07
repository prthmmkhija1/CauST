#!/usr/bin/env python3
"""
Standalone DLPFC Visium download script.

Downloads Visium sections from the DLPFC dataset
(Maynard et al. 2021, Nature Neuroscience) from NCBI GEO (GSE144239).

GEO provides 4 Visium sections (P4_rep1, P4_rep2, P6_rep1, P6_rep2) as
spaceranger output inside GSE144239_RAW.tar.  The full 12-section dataset
with IDs 151507–151676 used by spatialLIBD is NOT deposited on GEO.

    python scripts/get_dlpfc.py

Output: data/raw/DLPFC/{section_name}.h5ad  (~2000–4000 spots each)
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
ROOT    = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "raw" / "DLPFC"
DL_DIR  = ROOT / "data" / "raw" / "DLPFC_download"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DL_DIR.mkdir(parents=True, exist_ok=True)

FTP_HOST     = "ftp.ncbi.nlm.nih.gov"
ACCESSION    = "GSE144239"
SERIES_DIR   = f"/geo/series/GSE144nnn/{ACCESSION}/suppl/"
BASE_FTP_URL = f"https://{FTP_HOST}"

# The 4 Visium GSM accessions and their labels
VISIUM_SAMPLES = {
    "GSM4565823": "P4_rep1",
    "GSM4565824": "P4_rep2",
    "GSM4565825": "P6_rep1",
    "GSM4565826": "P6_rep2",
}


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


def is_valid_h5ad(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import anndata as ad
        tmp = ad.read_h5ad(path, backed="r")
        n   = tmp.n_obs
        tmp.file.close()
        return n >= 500
    except Exception:
        return False


def decompress_gz_file(src: Path, dest: Path):
    """Decompress a .gz file to dest."""
    if dest.exists():
        return
    with gzip.open(src, "rb") as fin, open(dest, "wb") as fout:
        shutil.copyfileobj(fin, fout)


# ── Strategy 1: assemble Visium sections from RAW.tar ────────────────────────

def strategy_raw_tar() -> int:
    """
    Download GSE144239_RAW.tar, extract, and assemble the 4 Visium sections
    from their spaceranger output files (matrix.mtx.gz, barcodes.tsv.gz,
    features.tsv.gz, spatial metadata).
    """
    import scanpy as sc

    raw_tar_name = "GSE144239_RAW.tar"
    raw_url      = f"{BASE_FTP_URL}{SERIES_DIR}{raw_tar_name}"
    raw_local    = DL_DIR / raw_tar_name

    if not download(raw_url, raw_local):
        return 0

    # ── Extract ───────────────────────────────────────────────────────────
    extract_dir = DL_DIR / "RAW_ext"
    extract_dir.mkdir(exist_ok=True)
    if not any(extract_dir.iterdir()):
        print(f"  Extracting {raw_tar_name} …")
        with tarfile.open(raw_local) as tar:
            tar.extractall(extract_dir)
        print("  Extraction done.")

    # ── Inventory extracted files ─────────────────────────────────────────
    all_files = {f.name: f for f in extract_dir.iterdir() if f.is_file()}
    print(f"  {len(all_files)} files in archive")

    converted = 0
    for gsm, label in VISIUM_SAMPLES.items():
        out_path = OUT_DIR / f"{label}.h5ad"
        if is_valid_h5ad(out_path):
            print(f"  [skip] {label}.h5ad already valid")
            converted += 1
            continue

        # Find the files for this GSM
        gsm_files = {k: v for k, v in all_files.items() if k.startswith(gsm)}
        if not gsm_files:
            print(f"  [WARN] No files for {gsm} ({label})")
            continue

        print(f"  Assembling {label} ({gsm}) …")

        # ── Build a 10x-format directory ──────────────────────────────────
        sample_dir  = DL_DIR / "assembled" / label
        spatial_dir = sample_dir / "spatial"
        sample_dir.mkdir(parents=True, exist_ok=True)
        spatial_dir.mkdir(exist_ok=True)

        # Matrix files: matrix.mtx, barcodes.tsv, features.tsv (or genes.tsv)
        mtx_gz = next((v for k, v in gsm_files.items() if "matrix.mtx" in k), None)
        bc_gz  = next((v for k, v in gsm_files.items() if "barcodes.tsv" in k), None)
        ft_gz  = next((v for k, v in gsm_files.items()
                       if "features.tsv" in k or "genes.tsv" in k), None)

        if not mtx_gz or not bc_gz or not ft_gz:
            print(f"  [WARN] Missing matrix/barcodes/features for {label}")
            print(f"    Found: {list(gsm_files.keys())}")
            continue

        # Decompress matrix files (keep .gz for scanpy — it reads .gz natively)
        # But these are DOUBLE compressed: the tar has .gz files that are themselves
        # gzipped content. Just copy them to the expected names.
        shutil.copy2(mtx_gz, sample_dir / "matrix.mtx.gz")
        shutil.copy2(bc_gz,  sample_dir / "barcodes.tsv.gz")
        shutil.copy2(ft_gz,  sample_dir / "features.tsv.gz")

        # Spatial files
        for key, src in gsm_files.items():
            if "tissue_positions" in key:
                decompress_gz_file(src, spatial_dir / "tissue_positions_list.csv")
            elif "scalefactors" in key:
                decompress_gz_file(src, spatial_dir / "scalefactors_json.json")
            elif "hires_image" in key:
                decompress_gz_file(src, spatial_dir / "tissue_hires_image.png")
            elif "lowres_image" in key:
                decompress_gz_file(src, spatial_dir / "tissue_lowres_image.png")

        # ── Read with scanpy (MTX format — no .h5 files in GEO deposit) ──
        try:
            import pandas as pd
            adata = sc.read_10x_mtx(sample_dir)
            adata.var_names_make_unique()

            # Attach spatial coordinates from tissue_positions_list.csv
            pos_file = spatial_dir / "tissue_positions_list.csv"
            if pos_file.exists():
                try:
                    # Newer spaceranger: has header row
                    pos = pd.read_csv(pos_file, header=0)
                    if pos.shape[1] == 6:
                        pos.columns = ["barcode", "in_tissue", "array_row",
                                       "array_col", "pxl_row", "pxl_col"]
                    else:
                        raise ValueError("unexpected columns")
                except Exception:
                    # Older spaceranger: no header
                    pos = pd.read_csv(
                        pos_file, header=None,
                        names=["barcode", "in_tissue", "array_row",
                               "array_col", "pxl_row", "pxl_col"],
                    )
                pos = pos.set_index("barcode")

                # Filter to spots present in adata (already tissue-only from barcodes.tsv)
                common = adata.obs_names[adata.obs_names.isin(pos.index)]
                if len(common) > 0:
                    pos = pos.reindex(adata.obs_names)
                    adata.obsm["spatial"] = (
                        pos[["pxl_col", "pxl_row"]].fillna(0).values.astype(float)
                    )
                    adata.obs["array_row"] = pos["array_row"].values
                    adata.obs["array_col"] = pos["array_col"].values
                    note = "spatial attached"
                else:
                    note = "no spatial overlap"
            else:
                note = "no spatial file"

            adata.write_h5ad(out_path)
            print(f"  {label}: {adata.n_obs} spots × {adata.n_vars} genes → saved ({note})")
            converted += 1
        except Exception as exc:
            print(f"  [WARN] Failed for {label}: {exc}")

    return converted


# ── Strategy 2: parse ST_Visium_counts.txt.gz ─────────────────────────────────

def strategy_visium_txt(files: dict[str, str]) -> int:
    """
    Parse the combined Visium count matrix (genes × spots).
    Column names are like "P4_AAACAAGTATCTCCCA-1_1" (patient_barcode_rep).
    Split by the last _N to get 4 sections: P4_1, P4_2, P6_1, P6_2.
    """
    import numpy as np
    import pandas as pd
    import anndata as ad
    from scipy.sparse import csr_matrix

    counts_name = next((n for n in files if "Visium_counts" in n), None)
    if not counts_name:
        return 0

    print("  Strategy: Visium txt.gz counts file …")
    counts_local = DL_DIR / counts_name
    if not download(f"{BASE_FTP_URL}{files[counts_name]}", counts_local):
        return 0

    print("  Loading Visium counts (may take a moment) …")
    counts = pd.read_csv(counts_local, sep="\t", compression="gzip", index_col=0)
    gene_names = counts.index.astype(str).values

    # Clean quoted column names: '"P4_AAAC..._1"' → 'P4_AAAC..._1'
    cols = [c.strip('"').strip("'") for c in counts.columns]

    # Extract section label: last part after splitting by _
    # "P4_AAACAAGTATCTCCCA-1_1" → section = "P4_rep1" (last char is 1 or 2)
    section_labels = []
    for c in cols:
        parts = c.split("_")
        if len(parts) >= 3:
            patient = parts[0]        # P4 or P6
            rep_num = parts[-1]       # 1 or 2
            section_labels.append(f"{patient}_rep{rep_num}")
        else:
            section_labels.append("unknown")
    section_labels = np.array(section_labels)

    unique_sections = sorted(set(section_labels) - {"unknown"})
    print(f"  Sections found: {unique_sections}")
    print(f"  Matrix: {counts.shape[1]} spots × {counts.shape[0]} genes")

    if not unique_sections:
        return 0

    mat = csr_matrix(counts.values.T)  # genes×spots → spots×genes

    converted = 0
    for sec in unique_sections:
        out = OUT_DIR / f"{sec}.h5ad"
        if is_valid_h5ad(out):
            print(f"  [skip] {sec}.h5ad"); converted += 1; continue

        mask = section_labels == sec
        if not mask.any():
            continue

        # Extract clean barcodes (strip patient prefix and rep suffix)
        barcodes = []
        for c in np.array(cols)[mask]:
            parts = c.split("_")
            bc = "_".join(parts[1:-1]) if len(parts) >= 3 else c
            barcodes.append(bc)

        adata = ad.AnnData(
            X=mat[mask],
            obs=pd.DataFrame(index=barcodes),
            var=pd.DataFrame(index=gene_names),
        )
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        adata.write_h5ad(out)
        print(f"  {sec}: {int(mask.sum())} spots – saved")
        converted += 1

    return converted


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    expected_labels = list(VISIUM_SAMPLES.values())
    already = sum(1 for s in expected_labels if is_valid_h5ad(OUT_DIR / f"{s}.h5ad"))
    if already == 4:
        print("All 4 DLPFC Visium sections already present. Nothing to do.")
        print(f"  Location: {OUT_DIR}")
        return

    print(f"DLPFC download  ({already}/4 sections already valid)")
    print(f"GEO accession : {ACCESSION}  (Maynard et al. 2021, Nat Neuroscience)")
    print(f"Output        : {OUT_DIR}")
    print()
    print("  NOTE: GEO provides 4 of the 12 Visium sections (P4, P6 × 2 reps).")
    print("  The full 12-section dataset (IDs 151507–151676) is hosted by LIBD's")
    print("  spatialLIBD R package, not on GEO. These 4 sections are sufficient")
    print("  for benchmarking.")
    print()

    # ── List FTP directory ────────────────────────────────────────────────
    print(f"Listing FTP: {SERIES_DIR}")
    paths = ftp_list(SERIES_DIR)
    if not paths:
        print(f"\n[ERROR] Could not list FTP directory. Check network.\n")
        sys.exit(1)

    files     = {os.path.basename(p): p for p in paths}
    basenames = list(files.keys())
    print(f"Found {len(basenames)} files:")
    for n in basenames:
        print(f"  {n}")
    print()

    # ── Strategy 1: assemble from RAW.tar (best — includes spatial data) ─
    converted = strategy_raw_tar()

    if converted == 0:
        # ── Strategy 2: parse txt.gz (no spatial, but still usable) ──────
        converted = strategy_visium_txt(files)

    print()
    if converted > 0:
        print(f"Done: {converted}/4 DLPFC Visium sections saved to {OUT_DIR}")
        print("Next: python scripts/02_preprocess.py")
    else:
        print("[FAILED] Could not process any sections.")
        print(f"GEO page: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={ACCESSION}")
        sys.exit(1)


if __name__ == "__main__":
    main()
