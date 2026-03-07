"""
scripts/02_preprocess.py
=========================
ONE-SHOT script: preprocess all raw .h5ad files.

Run once. All preprocessed files saved to  data/processed/.

    python scripts/02_preprocess.py

What it does
------------
  For every raw .h5ad in data/raw/:
    1.  Filter low-quality spots (< 200 genes expressed)
    2.  Filter rarely-seen genes (expressed in < 3 spots)
    3.  Normalize to 10,000 counts per spot
    4.  Log1p transform
    5.  Select top 3,000 highly-variable genes
    6.  Scale to zero-mean / unit-variance
    7.  Save processed file

Edit the CONFIG section below to adjust parameters.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── CONFIG ────────────────────────────────────────────────────────────────
N_TOP_GENES  = 3000     # number of highly-variable genes to keep
MIN_GENES    = 200      # spots with fewer expressed genes are dropped
MIN_CELLS    = 3        # genes expressed in fewer spots are dropped
# ─────────────────────────────────────────────────────────────────────────


from caust.data.loader import load_and_preprocess, save_processed


def process_directory(raw_dir: Path, out_dir: Path) -> int:
    """Process all .h5ad files in raw_dir and save to out_dir."""
    h5ad_files = list(raw_dir.glob("**/*.h5ad")) + list(raw_dir.glob("**/*.h5"))
    if not h5ad_files:
        print(f"  [skip] No .h5ad or .h5 files found in {raw_dir}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    success = 0

    for f in sorted(h5ad_files):
        out_path = out_dir / f.relative_to(raw_dir).with_suffix(".h5ad")

        if out_path.exists():
            print(f"  [skip] Already processed: {out_path.name}")
            success += 1
            continue

        print(f"\nProcessing: {f.relative_to(ROOT)}")
        try:
            adata = load_and_preprocess(
                path        = f,
                n_top_genes = N_TOP_GENES,
                min_genes   = MIN_GENES,
                min_cells   = MIN_CELLS,
                normalize   = True,
                scale       = True,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_processed(adata, out_path)
            success += 1
        except Exception as exc:
            print(f"  [ERROR] {f.name}: {exc}")

    return success


if __name__ == "__main__":
    import scanpy as sc
    sc.settings.verbosity = 1   # reduce scanpy's own printing

    print("=" * 55)
    print("  CauST – Preprocessing Script")
    print("=" * 55)

    RAW = ROOT / "data" / "raw"
    PROC = ROOT / "data" / "processed"

    for dataset in ["DLPFC", "MouseBrain", "MouseOB", "HumanBreastCancer", "STARmap"]:
        raw_dir = RAW / dataset
        out_dir = PROC / dataset
        print(f"\n{'─'*40}\nDataset: {dataset}")

        raw_files = list(raw_dir.glob("**/*.h5ad")) + list(raw_dir.glob("**/*.h5")) \
            if raw_dir.exists() else []
        proc_files = list(out_dir.glob("**/*.h5ad")) if out_dir.exists() else []

        # Detect unprocessed files sitting in data/processed/ (downloaded raw)
        needs_reprocess = False
        if proc_files and not raw_files:
            # Check if these "processed" files are actually raw (>5000 genes = not HVG-filtered)
            sample = proc_files[0]
            try:
                adata_check = sc.read_h5ad(sample, backed="r")
                if adata_check.n_vars > 5000:
                    print(f"  [WARN] Files in {out_dir} appear unprocessed ({adata_check.n_vars} genes).")
                    print(f"         Moving to {raw_dir} and re-processing …")
                    needs_reprocess = True
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    import shutil
                    for pf in proc_files:
                        dest = raw_dir / pf.name
                        if not dest.exists():
                            shutil.move(str(pf), str(dest))
                    raw_files = list(raw_dir.glob("**/*.h5ad")) + list(raw_dir.glob("**/*.h5"))
                    proc_files = []
                    adata_check.file.close()
                else:
                    adata_check.file.close()
                    print(f"  [ok] {len(proc_files)} file(s) already in {out_dir} — skipping re-processing.")
                    continue
            except Exception:
                print(f"  [ok] {len(proc_files)} file(s) already in {out_dir} — skipping re-processing.")
                continue

        if not raw_files and proc_files and not needs_reprocess:
            print(f"  [ok] {len(proc_files)} file(s) already in {out_dir} — skipping re-processing.")
            continue

        if not raw_files:
            print(f"  [skip] No files found in {raw_dir}. Run 01_download_data.py first.")
            continue

        n = process_directory(raw_dir, out_dir)
        print(f"  {n} file(s) processed → {out_dir}")

    print("\n✓ Preprocessing complete. Files in data/processed/")
