"""
scripts/download_spatiallbd.py
================================
Download spatialLIBD 12-slice DLPFC data with layer_guess annotations.

These 12 Visium slices (151507-151676) from 3 donors are the gold-standard
DLPFC benchmark for spatial domain identification. Each spot has a
'layer_guess' label (WM, Layer1-Layer6) enabling ARI/NMI evaluation.

Usage
-----
    python scripts/download_spatiallbd.py

Output
------
    data/raw/DLPFC/151507.h5ad  ... 151676.h5ad   (12 files)

Then preprocess with:
    python scripts/02_preprocess.py
"""

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "DLPFC"
RAW_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]

# Primary source: spatialLIBD S3 bucket (Lieber Institute)
S3_BASE = "https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5ad"

# Fallback source: GitHub release assets
GITHUB_BASE = (
    "https://github.com/LieberInstitute/spatialLIBD"
    "/releases/download/Bioc3.14"
)


def download_with_progress(url: str, dest: Path) -> bool:
    """Download a file showing a simple progress indicator."""

    def _hook(count, block_size, total_size):
        if total_size > 0:
            pct = min(100, int(count * block_size * 100 / total_size))
            mb = total_size / 1e6
            print(f"\r  {pct:3d}%  ({mb:.0f} MB)", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_hook)
        print()  # newline after progress bar
        return True
    except Exception as exc:
        print(f"\n  [WARN] {exc}")
        if dest.exists():
            dest.unlink()
        return False


def main() -> None:
    print("=" * 55)
    print("  spatialLIBD 12-slice DLPFC downloader")
    print("=" * 55)

    already_have = [s for s in SAMPLES if (RAW_DIR / f"{s}.h5ad").exists()]
    if already_have:
        print(f"\n[skip] Already downloaded: {already_have}\n")

    to_download = [s for s in SAMPLES if s not in already_have]
    if not to_download:
        print("All 12 slices already present — nothing to do.")
        print(f"Location: {RAW_DIR}")
        return

    failed = []

    for sid in to_download:
        dest = RAW_DIR / f"{sid}.h5ad"
        print(f"\nDownloading {sid} → {dest.name}")

        # Try primary URL
        url = f"{S3_BASE}/{sid}_filtered_feature_bc_matrix.h5ad"
        print(f"  URL: {url}")
        ok = download_with_progress(url, dest)

        # Try simple filename fallback
        if not ok:
            url2 = f"{S3_BASE}/{sid}.h5ad"
            print(f"  Retrying: {url2}")
            ok = download_with_progress(url2, dest)

        if ok:
            size_mb = dest.stat().st_size / 1e6
            print(f"  Saved {size_mb:.1f} MB")
        else:
            failed.append(sid)
            print(f"  [FAIL] Could not download {sid}")

    print("\n" + "=" * 55)
    if failed:
        print(f"Failed: {failed}")
        print(
            "\nManual download instructions:\n"
            "  1. Visit https://github.com/LieberInstitute/spatialLIBD\n"
            "  2. Or install the R package:\n"
            "       BiocManager::install('spatialLIBD')\n"
            "       library(spatialLIBD)\n"
            "       sce <- fetch_data(type='sce')\n"
            "  3. Or use the Bioconductor ExperimentHub:\n"
            "       eh <- ExperimentHub()\n"
            "       myfiles <- eh[['EH_id']]"
        )
        sys.exit(1)
    else:
        downloaded = len(SAMPLES) - len(already_have)
        print(f"Done! Downloaded {downloaded} slice(s).")
        print(f"Location: {RAW_DIR}")
        print("\nNext step:")
        print("  python scripts/02_preprocess.py")


if __name__ == "__main__":
    main()
