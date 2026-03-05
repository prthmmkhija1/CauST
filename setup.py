from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name             = "caust",
    version          = "0.1.0",
    author           = "Pratham Makhija",
    author_email     = "pratham@example.com",   # update with real email
    description      = (
        "CauST: Causal Gene Discovery for Spatial Transcriptomics "
        "via Graph Attention Autoencoders and Perturbation-Based Scoring"
    ),
    long_description      = long_description,
    long_description_content_type = "text/markdown",
    url              = "https://github.com/prthmmkhija1/CauST",
    packages         = find_packages(exclude=["tests*", "scripts*", "experiments*"]),
    python_requires  = ">=3.9",
    install_requires = [
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "scanpy>=1.9.6",
        "squidpy>=1.2.3",
        "anndata>=0.9.2",
        "numpy>=1.24.0",
        "pandas>=1.5.3",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.2",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.2",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "h5py>=3.8.0",
    ],
    extras_require = {
        "dev": ["pytest>=7.3.0", "statsmodels>=0.14.0"],
    },
    classifiers = [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords = (
        "spatial transcriptomics causal inference graph neural networks "
        "single-cell bioinformatics do-calculus STAGATE GraphST"
    ),
)
