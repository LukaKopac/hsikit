# hsikit

This repository contains a collection of Python functions and classes developed for **working with hyperspectral data acquired using ClydeHSI systems**.

The code reflects an **internal research toolbox**, built iteratively during multiple experiments.

⚠️ This package is under active development and its API is not yet stable (versioning is applied only to larger structural changes).

---

## Package structure

Main functionality lives in `hsikit/`, with utilities split by responsibility.

```text
project-root/
├── hsikit/
│   ├── __init__.py         # Package initialization
│   ├── visualizations.py   # HSI visualization
│   ├── binary_masks.py     # Binary mask functions for background removal
│   ├── extractors.py       # Extract spectra from HSI cubes
│   ├── hsi_io.py           # Load data
│   ├── masking_utility.py  # Masking related helper functions
│   ├── preprocessing.py    # Data normalization and preprocessing
│   ├── temp_bg_classes.py  # ⭐ Main bg removal and sample extraction pipeline
│   └── sandbox.py          # Unorganized, non-reviewed colection of utility from various experiments
│
├── CHANGELOG.md            # Version history (not updated regularly)
├── LICENSE                 # License info
├── README.md               # Project documentation
├── TODO.md                 # Planned improvements / tasks
└── pyproject.toml          # Build + dependency configuration
```

---

## Scope and intent

This repository is intended to:

- Provide **practical utilities** for handling ClydeHSI hyperspectral data
- Capture **real-world preprocessing and analysis patterns** used during research
- Serve as a **transparent methodological reference**

---

## What this code supports

The utilities in this repository include functionality such as:

- Loading and handling ClydeHSI-specific data formats (combination of .raw and .hdr files)
- Data visualization (hypercube, spectra, histograms)
- Basic preprocessing and normalization routines
- Masking and background removal
- Utilities for working with hyperspectral data in Python/NumPy workflows

---

## What this code does *not* guarantee

- Compatibility with non-ClydeHSI systems
- A stable or well-defined public API
- Optimized performance
- Backward compatibility between versions

The code is provided **as-is**, primarily for research use and reference.
