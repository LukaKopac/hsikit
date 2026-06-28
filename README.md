# hsikit

This repository contains a collection of Python functions and classes developed for working with **hyperspectral data**.

The code reflects an **internal research toolbox**, built iteratively during multiple experiments.

⚠️ This package is under active development and its API is not stable  
    (versioning is applied only to larger structural changes).

---

## Package structure

Main functionality lives in `hsikit/`, with modules split by responsibility.

```text
project-root/
├── hsikit/
│   ├── __init__.py
│   ├── hsi_io.py              # ⭐ Load data
│   ├── base_utils.py          # ⭐ Data transformation, processing, utility
│   ├── visualizations.py      # ⭐ HSI visualization
│   ├── cleaning.py            # ⭐ Clean data, remove dead pixels/extreme outliers
│   ├── masking_utility.py     # Masking related helper functions
│   ├── binary_masks.py        # Binary mask functions for background removal
│   ├── temp_bg_classes.py     # ⭐ Main bg removal and sample extraction pipeline
│   ├── extractors.py          # ⭐ Extract spectra from HSI cubes
│   ├── feature_extraction.py  # ⭐ Feature selection and extraction
│   ├── preprocessing.py       # ⭐ Data normalization and preprocessing
│   └── sandbox.py             # Unorganized, non-reviewed colection of utility from various experiments
│
├── CHANGELOG.md
├── LICENSE
├── README.md
├── TODO.md
└── pyproject.toml
```

---

## Scope and intent

This repository is intended to:

- Provide **practical utilities** for handling hyperspectral data
- Capture **real-world preprocessing and analysis patterns** used during research
- Serve as a **transparent methodological reference**

---

## What this code supports

The utilities in this repository include functionality such as:

- Loading and handling specific data formats (combination of .raw and .hdr files)
- Data visualization (hypercube, spectra, histograms)
- Basic preprocessing and normalization routines
- Masking and background removal
- Other utilities for working with hyperspectral data

---

## What this code does *not* guarantee

- A stable or well-defined public API
- Optimized performance
- Backward compatibility between versions

Disclaimer:  
AI tools were used to assist in the development of portions of this codebase.  
All generated content has been reviewed, tested, and validated to ensure correctness and functionality.  
The code is provided **as-is**, primarily for research use and reference.
