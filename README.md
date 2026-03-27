# hsikit

This repository contains a collection of Python functions and classes developed for **working with hyperspectral data acquired using ClydeHSI systems**.

The code reflects an **internal research toolbox**, built iteratively during multiple experiments.

---

## Package structure
```text
project-root/
├── hsikit/
│   ├── __init__.py
│   ├── base_utils.py
│   ├── binary_masks.py
│   ├── extractors.py
│   ├── hsi_io.py
│   ├── masking_utility.py
│   ├── preprocessing.py
│   └── temp_bg_classes.py
│
├── CHANGELOG.md
├── LICENSE
├── README.md
├── TODO.md
├── pyproject.toml
└── requirements.txt
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
- Utilities for working with pixel-level hyperspectral data in Python/NumPy workflows

---

## What this code does *not* guarantee

- Compatibility with non-ClydeHSI systems
- A stable or well-defined public API
- Optimized performance
- Comprehensive documentation or test coverage
- Backward compatibility between versions

The code is provided **as-is**, primarily for research use and reference.
