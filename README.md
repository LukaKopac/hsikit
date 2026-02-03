# hsikit

This repository contains a collection of Python functions and classes developed for **working with hyperspectral data acquired using ClydeHSI systems**.

The code reflects an **internal research toolbox**, built iteratively during multiple experiments. It is **not a general-purpose hyperspectral imaging framework**, nor is it intended to support sensors or file formats beyond ClydeHSI.

---

## Scope and intent

This repository is intended to:

- Provide **practical utilities** for handling ClydeHSI hyperspectral data
- Capture **real-world preprocessing and analysis patterns** used during research
- Serve as a **transparent reference**, not a polished or finalized pipeline

Important clarifications:

- There is **no single “official” processing workflow** in this repository
- Some functions/classes were used in different experiments; others were exploratory
- Code duplication and overlapping functionality exist as a result of iterative development
- The repository does **not contain experimental results or datasets**

---

## What this code supports

The utilities in this repository include functionality such as:

- Loading and handling ClydeHSI-specific data formats
- Basic preprocessing and normalization routines
- Masking and background removal
- Utilities for working with pixel-level hyperspectral data in Python/NumPy workflows
- Helper classes for organizing samples and experiments

All functionality assumes **ClydeHSI acquisition conventions**, sensor characteristics, and metadata structure.

---

## What this code does *not* guarantee

- Compatibility with non-ClydeHSI systems
- A stable or well-defined public API
- Optimized performance
- Comprehensive documentation or test coverage
- Backward compatibility between versions

The code is provided **as-is**, primarily for research use and reference.

---

## Relation to publications

This repository contains **supporting utilities only**.

- It does not encode a fixed experimental pipeline
- It does not define the preprocessing choices used in any specific paper
- Publication-specific processing steps should be described explicitly in the corresponding manuscript

The purpose of this repository is to improve **methodological transparency**, not to replace formal methodological descriptions.
