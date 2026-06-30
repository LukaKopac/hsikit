# hsikit
Internal document to keep track of package structure and individual functionality of classes and functions.

## Overview

---

### [Base utilities (base_utils.py)](#base_utilspy)

**Purpose:** General utility and helper functions.

**Input → Output:** `various`

**Functions**

* [block_average_cube()](#block_average_cube)
* [dict2Xy()](#dict2xy)
* [snr_per_band()](#snr_per_band)
* [class_variance_ratio()](#class_variance_ratio)

---

### [Binary masks (binary_masks.py)](#binary_maskspy)

**Purpose:** Compute binary masks to extract samples from hyperspectral data cubes.

**Input → Output:** `cube / image → binary mask`

**Functions**

* [manual_rect_split()](#manual_rect_split)
* [mask_top_contrast()](#mask_top_contrast)
* [mask_SAM()](#mask_sam)
* [mask_highpass_otsu()](#mask_highpass_otsu)
* [mask_kmeans()](#mask_kmeans)
* [mask_from_pca()](#mask_from_pca)
* [fixed_rect_extraction()](#fixed_rect_extraction)

---

### [Cleaning (cleaning.py)](#cleaningpy)

**Purpose:** Detect, identify, and repair dead pixels and line defects in hyperspectral data cubes.

**Input → Output:** `cube → dead_pixel_mask / repaired_cube`

**Classes**

* [DeadPixelProcessor](#deadpixelprocessor)

**Functions**

* [detect_dead_and_outlier_pixels()](#detect_dead_and_outlier_pixels)
* [detect_line_defects()](#detect_line_defects)
* [identify_dead_pixels()](#identify_dead_pixels)
* [interpolate_dead_pixels()](#interpolate_dead_pixels)
* [plot_defect_summary()](#plot_defect_summary)

---

### [Extractors (extractors.py)](#extractorspy)

**Purpose:** Extract spectra or regions of interest from hyperspectral data cubes.

**Input → Output:** `cube → spectra / ROIs`

**Classes**

* [ROIExtractor](#roiextractor)

**Functions**

* [extract_local_mean()](#extract_local_mean)
* [Grid_ROI_extractor()](#grid_roi_extractor)

---

### [Feature selection (feature_selection.py)](#feature_selectionpy)

**Purpose:** Select informative spectral features and reduce spectral dimensionality.

**Input → Output:** `spectra → selected features / reduced spectra`

**Classes**

* [AdaptiveSpectralBinner](#adaptivespectralbinner)

**Functions**

* [adaptive_binning_by_gradient()](#adaptive_binning_by_gradient)
* [compute_bin_centers()](#compute_bin_centers)
* [compute_binned_stats()](#compute_binned_stats)
* [plot_spectrum_with_bins()](#plot_spectrum_with_bins)
* [rmse_cv()](#rmse_cv)
* [CARS()](#cars)

---

### [Input - Output (hsi_io.py)](#hsi_iopy)

**Purpose:** Import and export hyperspectral datasets and metadata.

**Input → Output:** `raw files → HSI cube / wavelengths / mappings`

**Functions**

* [find_hsi_basepaths()](#find_hsi_basepaths)
* [load_hsi_raw()](#load_hsi_raw)
* [load_wavelengths()](#load_wavelengths)
* [load_sample_mapping()](#load_sample_mapping)
* [load_hsi_batch()](#load_hsi_batch)
* [export_tiff_stack()](#export_tiff_stack)

---

### [Masking utility (masking_utility.py)](#masking_utilitypy)

**Purpose:** Helper utilities for working with binary masks and extracting samples.

**Input → Output:** `cube + masks → sample cubes / regions`

**Functions**

* [otsu_separation_score()](#otsu_separation_score)
* [get_valid_regions()](#get_valid_regions)
* [estimate_rect_size()](#estimate_rect_size)
* [generate_rect_mask()](#generate_rect_mask)
* [rect_mask()](#rect_mask)
* [extract_sample_cubes()](#extract_sample_cubes)
* [extract_sample_cubes_from_masks()](#extract_sample_cubes_from_masks)

---

### [Preprocessing (preprocessing.py)](#preprocessingpy)

**Purpose:** Preprocess spectra and hyperspectral data prior to analysis.

**Input → Output:** `cube / X → preprocessed cube / preprocessed X`

**Classes**

* [SNV](#snv)
* [MSC](#msc)
* [SavitzkyGolay](#savitzkygolay)

**Functions**

* [normalize_min_max()](#normalize_min_max)
* [normalize_mean_std()](#normalize_mean_std)

---

### [Sandbox (sandbox.py)](#sandboxpy)

**Purpose:** Experimental algorithms, prototypes, and functions under evaluation.

**Input → Output:** `various`

**Classes**

* [SoftPLSDA](#softplsda)
* [MNF](#mnf)

**Functions**

* [adaptive_equalize_spectrum()](#adaptive_equalize_spectrum)
* [reflectance_to_absorbance()](#reflectance_to_absorbance)
* [asls_baseline()](#asls_baseline)
* [robust_snv()](#robust_snv)
* [rnv()](#rnv)

---

### [BG Pipelines (temp_bg_classes.py)](#pipelinepy)

**Purpose:** End-to-end pipelines for importing, masking, and processing hyperspectral datasets.

**Input → Output:** `multiple cubes → masks / mappings / extracted samples`

**Classes**

* [HSIImporter](#hsiimporter)
* [HSIProcessor](#hsiprocessor)
* [HSIProcessorV2](#hsiprocessorv2)

---

### [Visualizations (visualizations.py)](#visualizationspy)

**Purpose:** Visualization utilities for hyperspectral images, spectra, and data cubes.

**Input → Output:** `cube / spectra → plots / figures`

**Functions**

* [plot_image()](#plot_image)
* [plot_spectra()](#plot_spectra)
* [plot_spectral_hist()](#plot_spectral_hist)
* [plot_3D_slices()](#plot_3d_slices)
* [plot_3D_slices_interactive()](#plot_3d_slices_interactive)
* [plot_hsi_cube()](#plot_hsi_cube)

---

# Function Inventory

**Status legend:**
- ✅ Keep
- 🔄 Refactor
- 🚚 Move
- ❓ Review
- ❌ Remove

**Template:**

## module_name.py

### name()
- **Type:** function / class
- **IO:** main_input -> main_output
- **Status:** ✅ Keep
- **Purpose:** 
- **Key methods (for classes):**
- **Dependencies:**
- **Notes:**

---

## base_utils.py

[⬅ Back to Overview](#overview)

### block_average_cube()
- **Type:** function
- **IO:** cube -> subsampled_cube
- **Status:** ✅ Keep
- **Purpose:** Subsample / block-average HSI cube by possibly cropping it to enforce 'block_size' parameter. Returns mean spectrum per block as a subsampled cube (H/block_size, W/block_size, B).
- **Notes:**
    - option that doesn't crop the original cube

### dict2Xy()
- **Type:** function
- **IO:** dict -> X, y
- **Status:** ✅ Keep
- **Purpose:** Converts a sample dictionary (either [key:np.ndarray] or [key:list[np.ndarray]]) to X matrix (n_samples, B) and y label vector (n_samples,).
- **Notes:**

### snr_per_band()
- **Type:** function
- **IO:** cube -> snr (np.ndarray)
- **Status:** ✅ Keep
- **Purpose:** Computes signal-to-noise ratio (SNR = mean / std) per band with optional valid pixel masking.
- **Notes:**

### class_variance_ratio()
- **Type:** function
- **IO:** X, y -> within_var, between_var, ratio
- **Status:** ✅ Keep
- **Purpose:** Computes between classes variance, within classes variance and their ratio to indicate discriminative power.
- **Notes:**
    - return scatter matrices

---

# 📦 binary_masks.py

[⬅ Back to Overview](#overview)

### manual_rect_split()
- **Type:** function
- **IO:** cube -> list of 2D binary masks
- **Status:** ✅ Keep
- **Purpose:** Generates a grid of 2D binary masks for rectangular samples in a HSI cube scene. Parameters need to be set manually (sample size, grid shape, start, spacing). Individual masks can be used to extract smaller cubes / spectra. Optional visualization of masking.
- **Notes:**
    - Can merge with grid-based spectra extraction (for example optional extraction).

### mask_top_contrast()
- **Type:** function
- **IO:** cube -> single 2D binary mask
- **Status:** ✅ Keep
- **Purpose:** Computes a foreground mask by *cropping the original cube*, selecting top_n bands with highest contrast (std), boosting contrast, automatic thresholding (Otsu), and combining masks by majority voting. Includes shadow correction and morphological cleaning. Optional visualization.
- **Notes:**
    - Also return individual masks (optionally)
    - Morphological cleaning (optional, improve param parsing, apply to individual masks?)

### mask_SAM()
- **Type:** function
- **IO:** cube, reference_spectrum, angle -> single 2D binary mask
- **Status:** ✅ Keep
- **Purpose:** Creates a foreground mask using Spectral Angle Mapper (SAM) and optionally automatic thresholding (Otsu). Includes shadow correction and morphological cleaning. Optional visualization of angle map, before cleaning, final mask.
- **Notes:**
    - Does Otsu threshold need to be returned?

### mask_highpass_otsu()
- **Type:** function
- **IO:** cube, band_idx -> single 2D binary mask
- **Status:** ❓ Review
- **Purpose:** Creates a foreground mask using high-pass filtering via Gaussian subtraction and automatic thresholding (Otsu). Detects high-frequency spatial variation (local intensity changes / edges), which is related to texture. Includes morphological cleaning. Optional visualization.
- **Notes:**
    - Esentially operates on an image at a single band even though the actual input is a cube.
    - Possible wrong application of high-pass filtering.
    - Can add manual thresholding.
    - Check actual functionality and consider deleting.

### mask_kmeans()
- **Type:** function
- **IO:** cube -> single 2D binary mask
- **Status:** ✅ Keep
- **Purpose:** Creates a foreground mask using k-means (k=2) on standardized spectra. Includes shadow correction and morphological cleaning. Optional visualization.
- **Notes:**
    - Check functionality and processing time.

### mask_from_pca()
- **Type:** function
- **IO:** pca_cube, cube -> single 2D binary mask
- **Status:** ✅ Keep
- **Purpose:** Creates a foreground mask by selecting the best PC (manual, contrast, std, entropy, otsu) and either use automatic thresholding (Otsu) or boost contrast with manual thresholding. Includes shadow correction and morphological cleaning. Optional visualization.
- **Notes:**
    - Need to perform PCA individually, consider integrating it into the function.
    - Similar to mask_top_contrast - consider merging

### fixed_rect_extraction()
- **Type:** function
- **IO:** 2D binary mask, sample_size -> list of individual rectangular masks (coords)
- **Status:** 🚚 Move / ✅ Keep
- **Purpose:** From a 2D binary mask, find connected components and return fixed-size rectangular masks centered on each object's centroid. Can group and sort either row-wise or column-wise. Optional visualization (numbered masks).
- **Notes:**
    - Relatively poor fit with other masking functions in this module - consider moving
    - Good addition for functions that return a single mask

---

# 📦 cleaning.py

[⬅ Back to Overview](#overview)

### DeadPixelProcessor


### detect_dead_and_outlier_pixels()


### detect_line_defects()


### identify_dead_pixels()


### interpolate_dead_pixels()


### plot_defect_summary()



---

# 📦 extractors.py

[⬅ Back to Overview](#overview)

### ROIExtractor


### extract_local_mean()


### Grid_ROI_extractor()


---

# 📦 feature_selection.py

[⬅ Back to Overview](#overview)

### AdaptiveSpectralBinner


### adaptive_binning_by_gradient()


### compute_bin_centers()


### compute_binned_stats()


### plot_spectrum_with_bins()


### rmse_cv()


### CARS()


---

# 📦 hsi_io.py

[⬅ Back to Overview](#overview)

### find_hsi_basepaths()


### load_hsi_raw()


### load_wavelengths()


### load_sample_mapping()


### load_hsi_batch()


### export_tiff_stack()


---

# 📦 masking_utility.py

[⬅ Back to Overview](#overview)

### otsu_separation_score()


### get_valid_regions()


### estimate_rect_size()


### generate_rect_mask()


### rect_mask()


### extract_sample_cubes()


### extract_sample_cubes_from_masks()


---

# 📦 preprocessing.py

[⬅ Back to Overview](#overview)

### SNV


### MSC


### SavitzkyGolay


### normalize_min_max()


### normalize_mean_std()



---

# 📦 sandbox.py

[⬅ Back to Overview](#overview)

### SoftPLSDA


### MNF


### adaptive_equalize_spectrum()


### reflectance_to_absorbance()


### asls_baseline()


### robust_snv()


### rnv()



---

# 📦 pipeline.py

[⬅ Back to Overview](#overview)

### HSIImporter


### HSIProcessor


### HSIProcessorV2



---

# 📦 visualizations.py

[⬅ Back to Overview](#overview)

### plot_image()


### plot_spectra()


### plot_spectral_hist()


### plot_3D_slices()


### plot_3D_slices_interactive()


### plot_hsi_cube()

