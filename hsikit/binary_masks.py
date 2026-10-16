"""
Utilities for masking background of hyperspectral data cubes.

This module provides helper functions for producing binary masks using various functionality.

Note: This module is under active development and may change.
"""

import warnings
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects, disk
from skimage.measure import shannon_entropy, label, regionprops
from scipy.ndimage import gaussian_filter, generate_binary_structure, binary_closing, binary_opening, binary_fill_holes

from typing import Optional, Literal, Any

from hsikit.masking_utility import otsu_separation_score

# Masks

def manual_rect_split(
    cube: NDArray,
    sample_size: tuple[int, int],
    grid_shape: tuple[int, int],
    start: tuple[int, int] = (0, 0),
    spacing: int | tuple[int, int] = (0, 0),
    visualize: bool = False
) -> list[NDArray]:
    """
    Generate a grid of masks for rectangular samples in a HSI cube.  
    Returns a list of 2D binary masks.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    sample_size : tuple[int, int]
        Sample height and width.
    grid_shape : tuple[int, int]
        Number of rows and cols.
    start : tuple[int, int]
        Index of starting top left corner (y0, x0)
    spacing : int | tuple[int, int]
        Uniform if int or separate vectical/horizontal spacing if tuple (dy, dx)
    visualize : bool
        Plot masks over the average image of cube (across bands)

    Returns
    -------
    list[NDArray]
        List of 2D binary masks (H, W)
    """

    h, w = cube.shape[:2]
    sample_h, sample_w = sample_size
    rows, cols = grid_shape
    y0, x0 = start

    if isinstance(spacing, int):
        dy = dx = spacing
    elif isinstance(spacing, tuple):
        dy, dx = spacing
    else:
        raise TypeError("Wrong 'spacing' type - expected int or tuple")

    masks = []

    for r in range(rows):
        for c in range(cols):
            y_start = y0 + r * (sample_h + dy)
            y_end = y_start + sample_h
            x_start = x0 + c * (sample_w + dx)
            x_end = x_start + sample_w

            mask = np.zeros((h, w), dtype=bool)
            mask[y_start:y_end, x_start:x_end] = True
            masks.append(mask)

    if visualize:
        average_img = np.mean(cube, 2)
        plt.imshow(average_img, cmap='gray')
        plt.imshow(np.stack(masks).max(0), cmap='Reds', alpha=0.3)
        plt.show()

    return masks


def mask_top_contrast(
    cube: NDArray,
    top_n: int = 5,
    cont_boost: tuple[float, float] = (0.3, 0.7),
    shadow_quantile: float = 0.1,
    crop: tuple[int, int] = (0, 0),
    min_size: int = 500,
    hole_size: int = 100,
    manual_max_band: Optional[int] = None,
    visualize: bool = False,
    title: Optional[str] = None
) -> NDArray:
    """
    Compute a foreground mask by cropping the original cube, selecting top_n bands with highest contrast,
    boosting contrast, thresholding, and combining masks by majority voting.
    
    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    top_n : int
        Number of top contrast bands to use.
    cont_boost : tuple[float, float]
        Contrast boosting limits, both in the range [0, 1].
    shadow_quantile : float
        A quantile of pixels with low values (shadow pixels) in the range [0, 1].
    crop : tuple[int, int]
        Crop the original image and project the mask back into full-size mask.
    min_size : int
        Minimum size of regions to retain in pixels (remove area if < min_size).
    hole_size : int
        Maximum hole size to fill for small holes removal (fill hole if < hole_size).
    manual_max_band : Optional[int]
        Optional index of maximum band to check for contrast.
    visualize : bool
        Plot intermediate results if True.
    title : Optional[str]
        Optional plot title.
    
    Returns
    -------
    NDArray
        Mask as a 2D boolean array, shape (H, W).
    """
    ycrop, xcrop = crop
    low, high = cont_boost
    original_shape = cube.shape[:2]
    
    # crop
    hs_crop = cube[ycrop:original_shape[0]-ycrop, xcrop:original_shape[1]-xcrop, :]
    
    def contrast(img):
        norm = (img - img.min()) / (img.max() - img.min())
        return norm.std()

    def boost(img):
        img = np.clip((img - low) / (high - low), 0, 1)
        return img

    if manual_max_band is not None:
        num_bands = manual_max_band
    else:
        num_bands = hs_crop.shape[2]
        
    # Select bands with highest contrast
    contrast_vals = np.zeros(num_bands)
    for b in range(num_bands):
        contrast_vals[b] = contrast(hs_crop[:, :, b])
    
    top_indices = contrast_vals.argsort()[-top_n:][::-1]
    
    # Compute masks
    masks = []

    for idx in top_indices:
        band = hs_crop[..., idx]

        # Boost contrast
        norm_band = (band - band.min()) / (band.max() - band.min())
        contrast_band = boost(norm_band)

        # Shadow correction
        p_low = np.quantile(contrast_band, shadow_quantile)
        contrast_band[contrast_band < p_low] = 0
        
        # Automatic thresholding (Otsu)
        thresh = threshold_otsu(contrast_band)
        mask = contrast_band > thresh
        masks.append(mask)

    # Combine masks
    combined_crop_mask = np.sum(masks, axis=0) > (top_n // 2)

    # Morphological cleaning
    combined_crop_mask = remove_small_objects(combined_crop_mask, max_size=min_size)
    combined_crop_mask = remove_small_holes(combined_crop_mask, max_size=hole_size)

    # Place cropped mask back into full-size mask
    combined_mask = np.zeros(original_shape, dtype=bool)
    combined_mask[ycrop:original_shape[0]-ycrop, xcrop:original_shape[1]-xcrop] = combined_crop_mask

    # Visualization
    if visualize:
        fig, ax = plt.subplots(1, top_n + 1, figsize=(4 * (top_n + 1), 4))
        for i, idx in enumerate(top_indices):
            ax[i].imshow(hs_crop[..., idx], cmap='gray')
            ax[i].set_title(f'Band {idx}')
            ax[i].axis('off')
            ax[i].imshow(masks[i], cmap='Reds', alpha=0.3)
        ax[-1].imshow(combined_mask, cmap='gray')
        title = f'Combined mask - {title}' if title else 'Combined mask'
        ax[-1].set_title(title)
        ax[-1].axis('off')
        plt.tight_layout()
        plt.show()

    return combined_mask

def mask_SAM(
    cube: NDArray,
    ref_spectrum: NDArray,
    mode: Literal['otsu', 'manual'] = 'otsu',
    angle_thresh_deg: float = 10,
    shadow_quantile: Optional[float] = None,
    bg_reference: bool = True,
    post_process: bool = True,
    post_params: Optional[dict[str, Any]] = None,
    visualize: bool = False,
) -> NDArray | tuple[NDArray, float]:
    """
    Create and clean a foreground mask using Spectral Angle Mapper (SAM) and optionally Otsu thresholding.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B)
    ref_spectrum : NDArray
        Reference array, expected shape (B,)
    mode : Literal['otsu', 'manual']
        Threshold set mode - either 'otsu' or 'manual'
    angle_thresh_deg : float
        Angle threshold if mode='manual'
    shadow_quantile : Optional[float]
        Quantile threshold for removing shadow/low-intensity pixels
        Pixels below this intensity quantile are excluded from mask
    bg_reference : bool
        If the reference array belongs to background (inverts the mask)
    post_process : bool
        If post processing is applied to the mask
    post_params : Optional[dict]
        Parameters for post processing functions:
        - min_object_size : int (min object size to keep, def 500)
        - fill_holes : bool (whether to fill internal holes, def True)
        - radius : int (radius for morphological opening/closing, def 2)
        - smooth_sigma : float (sigma for optional Gaussian smoothing, def None)
    visualize : bool
        If True, plots input image, SAM angle image, mask before cleaning and final mask

    Returns
    -------
    mask_cln : NDArray
        2D cleaned mask of shape (H, W), dtype=bool
    threshold : float, optional
        Threshold value used (returned if mode='otsu')
    """
    
    if post_params is None:
        post_params = {}

    min_object_size = post_params.get("min_object_size", 500)
    fill_holes = post_params.get("fill_holes", True)
    radius = post_params.get("radius", 2)
    smooth_sigma = post_params.get("smooth_sigma", None)

    H, W, B = cube.shape

    # --- reshape ---
    pixels = cube.reshape(-1, B)

    # --- valid pixels ---
    pixel_norms = np.linalg.norm(pixels, axis=1)

    shadow_mask = None
    if shadow_quantile is not None:
        intensity = pixel_norms.reshape(H, W)

        # normalize
        intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-8)
        p_low = np.quantile(intensity_norm, shadow_quantile)
        shadow_mask = (intensity_norm > p_low) & (intensity > 1e-6) # keep only bright pixels

    valid_mask = pixel_norms > 1e-6

    pixels_norm = np.zeros_like(pixels)
    pixels_norm[valid_mask] = pixels[valid_mask] / pixel_norms[valid_mask, None]

    ref_norm = ref_spectrum / np.linalg.norm(ref_spectrum)

    # --- SAM ---
    cos_theta = np.zeros(len(pixels))
    cos_theta[valid_mask] = np.clip(
        np.dot(pixels_norm[valid_mask], ref_norm), -1.0, 1.0
    )

    angles_deg = np.full(len(pixels), np.nan)
    angles_deg[valid_mask] = np.degrees(np.arccos(cos_theta[valid_mask]))

    angle_map = angles_deg.reshape(H, W)

    # --- OTSU THRESHOLD ---
    valid_angles = angle_map[valid_mask.reshape(H, W)]
    
    if mode == 'otsu':
        thresh = threshold_otsu(valid_angles)
    elif mode == 'manual':
        thresh = angle_thresh_deg
    else:
        raise ValueError("mode can only be 'otsu' or 'manual'")

    mask = angle_map < thresh  # similar pixels

    if bg_reference:
        mask = ~mask

    if shadow_mask is not None:
        mask = mask & shadow_mask

    # --- POST PROCESSING ---
    if post_process:
        mask_cln = remove_small_objects(mask, max_size=min_object_size)
        if fill_holes:
            mask_cln = binary_fill_holes(mask_cln)
        mask_cln = binary_opening(mask_cln, structure=disk(radius))
        mask_cln = binary_closing(mask_cln, structure=disk(radius))
        if smooth_sigma is not None:
            smooth = gaussian_filter(mask_cln.astype(float), sigma=smooth_sigma)
            mask_cln = smooth > 0.5
    else:
        mask_cln = mask

    # --- visualization ---
    if visualize:
        fig, ax = plt.subplots(1, 4, figsize=(15, 5))

        ax[0].imshow(cube[:, :, 150], cmap="gray")
        ax[0].set_title("Input")

        im = ax[1].imshow(angle_map, cmap="inferno")
        ax[1].set_title(f"SAM angles (Threshold={thresh:.2f}°)")
        plt.colorbar(im, ax=ax[1])

        ax[2].imshow(mask, cmap="grey")
        ax[2].set_title("Mask before cleaning")

        ax[3].imshow(mask_cln, cmap="grey")
        ax[3].set_title("Final Mask")
        
        for a in ax:
            a.axis("off")

        plt.tight_layout()
        plt.show()

    if mode == 'otsu':
        return mask_cln, thresh
    else:
        return mask_cln

def mask_highpass_otsu(
    cube: NDArray,
    band_index: int,
    sigma: float = 5,
    min_size: int = 500,
    hole_size: int = 100,
    invert: bool = False,
    visualize: bool = False
) -> NDArray:
    """
    Creates a foreground mask using high-pass filtering (Gaussian subtraction),
    Otsu thresholding, and morphological cleaning.
    Detects high-frequency spatial variation (local intensity changes / edges), which is related to texture.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    band_index : int
        Index of the band to process.
    sigma : float
        Gaussian blur sigma (larger sigma = wider kernel).
    min_size : int
        Minimum size of regions to retain (in pixels).
    hole_size : int
        Maximum hole size to fill for small holes removal (fill hole if < hole_size).
    invert : bool
        Inverts the mask if True.
    visualize : bool
        Plots original band image, filtered image and final mask if True.

    Returns
    -------
    NDArray
        Mask as a 2D boolean array, shape (H, W).
    """

    band_img = cube[:, :, band_index]

    # High-pass filtering via Gaussian subtraction
    blurred = gaussian_filter(band_img, sigma=sigma)
    high_pass = band_img - blurred

    # Normalize to [0, 1]
    high_pass = (high_pass - high_pass.min()) / (high_pass.max() - high_pass.min())

    # Otsu thresholding
    thresh = threshold_otsu(high_pass)
    mask = high_pass > thresh

    # Morphological cleaning
    mask = remove_small_objects(mask, max_size=min_size)
    mask = remove_small_holes(mask, max_size=hole_size)
    if invert:
        mask = ~mask

    # Visualization
    if visualize:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))

        axs[0].imshow(band_img, cmap='gray')
        axs[0].set_title(f'Original Band {band_index}')
        axs[0].axis('off')

        axs[1].imshow(high_pass, cmap='gray')
        axs[1].set_title('High-Pass Filtered Image')
        axs[1].axis('off')

        axs[2].imshow(mask, cmap='gray')
        axs[2].set_title('Final Mask')
        axs[2].axis('off')

        plt.show()

    return mask

def mask_kmeans(
    cube: NDArray,
    shadow_quantile: float = 0.1,
    n_clusters: int = 2,
    target: Literal['small', 'large'] = 'small',
    cleaning_structure: int = 5,
    visualize: bool = True
) -> NDArray:
    """
    Generates a cleaned binary mask from a hyperspectral cube using K-Means clustering.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    shadow_quantile : float
        Pixels below this quantile (brightness) are masked as shadow.
    n_clusters : int
        Number of KMeans clusters (default 2).
    target : str ['small' or 'large']
        Selects which cluster is treated as foreground.
    cleaning_structure : int
        Size of morphological structuring element.
    visualize : bool
        Plots raw K-Means mask and the final mask if True.

    Returns
    -------
    NDArray
        Mask as a 2D boolean array, shape (H, W).
    """
    h, w, bands = cube.shape
    pixels = cube.reshape(-1, bands)

    # Shadow Removal
    brightness = pixels.mean(axis=1)
    threshold = np.quantile(brightness, shadow_quantile)
    shadow_mask = brightness > threshold  # Keep only brighter pixels

    # Spectral Normalization (only on non-shadow pixels)
    pixels_normalized = pixels.copy()
    pixels_normalized[shadow_mask] = (pixels[shadow_mask] - pixels[shadow_mask].mean(axis=1, keepdims=True)) / \
                                     (pixels[shadow_mask].std(axis=1, keepdims=True) + 1e-8)

    # K-Means Clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = np.full(pixels.shape[0], fill_value=-1)
    labels[shadow_mask] = kmeans.fit_predict(pixels_normalized[shadow_mask])

    mask = labels.reshape(h, w)

    counts = np.bincount(labels[shadow_mask])
    foreground_cluster = np.argmin(counts) if target == 'small' else np.argmax(counts)

    # Build binary mask
    binary_mask = (mask == foreground_cluster)

    # Morphological Cleaning
    struct_elem = np.ones((cleaning_structure, cleaning_structure), dtype=bool)
    mask_cleaned = binary_opening(binary_mask, structure=struct_elem)
    mask_cleaned = binary_closing(mask_cleaned, structure=struct_elem)

    # Visualization
    if visualize:
        fig, axs = plt.subplots(1, 2, figsize=(5, 4))

        axs[0].imshow(binary_mask, cmap='gray')
        axs[0].set_title('Raw K-Means Mask')
        axs[0].axis('off')

        axs[1].imshow(mask_cleaned, cmap='gray')
        axs[1].set_title('After Morphological Cleaning')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    return mask_cleaned

def mask_from_pca(
    pca_cube: NDArray,
    cube: NDArray,
    shadow_quantile: float = 0.1,
    selection_mode: Literal['contrast', 'std', 'entropy', 'otsu'] | int = 'contrast',
    mode: Literal['threshold', 'contrast'] = 'threshold',
    threshold_cont: float = 0.9,
    upper_cont: float = 0.7,
    lower_cont: float = 0.3,
    visualize: bool = False,
    verbose: bool = False
) -> NDArray:
    """
    Create a binary mask from a selected PCA component and either otsu thresholding or contrast increase.

    Parameters
    ----------
    pca_cube : NDArray
        Previously computed PCA data cube, shape (H, W, n_components).
    hsi_raw : NDArray
        HSI 3D array, expected shape (H, W, B).
    shadow_quantile : float
        Shadow removal quantile.
    selection_mode : Literal['contrast', 'std', 'entropy', 'otsu'] | int
        Possible modes to determine best PC: 'contrast', 'std', 'entropy', 'otsu' or int (0-n_components) for manual selection.
    mode : Literal['threshold', 'contrast']
        Operation mode, either 'threshold' or 'contrast'.
    threshold_cont : float
        If mode='contrast' - threshold above which the image is True.
    upper_cont : float
        If mode='contrast' - upper boundary for contrast clipping
    lower_cont : float
        If mode='contrast' - lower boundary for contrast clipping

    Returns
    -------
    NDArray
        Mask as a 2D boolean array, shape (H, W).
    """
    # Shape checking
    if not (pca_cube.ndim == 3 and cube.ndim == 3):
        raise ValueError("pca_cube and hsi_raw must be 3D arrays.")
    if pca_cube.shape[:2] != cube.shape[:2]:
        raise ValueError("Spatial dimensions of pca_cube and hsi_raw must match.")
    n_components = pca_cube.shape[2]

    # Best PC selection
    if isinstance(selection_mode, int):
        if 0 <= selection_mode < n_components:
            pc_image = pca_cube[:, :, selection_mode]
            if verbose:
                print(f"Using manually selected PCA component: {selection_mode}")
        else:
            raise ValueError(f"Integer selection_mode out of range (0 to {n_components - 1})")
    else:
        scores = []
        for i in range(n_components):
            pc_img = pca_cube[:, :, i]

            if selection_mode == 'contrast':
                score = pc_img.std() / (pc_img.max() - pc_img.min() + 1e-6)
            elif selection_mode == 'std':
                score = pc_img.std()
            elif selection_mode == 'entropy':
                score = shannon_entropy(pc_img)
            elif selection_mode == 'otsu':
                score = otsu_separation_score(pc_img)
            else:
                raise ValueError(f"Invalid selection_mode: '{selection_mode}'")

            scores.append(score)

        best_pc_index = int(np.argmax(scores))
        pc_image = pca_cube[:, :, best_pc_index]

        if verbose:
            print(f"Auto-selected PC {best_pc_index} using '{selection_mode}' with score {scores[best_pc_index]:.4f}")

    # Thresholding
    if mode == 'threshold':
        threshold = threshold_otsu(pc_image)
        binary_mask = pc_image > threshold
        if verbose:
            print(f'Otsu threshold: {threshold:.4f}')
    elif mode == 'contrast':
        img_cont = np.clip((pc_image - lower_cont) / (upper_cont - lower_cont), 0, 1)
        binary_mask = img_cont > threshold_cont
        if verbose:
            print(f'Contrast mode threshold: {threshold_cont}')
    else:
        raise ValueError(f"Invalid mode '{mode}'. Use 'threshold' or 'contrast'.")

    # Structure cleaning
    structure = generate_binary_structure(2, 2)
    binary_mask = binary_opening(binary_mask, structure=structure)
    binary_mask = binary_closing(binary_mask, structure=structure)
    binary_mask = binary_fill_holes(binary_mask)
    if binary_mask is None:
        raise RuntimeError("binary_fill_holes returned None unexpectedly")
    binary_mask = np.asarray(binary_mask)

    # Shadow mask
    mean_reflectance = cube.mean(axis=2)
    shadow_mask = mean_reflectance < np.quantile(mean_reflectance, shadow_quantile)
    binary_mask[shadow_mask] = False

    # Visualization
    if visualize:
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Generated binary mask')
        plt.axis('off')
        plt.show()
    
    return binary_mask

def rect_extraction(
    binary_mask: NDArray,
    rect_dims: tuple[int, int] | None = None,
    mode: Literal["row", "column"] = "column",
    min_frac: float = 0.9,
    visualize: bool = True,
    *,
    auto_size_mode: Literal["min", "minmin", "fit_all"] = "min",
    margin: int = 0,
    min_area: int = 10,
    fallback_dims: tuple[int, int] = (32, 32),
) -> tuple[list, list]:
    """
    Extract fixed-size rectangular masks centered on connected components.

    Rectangle dimensions can be provided manually or inferred automatically
    from the component bounding boxes.

    Parameters
    ----------
    binary_mask : NDArray
        Base binary mask.
    rect_dims : tuple[int, int] | None
        Rectangle height and width. If None, determine dimensions automatically.
    mode : Literal["row", "column"]
        Extraction ordering mode.
    min_frac : float
        Minimum fraction of foreground pixels required inside each rectangle.
    visualize : bool
        Plot the extracted rectangles if True.
    auto_size_mode : Literal["min", "minmin", "fit_all"]
        Automatic sizing strategy:
        - "min": independently minimize height and width across components.
        - "minmin": use the smallest height and width across all specimens.
        - "fit_all": use the maximum required height and width to fit all
          component bounding boxes.
    margin : int
        Inward margin applied to each side of each component bounding box
        when estimating automatic dimensions. Positive values shrink the
        estimated dimensions by 2 * margin. Does not affect manual dimensions.
    min_area : int
        Minimum allowed connected-component area for reliable automatic sizing.
        If any component is smaller, issue a warning and use fallback_dims.
    fallback_dims : tuple[int, int]
        Manually specified dimensions used if automatic sizing is unreliable.

    Returns
    -------
    tuple[list, list]
        (sorted rectangular masks, sorted (top, left) coordinates).
    """
    if binary_mask.ndim != 2:
        raise ValueError("binary_mask must be a 2D array.")
    if mode not in ("row", "column"):
        raise ValueError("mode must be 'row' or 'column'.")
    if not 0 <= min_frac <= 1:
        raise ValueError("min_frac must be between 0 and 1.")
    if margin < 0:
        raise ValueError("margin must be non-negative.")
    if min_area < 1:
        raise ValueError("min_area must be at least 1.")

    labeled = label(binary_mask)
    regions = regionprops(labeled)
    H, W = binary_mask.shape

    def validate_dims(dims, name):
        if (
            len(dims) != 2
            or any(not isinstance(d, (int, np.integer)) or d <= 0 for d in dims)
        ):
            raise ValueError(f"{name} must contain two positive integers.")
        if dims[0] > H or dims[1] > W:
            raise ValueError(f"{name} must not exceed the image dimensions.")
        return tuple(map(int, dims))

    fallback_dims = validate_dims(fallback_dims, "fallback_dims")

    # Determine rectangle dimensions.
    if rect_dims is not None:
        rect_height, rect_width = validate_dims(rect_dims, "rect_dims")
        per_sample_dims = None
    
    elif not regions:
        warnings.warn(
            "No connected components found; using fallback_dims.",
            UserWarning,
            stacklevel=2,
        )
        rect_height, rect_width = fallback_dims
        per_sample_dims = None
    
    else:
        too_small = [region for region in regions if region.area < min_area]
    
        if too_small:
            warnings.warn(
                f"{len(too_small)} component(s) have area below "
                f"min_area={min_area}. Using fallback_dims={fallback_dims}.",
                UserWarning,
                stacklevel=2,
            )
            rect_height, rect_width = fallback_dims
            per_sample_dims = None
    
        else:
            sizes = []
    
            for region in regions:
                min_row, min_col, max_row, max_col = region.bbox
                height = max_row - min_row - 2 * margin
                width = max_col - min_col - 2 * margin
    
                if height <= 0 or width <= 0:
                    warnings.warn(
                        "Margin leaves a non-positive bounding-box dimension. "
                        f"Using fallback_dims={fallback_dims}.",
                        UserWarning,
                        stacklevel=2,
                    )
                    sizes = None
                    rect_height, rect_width = fallback_dims
                    per_sample_dims = None
                    break
    
                sizes.append((height, width))
    
            if sizes is not None:
                sizes = np.asarray(sizes, dtype=int)
            
                if auto_size_mode == "minmin":
                    rect_height = int(sizes[:, 0].min())
                    rect_width = int(sizes[:, 1].min())
                    per_sample_dims = None
            
                elif auto_size_mode == "min":
                    per_sample_dims = sizes
                    rect_height = rect_width = None
            
                elif auto_size_mode == "fit_all":
                    rect_height = int(sizes[:, 0].max())
                    rect_width = int(sizes[:, 1].max())
                    per_sample_dims = None
            
                else:
                    raise ValueError(
                        "auto_size_mode must be 'min', 'minmin', or 'fit_all'."
                    )

    if per_sample_dims is None:
        if rect_height > H or rect_width > W:
            raise ValueError(
                f"Rectangle dimensions {(rect_height, rect_width)} exceed "
                f"image dimensions {(H, W)}."
            )
    else:
        if np.any(per_sample_dims[:, 0] > H) or np.any(per_sample_dims[:, 1] > W):
            raise ValueError("Some automatically detected dimensions exceed the image.")

    # Extract rectangles.
    masks_list = []
    coords_list = []
    dims_list = []
    
    for i, region in enumerate(regions):
        if per_sample_dims is not None:
            rect_height, rect_width = map(int, per_sample_dims[i])
    
        cy, cx = region.centroid
        cy, cx = int(round(cy)), int(round(cx))
    
        top = int(np.clip(cy - rect_height // 2, 0, H - rect_height))
        left = int(np.clip(cx - rect_width // 2, 0, W - rect_width))
        bottom = top + rect_height
        right = left + rect_width
    
        rect_region = binary_mask[top:bottom, left:right]
        frac_true = np.count_nonzero(rect_region) / (rect_height * rect_width)
    
        if frac_true < min_frac:
            continue
    
        rect_mask = np.zeros_like(binary_mask, dtype=bool)
        rect_mask[top:bottom, left:right] = True
    
        masks_list.append(rect_mask)
        coords_list.append((top, left))
        dims_list.append((rect_height, rect_width))

    # Group and sort rectangles.
    if coords_list:
        coords_array = np.asarray(coords_list)
    
        if mode == "row":
            group_axis = 0
            sort_key = 1
        else:
            group_axis = 1
            sort_key = 0
    
        # Use a consistent reference threshold for grouping.
        if per_sample_dims is not None:
            threshold_dims = np.median(per_sample_dims, axis=0).astype(int)
            threshold = max(1, int(threshold_dims[group_axis] // 4))
        else:
            threshold = max(
                1,
                (rect_height if mode == "row" else rect_width) // 4,
            )
    
        sorted_indices = np.argsort(coords_array[:, group_axis])
    
        items = [
            (coords_list[i], masks_list[i], dims_list[i])
            for i in sorted_indices
        ]
    
        groups = []
        current_group = []
        last_coord = items[0][0][group_axis]
    
        for item in items:
            current_coord = item[0][group_axis]
    
            if abs(current_coord - last_coord) > threshold and current_group:
                groups.append(current_group)
                current_group = []
    
            current_group.append(item)
            last_coord = current_coord
    
        if current_group:
            groups.append(current_group)
    
        sorted_items = []
        for group in groups:
            sorted_items.extend(
                sorted(group, key=lambda item: item[0][sort_key])
            )
    
        coords_list_sorted = [item[0] for item in sorted_items]
        masks_list_sorted = [item[1] for item in sorted_items]
        dims_list_sorted = [item[2] for item in sorted_items]
    
    else:
        coords_list_sorted, masks_list_sorted, dims_list_sorted = [], [], []

    # Visualization.
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(binary_mask, cmap="gray")
        ax.set_title(f"Extracted rectangles ({len(coords_list_sorted)} samples)")

        for i, ((top, left), (rect_height, rect_width)) in enumerate(
            zip(coords_list_sorted, dims_list_sorted)
        ):
            rect = patches.Rectangle(
                (left, top),
                rect_width,
                rect_height,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                left + rect_width / 2,
                top + rect_height / 2,
                str(i + 1),
                color="red",
                fontsize=12,
                ha="center",
                va="center",
                fontweight="bold",
            )

        ax.axis("off")
        plt.show()

    return masks_list_sorted, coords_list_sorted