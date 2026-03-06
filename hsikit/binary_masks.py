import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import shannon_entropy, label, regionprops
from scipy.ndimage import gaussian_filter, generate_binary_structure, binary_closing, binary_opening, binary_fill_holes

from typing import Optional, Literal

from masking_utility import otsu_separation_score

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
    Generate binary masks for rectangular samples in a HSI scene.

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

def mask_manual_pca_thresh(pca_image: NDArray, threshold: float = 0.8, visualize: bool = False) -> NDArray:
    """
    Manual threshold operation on a selected PCA image.
    Histogram visualization to simplify the selection of a threshold value.

    Parameters
    ----------
    pca_image : NDArray
        Previously computed PCA image, expected shape (H, W).
    threshold : float
        Threshold value in the range [0, 1].
    visualize : bool
        Plots the histogram and generated binary mask if True.

    Returns
    -------
    NDArray
        Mask as a 2D boolean array, shape (H, W).
    """
    pc_norm = (pca_image - np.min(pca_image)) / (np.max(pca_image) - np.min(pca_image))
    binary_mask = pc_norm < threshold
    
    if visualize:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        pc_hist = pc_norm.flatten()
        axs[0].hist(pc_hist, bins=100)
        axs[0].grid(True)

        axs[1].imshow(binary_mask, cmap='gray')
        axs[1].set_title(f"Thresholded PC (T = {threshold:.2f})")
        plt.show()
    
    return binary_mask

def mask_top_contrast(
    cube: NDArray,
    top_n: int = 5,
    cont_boost: tuple[float, float] = (0.3, 0.7),
    shadow_quantile: float = 0.1,
    min_size: int = 500,
    hole_size: int = 100,
    manual_max_band: Optional[int] = None,
    visualize: bool = False
) -> NDArray:
    """
    Compute a foreground mask by selecting top_n bands with highest contrast,
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
    min_size : int
        The minimum number of pixels for small objects removal (remove area if < min_size).
    hole_size : int
        Maximum hole size to fill for small holes removal (fill hole if < hole_size).
    manual_max_band : Optional[int]
        Optional index of maximum band to check for contrast.
    visualize: bool
        Plot intermediate results if True.
    
    Returns
    -------
    NDArray
        Mask as a 2D boolean array, shape (H, W).
    """
    low, high = cont_boost

    def contrast(img):
        norm = (img - img.min()) / (img.max() - img.min())
        return norm.std()

    def boost(img):
        img = np.clip((img - low) / (high - low), 0, 1)
        return img

    if manual_max_band is not None:
        num_bands = manual_max_band
    else:
        num_bands = cube.shape[2]
        
    contrast_vals = np.zeros(num_bands)

    for b in range(num_bands):
        contrast_vals[b] = contrast(cube[:, :, b])

    top_indices = contrast_vals.argsort()[-top_n:][::-1]
    masks = []

    for idx in top_indices:
        band = cube[..., idx]
        norm_band = (band - band.min()) / (band.max() - band.min())
        contrast_band = boost(norm_band)

        p_low = np.quantile(contrast_band, shadow_quantile)
        contrast_band[contrast_band < p_low] = 0
        
        thresh = threshold_otsu(contrast_band)
        mask = contrast_band > thresh
        masks.append(mask)

    combined_mask = np.sum(masks, axis=0) > (top_n // 2)
    combined_mask = remove_small_objects(combined_mask, max_size=min_size)
    combined_mask = remove_small_holes(combined_mask, max_size=hole_size)

    if visualize:
        fig, ax = plt.subplots(1, top_n + 1, figsize=(4 * (top_n + 1), 4))
        for i, idx in enumerate(top_indices):
            ax[i].imshow(cube[..., idx], cmap='gray')
            ax[i].set_title(f'Band {idx}')
            ax[i].axis('off')
            ax[i].imshow(masks[i], cmap='Reds', alpha=0.3)
        ax[-1].imshow(combined_mask, cmap='gray')
        ax[-1].set_title('Combined mask')
        ax[-1].axis('off')
        plt.tight_layout()
        plt.show()

    return combined_mask

def mask_top_contrastV2(
    cube: NDArray,
    top_n: int = 5,
    cont_boost: tuple[int, int] = (0.3, 0.7),
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
        
    contrast_vals = np.zeros(num_bands)
    for b in range(num_bands):
        contrast_vals[b] = contrast(hs_crop[:, :, b])

    top_indices = contrast_vals.argsort()[-top_n:][::-1]
    masks = []

    for idx in top_indices:
        band = hs_crop[..., idx]
        norm_band = (band - band.min()) / (band.max() - band.min())
        contrast_band = boost(norm_band)

        p_low = np.quantile(contrast_band, shadow_quantile)
        contrast_band[contrast_band < p_low] = 0
        
        thresh = threshold_otsu(contrast_band)
        mask = contrast_band > thresh
        masks.append(mask)

    combined_crop_mask = np.sum(masks, axis=0) > (top_n // 2)
    combined_crop_mask = remove_small_objects(combined_crop_mask, max_size=min_size)
    combined_crop_mask = remove_small_holes(combined_crop_mask, max_size=hole_size)

    # Place cropped mask back into full-size mask
    combined_mask = np.zeros(original_shape, dtype=bool)
    combined_mask[ycrop:original_shape[0]-ycrop, xcrop:original_shape[1]-xcrop] = combined_crop_mask

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

def fixed_rect_extraction(
    binary_mask: NDArray,
    rect_dims: tuple[int, int],
    mode: Literal['row', 'column'] = 'column',
    min_frac: float = 0.9,
    visualize: bool = True
) -> tuple[list, list]:
    """
    From a binary mask, find connected components and return fixed-size
    rectangular masks centered on each object's centroid.

    Can group and sort either row-wise or column-wise.

    Parameters
    ----------
    binary_mask : NDArray
        Base binary mask.
    rect_dims : tuple[int, int]
        Rectangle height and width.
    mode : Literal['row', 'column']
        Extraction mode - either row-wise or column-wise.
    min_frac : float
        Fraction of True pixels required in rectangle, range [0, 1].
    visualize : bool
        Plot the extracted masks if True.

    Returns
    -------
    tuple[list, list]
        A tuple containing:
        - sorted list of NDArray masks, shape (H, W)
        - sorted list of mask coordinates, tuple[int, int].
    """
    labeled = label(binary_mask)
    regions = regionprops(labeled)
    rect_height, rect_width = rect_dims

    masks_list = []
    coords_list = []
    H, W = binary_mask.shape

    for region in regions:
        cy, cx = region.centroid
        cy, cx = int(round(cy)), int(round(cx))

        top = max(0, cy - rect_height // 2)
        left = max(0, cx - rect_width // 2)
        bottom = min(H, top + rect_height)
        right = min(W, left + rect_width)

        if bottom - top < rect_height:
            top = max(0, bottom - rect_height)
        if right - left < rect_width:
            left = max(0, right - rect_width)

        rect_region = binary_mask[top:bottom, left:right]
        rect_area = (bottom - top) * (right - left)
        frac_true = np.sum(rect_region) / rect_area
        if frac_true < min_frac:
            continue

        rect_mask = np.zeros_like(binary_mask, dtype=bool)
        rect_mask[top:bottom, left:right] = True

        masks_list.append(rect_mask)
        coords_list.append((top, left))

    # Grouping
    if coords_list:
        coords_array = np.array(coords_list)
        if mode == 'row':
            group_coords = coords_array[:, 0] # top
            sort_key = 1 # sort each row by left
        elif mode == 'column':
            group_coords = coords_array[:, 1] # left
            sort_key = 0 # sort each column by top
        else:
            raise ValueError("mode must be 'row' or 'column'")

        sorted_indices = np.argsort(group_coords)
        coords_array = coords_array[sorted_indices]
        masks_list = [masks_list[i] for i in sorted_indices]

        groups = []
        current_group = []
        last_coord = coords_array[0, 0] if mode == 'row' else coords_array[0, 1]
        threshold = rect_height // 4 if mode == 'row' else rect_width // 4

        for coord, mask in zip(coords_array, masks_list):
            current_coord = coord[0] if mode == 'row' else coord[1]
            if abs(current_coord - last_coord) > threshold:
                groups.append(current_group)
                current_group = [(coord, mask)]
                last_coord = current_coord
            else:
                current_group.append((coord, mask))
        groups.append(current_group)

        # Sort each group
        sorted_items = []
        for group in groups:
            group_sorted = sorted(group, key=lambda x: x[0][sort_key])
            sorted_items.extend(group_sorted)

        coords_list_sorted, masks_list_sorted = zip(*sorted_items) if sorted_items else ([], [])
    else:
        coords_list_sorted, masks_list_sorted = [], []

    # Visualization
    if visualize:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(binary_mask, cmap='gray')
        ax.set_title('Combined mask')

        for i, (top, left) in enumerate(coords_list_sorted):
            rect = patches.Rectangle((left, top), rect_width, rect_height,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

            cx = left + rect_width / 2
            cy = top + rect_height / 2
            ax.text(cx, cy, str(i+1), color='red', fontsize=12,
                    ha='center', va='center', fontweight='bold')
        
        plt.axis('off')
        plt.show()

    return list(masks_list_sorted), list(coords_list_sorted)