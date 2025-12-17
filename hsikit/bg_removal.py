import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.measure import shannon_entropy, label, regionprops
from scipy.ndimage import find_objects
from scipy.ndimage import generate_binary_structure, binary_closing, binary_opening, binary_fill_holes

# Masks

def manual_rect_split(cube, sample_size, grid_shape, start=(0, 0), spacing=0, visualize=False):
    """
    Generate binary masks for rectangular samples in a HSI scene.

    Parameters:
    - cube (ndarray): hyperspectral image of shape (h, w, b)
    - sample_size (tuple): sample height, sample width
    - grid_shape (tuple): rows, cols
    - start (tuple): starting top left corner (y0, x0)
    - spacing (int or tuple): uniform or separate vectical/horizontal spacing (dy, dx)

    Returns:
    - masks (list): list of 2D binary masks (h, w)
    """

    h, w = cube.shape[:2]
    sample_h, sample_w = sample_size
    rows, cols = grid_shape
    y0, x0 = start

    if isinstance(spacing, int):
        dy = dx = spacing
    else:
        dy, dx = spacing

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

def mask_manual_pca_thresh(pca_images: np.ndarray, n_pca: int):
    """
    Manual threshold operation on a selected PCA image.
    Histogram visualization to simplify the selection of a threshold value.

    Parameters:
    - pca_images (array): previously computed array of PCA images
    - n_pca (array): index of a PCA image used for thresholding
    - threshold (float): upper threshold value (between 0 and 1)
    - visualize (bool): whether to plot pca images and histogram

    Returns:
    - binary_mask (array): 2D boolean array
    """
    pc = pca_images[:, :, n_pca]
    pc_norm = (pc - np.min(pc)) / (np.max(pc) - np.min(pc))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    pc_hist = pc_norm.flatten()
    plt.hist(pc_hist, bins=100)
    plt.grid(True)
    plt.show()

    threshold = input('Input the threshold value (between 0 and 1):')
    binary_mask = pc_norm < threshold
    plt.imshow(binary_mask, cmap='gray')
    plt.set_title(f"Thresholded PC{n_pca+1} (T = {threshold:.2f})")
    
    return binary_mask

def mask_top_contrast(hs_cube, top_n=5, low=0.3, high=0.7, shadow_percentile=10, min_size=500, hole_size=100, manual_max_band=None, visualize=False):
    """
    Compute a foreground mask by selecting top_n bands with highest contrast,
    boosting contrast, thresholding, and combining masks by majority voting.
    
    Parameters:
    - hs_cube: np.array, shape (H, W, Bands), hyperspectral data cube
    - top_n: int, number of top contrast bands to use
    - low, high: float, contrast stretching limits (0 to 1)
    - visualize: bool, whether to plot intermediate results
    
    Returns:
    - combined_mask: np.array of bools, foreground mask
    """

    def contrast(img):
        norm = (img - img.min()) / (img.max() - img.min())
        return norm.std()

    def boost(img):
        img = np.clip((img - low) / (high - low), 0, 1)
        return img

    if manual_max_band is not None:
        num_bands = manual_max_band
    else:
        num_bands = hs_cube.shape[2]
        
    contrast_vals = np.zeros(num_bands)

    for b in range(num_bands):
        contrast_vals[b] = contrast(hs_cube[:, :, b])

    top_indices = contrast_vals.argsort()[-top_n:][::-1]
    masks = []

    for idx in top_indices:
        band = hs_cube[..., idx]
        norm_band = (band - band.min()) / (band.max() - band.min())
        contrast_band = boost(norm_band)

        p_low = np.percentile(contrast_band, shadow_percentile)
        contrast_band[contrast_band < p_low] = 0
        
        thresh = threshold_otsu(contrast_band)
        mask = contrast_band > thresh
        masks.append(mask)

    combined_mask = np.sum(masks, axis=0) > (top_n // 2)
    combined_mask = remove_small_objects(combined_mask, min_size=min_size)
    combined_mask = remove_small_holes(combined_mask, area_threshold=hole_size)

    if visualize:
        fig, ax = plt.subplots(1, top_n + 1, figsize=(4 * (top_n + 1), 4))
        for i, idx in enumerate(top_indices):
            ax[i].imshow(hs_cube[..., idx], cmap='gray')
            ax[i].set_title(f'Band {idx}')
            ax[i].axis('off')
            ax[i].imshow(masks[i], cmap='Reds', alpha=0.3)
        ax[-1].imshow(combined_mask, cmap='gray')
        ax[-1].set_title('Combined mask')
        ax[-1].axis('off')
        plt.tight_layout()
        plt.show()

    return combined_mask

def mask_top_contrastV2(hs_cube, top_n=5, low=0.3, high=0.7, shadow_percentile=10, ycrop=0, xcrop=0, min_size=500, hole_size=100, manual_max_band=None, visualize=False, cube_name=None):
    original_shape = hs_cube.shape[:2]
    
    # Symmetric crop
    hs_crop = hs_cube[ycrop:original_shape[0]-ycrop, xcrop:original_shape[1]-xcrop, :]
    
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

        p_low = np.percentile(contrast_band, shadow_percentile)
        contrast_band[contrast_band < p_low] = 0
        
        thresh = threshold_otsu(contrast_band)
        mask = contrast_band > thresh
        masks.append(mask)

    combined_crop_mask = np.sum(masks, axis=0) > (top_n // 2)
    combined_crop_mask = remove_small_objects(combined_crop_mask, min_size=min_size)
    combined_crop_mask = remove_small_holes(combined_crop_mask, area_threshold=hole_size)

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
        title = f'Combined mask - {cube_name}' if cube_name else 'Combined mask'
        ax[-1].set_title(title)
        ax[-1].axis('off')
        plt.tight_layout()
        plt.show()

    return combined_mask

def mask_highpass_otsu(hs_cube, band_index, sigma=5, min_region_size=200, visualize=False):
    """
    Creates a foreground mask using high-pass filtering (Gaussian subtraction),
    Otsu thresholding, and morphological cleaning.

    Parameters:
    - hs_cube: numpy array (H, W, Bands)
    - band_index: index of the band to process (choose a band with clear texture)
    - sigma: Gaussian blur sigma (larger sigma = more background suppression)
    - min_region_size: minimum size of regions to retain (in pixels)
    - visualize: bool, whether to plot steps

    Returns:
    - final_mask: cleaned binary mask (foreground=True)
    """

    band_img = hs_cube[:, :, band_index]

    # High-pass filtering via Gaussian subtraction
    blurred = gaussian_filter(band_img, sigma=sigma)
    high_pass = band_img - blurred

    # Normalize to [0, 1]
    high_pass = (high_pass - high_pass.min()) / (high_pass.max() - high_pass.min())

    # Otsu thresholding
    thresh = filters.threshold_otsu(high_pass)
    mask = high_pass > thresh

    # Morphological cleaning
    mask = remove_small_objects(mask, min_size=min_region_size)
    mask = remove_small_holes(mask, area_threshold=min_region_size)
    mask = ~mask

    if visualize:
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(band_img, cmap='gray')
        plt.title(f'Original Band {band_index}')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(high_pass, cmap='gray')
        plt.title('High-Pass Filtered Image')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(mask, cmap='gray')
        plt.title('Final Mask')
        plt.axis('off')

        plt.show()

    return mask

def mask_kmeans(hs_cube, shadow_percentile=2, n_clusters=2, target='small', cleaning_structure=5, visualize=True):
    """
    Generates a cleaned binary mask from a hyperspectral cube using K-Means clustering.

    Parameters:
    - hs_cube: numpy array of shape (height, width, bands)
    - shadow_percentile: float, pixels below this percentile (brightness) are masked as shadow
    - n_clusters: int, number of KMeans clusters (default 2)
    - target: 'small' or 'large' - selects which cluster is treated as foreground
    - cleaning_structure: int, size of morphological structuring element
    - visualize: bool, if True shows visualizations

    Returns:
    - final_mask: 2D boolean array (True = foreground)
    """
    h, w, bands = hs_cube.shape
    pixels = hs_cube.reshape(-1, bands)

    # --- Step 1: Shadow Removal ---
    brightness = pixels.mean(axis=1)
    threshold = np.percentile(brightness, shadow_percentile)
    shadow_mask = brightness > threshold  # Keep only brighter pixels

    # --- Step 2: Spectral Normalization (only on non-shadow pixels) ---
    pixels_normalized = pixels.copy()
    pixels_normalized[shadow_mask] = (pixels[shadow_mask] - pixels[shadow_mask].mean(axis=1, keepdims=True)) / \
                                     (pixels[shadow_mask].std(axis=1, keepdims=True) + 1e-8)

    # --- Step 3: K-Means Clustering ---
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = np.full(pixels.shape[0], fill_value=-1)
    labels[shadow_mask] = kmeans.fit_predict(pixels_normalized[shadow_mask])

    mask = labels.reshape(h, w)

    counts = np.bincount(labels[shadow_mask])
    foreground_cluster = np.argmin(counts) if target == 'small' else np.argmax(counts)

    # Build binary mask
    binary_mask = (mask == foreground_cluster)

    # --- Step 4: Morphological Cleaning ---
    struct_elem = np.ones((cleaning_structure, cleaning_structure), dtype=bool)
    mask_cleaned = binary_opening(binary_mask, structure=struct_elem)
    mask_cleaned = binary_closing(mask_cleaned, structure=struct_elem)

    # --- Step 5: Visualization ---
    if visualize:
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Raw K-Means Mask')
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mask_cleaned, cmap='gray')
        plt.title('After Morphological Cleaning')
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(shadow_mask.reshape(h, w), cmap='gray')
        plt.title('Shadow Mask (Brighter Pixels)')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return mask_cleaned

# Helper function
def otsu_separation_score(pc_img):
    hist, bin_edges = np.histogram(pc_img.ravel(), bins=256, range=(pc_img.min(), pc_img.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    total = hist.sum()
    total_mean = (hist * bin_centers).sum() / total

    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg

    mean_bg = np.cumsum(hist * bin_centers) / (weight_bg + 1e-10)
    mean_fg = (total_mean * total - np.cumsum(hist * bin_centers)) / (weight_fg + 1e-10)

    between_var = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    between_var = np.nan_to_num(between_var, nan=0.0, posinf=0.0, neginf=0.0)

    max_var = np.max(between_var)

    return max_var

def mask_from_pca(pca_cube, hsi_raw, shadow_percentile=20,
                  selection_mode='contrast', mode='threshold',
                  threshold_cont=0.9, upper_cont=0.7, lower_cont=0.3,
                  visualize=False, verbose=False):
    """
    Create a binary mask from a selected PCA component and either otsu thresholding or contrast increase.

    Parameters:
    - pca_cube (array): previously computed PCA data cube, shape (H, W, n_components)
    - hsi_raw (array): original hsi data, expected shape (H, W, B)
    - shadow_percentile (float): shadow removal percentile
    - selection_mode (str or int): possible modes for best PC: 'contrast', 'std', 'entropy', 'otsu' and int (0-n_components) for manual selection
    - mode (str): either 'threshold' or 'contrast'
    - threshold_cont (float): if mode='contrast' - threshold above which the image is True
    - upper_cont (float): if mode='contrast' - upper boundary for contrast clipping
    - lower_cont (float): if mode='contrast' - lower boundary for contrast clipping

    Returns:
    - binary_mask (array): The binary mask as a boolean array.
    """
    
    if not (pca_cube.ndim == 3 and hsi_raw.ndim == 3):
        raise ValueError("pca_cube and hsi_raw must be 3D arrays.")
    if pca_cube.shape[:2] != hsi_raw.shape[:2]:
        raise ValueError("Spatial dimensions of pca_cube and hsi_raw must match.")
    n_components = pca_cube.shape[2]

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

    structure = generate_binary_structure(2, 2)
    binary_mask = binary_opening(binary_mask, structure=structure)
    binary_mask = binary_closing(binary_mask, structure=structure)
    binary_mask = binary_fill_holes(binary_mask)

    mean_reflectance = hsi_raw.mean(axis=2)
    shadow_mask = mean_reflectance < np.percentile(mean_reflectance, shadow_percentile)
    binary_mask[shadow_mask] = False

    if visualize:
        plt.imshow(binary_mask, cmap='gray')
        plt.title('Generated binary mask')
        plt.axis('off')
        plt.show()
    
    return binary_mask

# Get regions, generate synthetic rectangles, extract data

def fixed_rect_mask_rowwise(binary_mask, rect_height, rect_width, min_frac=0.9, visualize=True):
    """
    From a binary mask, find connected components and return fixed-size
    rectangular masks centered on each object’s centroid.

    Parameters:
    - binary_mask (array):
    - rect_height (int):
    - rect_width (int):
    - min_frac (float): how much mask needs to be in True region (0-1)
    - visualize (bool):

    Returns:
    - masks_list_sorted (list)
    - coords_list_sorted (list)
    """
    labeled = label(binary_mask)
    regions = regionprops(labeled)

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
            top = bottom - rect_height if bottom - rect_height >= 0 else 0
        if right - left < rect_width:
            left = right - rect_width if right - rect_width >= 0 else 0

        rect_region = binary_mask[top:bottom, left:right]
        rect_area = (bottom - top) * (right - left)
        frac_true = np.sum(rect_region) / rect_area

        if frac_true < min_frac:
            continue

        rect_mask = np.zeros_like(binary_mask, dtype=bool)
        rect_mask[top:bottom, left:right] = True

        masks_list.append(rect_mask)
        coords_list.append((top, left))

    # Group rectangles into rows based on top coordinate within tolerance
    if coords_list:
        coords_array = np.array(coords_list)
        tops = coords_array[:, 0]
        sorted_indices = np.argsort(tops)
        coords_array = coords_array[sorted_indices]
        masks_list = [masks_list[i] for i in sorted_indices]

        rows = []
        current_row = []
        last_top = coords_array[0, 0]

        for coord, mask in zip(coords_array, masks_list):
            if abs(coord[0] - last_top) > rect_height // 4:
                # Start new row
                rows.append(current_row)
                current_row = [(coord, mask)]
                last_top = coord[0]
            else:
                current_row.append((coord, mask))
        rows.append(current_row)

        # Sort each row left-to-right
        sorted_items = []
        for row in rows:
            row_sorted = sorted(row, key=lambda x: x[0][1])  # sort by left coordinate
            sorted_items.extend(row_sorted)

        coords_list_sorted, masks_list_sorted = zip(*sorted_items) if sorted_items else ([], [])

    else:
        coords_list_sorted, masks_list_sorted = [], []

    if visualize:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(binary_mask, cmap='gray')
        ax.set_title('Combined mask')

        for (top, left) in coords_list_sorted:
            rect = patches.Rectangle((left, top), rect_width, rect_height,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        plt.axis('off')
        plt.show()

        # Plot each rectangular mask individually
        n = len(masks_list_sorted)
        fig, axs = plt.subplots(1, n, figsize=(3*n, 4))
        if n == 1:
            axs = [axs]
        for i, (mask_rect, axm) in enumerate(zip(masks_list_sorted, axs)):
            axm.imshow(mask_rect, cmap='gray')
            axm.set_title(f'Mask {i+1}')
            axm.axis('off')
        plt.show()

    return list(masks_list_sorted), list(coords_list_sorted)

def fixed_rect_mask_columnwise(binary_mask, rect_height, rect_width, min_frac=0.9, visualize=True):
    """
    From a binary mask, find connected components and return fixed-size
    rectangular masks centered on each object’s centroid.

    Parameters:
    - binary_mask (array):
    - rect_height (int):
    - rect_width (int):
    - min_frac (float): how much mask needs to be in True region (0-1)
    - visualize (bool):

    Returns:
    - masks_list_sorted (list)
    - coords_list_sorted (list)
    """
    labeled = label(binary_mask)
    regions = regionprops(labeled)

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
            top = bottom - rect_height if bottom - rect_height >= 0 else 0
        if right - left < rect_width:
            left = right - rect_width if right - rect_width >= 0 else 0

        rect_region = binary_mask[top:bottom, left:right]
        rect_area = (bottom - top) * (right - left)
        frac_true = np.sum(rect_region) / rect_area

        if frac_true < min_frac:
            continue

        rect_mask = np.zeros_like(binary_mask, dtype=bool)
        rect_mask[top:bottom, left:right] = True

        masks_list.append(rect_mask)
        coords_list.append((top, left))

    # Group rectangles into columns based on left coordinate within tolerance
    if coords_list:
        coords_array = np.array(coords_list)
        lefts = coords_array[:, 1]
        sorted_indices = np.argsort(lefts)
        coords_array = coords_array[sorted_indices]
        masks_list = [masks_list[i] for i in sorted_indices]

        cols = []
        current_col = []
        last_left = coords_array[0, 1]

        for coord, mask in zip(coords_array, masks_list):
            if abs(coord[1] - last_left) > rect_width // 4:
                # Start new row
                cols.append(current_col)
                current_col = [(coord, mask)]
                last_left = coord[1]
            else:
                current_col.append((coord, mask))
        cols.append(current_col)

        # Sort each column top-to-bottom
        sorted_items = []
        for col in cols:
            col_sorted = sorted(col, key=lambda x: x[0][0])  # sort by top coordinate
            sorted_items.extend(col_sorted)

        coords_list_sorted, masks_list_sorted = zip(*sorted_items) if sorted_items else ([], [])

    else:
        coords_list_sorted, masks_list_sorted = [], []

    if visualize:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(binary_mask, cmap='gray')
        ax.set_title('Combined mask')

        for (top, left) in coords_list_sorted:
            rect = patches.Rectangle((left, top), rect_width, rect_height,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        plt.axis('off')
        plt.show()

        # Plot each rectangular mask individually
        n = len(masks_list_sorted)
        fig, axs = plt.subplots(1, n, figsize=(3*n, 4))
        if n == 1:
            axs = [axs]
        for i, (mask_rect, axm) in enumerate(zip(masks_list_sorted, axs)):
            axm.imshow(mask_rect, cmap='gray')
            axm.set_title(f'Mask {i+1}')
            axm.axis('off')
        plt.show()

    return list(masks_list_sorted), list(coords_list_sorted)

def get_valid_regions(binary_mask, true_threshold, n_regions):
    labeled_mask = label(binary_mask)
    regions = find_objects(labeled_mask)

    region_data = []
    for region in regions:
        if region is None:
            continue
        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop

        region_area = (y2 - y1) * (x2 - x1)
        true_ratio = np.sum(binary_mask[y1:y2, x1:x2]) / region_area

        if true_ratio >= true_threshold:
            region_data.append((region_area, region))

    region_data.sort(reverse=True, key=lambda x: x[0])
    selected_regions = [r[1] for r in region_data[:n_regions]]

    return selected_regions

def estimate_rect_size(selected_regions, margin):
    heights = []
    widths = []

    for region in selected_regions:
        if region is None:
            continue

        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop

        heights.append(y2 - y1)
        widths.append(x2 - x1)

    median_height = int(np.median(heights))
    median_width = int(np.median(widths))

    rect_height = max(1, median_height - 2 * margin)
    rect_width = max(1, median_width - 2 * margin)

    return rect_height, rect_width

def generate_rect_mask(binary_mask_shape, selected_regions, rect_height, rect_width):
    mask = np.zeros(shape, dtype=bool)

    for region in selected_regions:
        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop
        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2

        ry1 = max(0, center_y - rect_height // 2)
        ry2 = min(shape[0], center_y + rect_height // 2)
        rx1 = max(0, center_x - rect_width // 2)
        rx2 = min(shape[1], center_x + rect_width // 2)

        mask[ry1:ry2, rx1:rx2] = True

    return mask

def rect_mask(binary_mask, rect_height: int, rect_width: int, true_threshold: float, n_regions: int, original_pca_image, original_data):
    structure = generate_binary_structure(2, 2)
    binary_mask_closed = binary_closing(binary_mask, structure=structure)

    rect_mask = np.zeros_like(binary_mask, dtype=bool)
    labeled_mask, num_features = label(binary_mask_closed)

    regions = find_objects(labeled_mask)

    region_data = []
    for region in regions:
        if region is None:
            continue

        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop

        region_size = (y2 - y1) * (x2 - x1)
        true_ratio = np.sum(binary_mask_closed[y1:y2, x1:x2]) / region_size

        if true_ratio > true_threshold:
            region_data.append((region_size, region))

    region_data.sort(reverse=True, key=lambda x: x[0])
    selected_regions = [r[1] for r in region_data[:n_regions]]

    for region in selected_regions:
        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop

        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    
        ry1, ry2 = max(0, center_y - rect_height // 2), min(binary_mask.shape[0], center_y + rect_height // 2)
        rx1, rx2 = max(0, center_x - rect_width // 2), min(binary_mask.shape[1], center_x + rect_width // 2)

        rect_mask[ry1:ry2, rx1:rx2] = True

    overlay = np.zeros((*rect_mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = rect_mask * 1.0
    overlay[..., 3] = rect_mask * 0.4


    plt.figure(figsize=(15, 5))
    plt.imshow(original_pca_image, cmap='grey')
    plt.imshow(overlay)
    plt.title("Original PCA with overlaid mask")
    plt.axis('off')

    for i, region in enumerate(selected_regions):
        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop
        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
        plt.text(center_x, center_y, str(i+1), color='white', fontsize=12)

    plt.show()

    samples_dict = {}

    for i, region in enumerate(selected_regions):
        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop

        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2

        adjusted_y1 = max(0, center_y - rect_height // 2)
        adjusted_y2 = min(original_data.shape[1], center_y + rect_height // 2)
        adjusted_x1 = max(0, center_x - rect_width // 2)
        adjusted_x2 = min(original_data.shape[2], center_x + rect_width //2 )

        sample_cube = original_data[:, adjusted_y1:adjusted_y2, adjusted_x1: adjusted_x2]
        samples_dict[i] = sample_cube
    
    return samples_dict

def extract_sample_cubes(original_data, selected_regions, rect_height, rect_width):
    samples_dict = {}
    for i, region in enumerate(selected_regions):
        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2

        ry1 = max(0, cy - rect_height // 2)
        ry2 = min(original_data.shape[1], cy + rect_height // 2)
        rx1 = max(0, cx - rect_width // 2)
        rx2 = min(original_data.shape[2], cx + rect_width // 2)

        samples_dict[i] = original_data[:, ry1:ry2, rx1:rx2]
    return samples_dict

def extract_sample_cubes_from_masks(cube, masks, species_list=None):
    """
    Extract rectangular cubes from a hyperspectral cube using binary masks.

    Parameters:
    - cube (ndarray): shape (h, w, b)
    - masks: single 2D boolean mask or list of masks (h, w)
    - species_list: optional list of species names, same length as masks
    """

    if isinstance(masks, np.ndarray) and masks.ndim == 2:
        masks = [masks]

    samples_dict = {}

    for i, mask in enumerate(masks):
        ys, xs = np.where(mask)
        if len(ys) == 0 or len(xs) == 0:
            continue

        y1, y2 = ys.min(), ys.max() + 1
        x1, x2 = xs.min(), xs.max() + 1

        sample_cube = cube[y1:y2, x1:x2, :]

        if species_list:
            key = species_list[i]
            samples_dict.setdefault(key, []).append(sample_cube)
        else:
            samples_dict[i] = sample_cube

    return samples_dict