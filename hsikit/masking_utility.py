"""
Various masking utilities supporting binary masks and BG removal classes.

Note: This module is under active development and may change.
"""

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy.ndimage import find_objects, generate_binary_structure, binary_closing

from typing import Optional

def otsu_separation_score(pc_img: NDArray) -> float:
    """
    Computes max variance - Otsu separation score.
    Used in mask_from_pca (binary_masks).

    Parameters
    ----------
    pc_img : NDArray
        PCA-based single component image.

    Returns
    -------
    float
    """
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

def get_valid_regions(binary_mask: np.ndarray, true_threshold: float, n_regions: int) -> list[tuple[slice, slice]]:
    """
    Find objects from a binary mask and return selected regions.

    Parameters
    ----------
    binary_mask : np.ndarray
        Bool/binary mask used for region selection.
    true_threshold : float
        Region is kept if the fraction of True pixels is >= threshold (0-1).
    n_regions : int
        Maximum number of regions to return.

    Returns
    -------
    list[tuple[slice, slice]]
        Selected regions represented as slices.
    """
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

def estimate_rect_size(selected_regions: list, margin: int) -> tuple[int, int]:
    """
    Estimates rectangle size based on selected regions.

    Parameters
    ----------
    selected_regions : list[tuple[slice, slice]]
        List of tuples with slice objects.
    margin : int
        Value by which to decrease height and width from both sides.

    Returns
    -------
    tuple[int, int]
        Returns tuple of rectangles median height and median width reduced by 2 * margin
    """
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

def generate_rect_mask(binary_mask_shape: tuple[int, int], selected_regions: list, rect_dims: tuple[int, int]) -> NDArray:
    """
    Generates rectangle mask based on selected regions and rectangle dimensions.

    Parameters
    ----------
    binary_mask_shape : tuple[int, int]
        The shape of input binary mask.
    selected_regions : list[tuple[slice, slice]]
        Regions selected from binary mask.
    rect_dims : tuple[int, int]
        Height and width of rectangle.
    
    Returns
    -------
    NDArray
        Rectangle binary mask based on binary mask.
    """
    mask = np.zeros(binary_mask_shape, dtype=bool)
    rect_height, rect_width = rect_dims

    for region in selected_regions:
        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop
        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2

        ry1 = max(0, center_y - rect_height // 2)
        ry2 = min(binary_mask_shape[0], center_y + rect_height // 2)
        rx1 = max(0, center_x - rect_width // 2)
        rx2 = min(binary_mask_shape[1], center_x + rect_width // 2)

        mask[ry1:ry2, rx1:rx2] = True

    return mask

def rect_mask(
    binary_mask: NDArray,
    rect_dims: tuple[int, int],
    true_threshold: float,
    n_regions: int,
    pca_image: NDArray,
    cube: NDArray
) -> dict:
    """
    Generates a binary mask of rectangles based on found objects and a base binary mask.

    Parameters
    ----------
    binary_mask : NDArray
        Base binary mask, expected shape (H, W)
    rect_dims : tuple[int, int]
        Height and width of rectangles.
    true_threshold : float
        Threshold for ratio between True and False pixels.
    n_regions : int
        Number of expected regions.
    pca_image : NDArray
        PCA-based image for visualization, expected shape (H, W)
    cube : NDArray
        HSI 3D array, expected shape (B, H, W)

    Returns
    -------
    dict[int, NDArray]
        Dictionary of extracted cubes with index keys (number in order of extraction)
    """
    structure = generate_binary_structure(2, 2)
    binary_mask_closed = binary_closing(binary_mask, structure=structure)

    rect_mask = np.zeros_like(binary_mask, dtype=bool)
    labeled_mask = label(binary_mask_closed)
    rect_height, rect_width = rect_dims

    regions = find_objects(labeled_mask) # returns list[tuple[slice, slice]]

    region_data = []
    for region in regions:
        if region is None:
            continue

        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop

        region_size = (y2 - y1) * (x2 - x1)
        true_ratio = np.sum(binary_mask_closed[y1:y2, x1:x2]) / region_size

        if true_ratio > true_threshold:
            region_data.append((region_size, region)) # Region data list[tuple[region_size, tuple[slice, slice]]]

    region_data.sort(reverse=True, key=lambda x: x[0]) # sort regions based on size
    selected_regions = [r[1] for r in region_data[:n_regions]] # select largest regions up to n_regions, list[tuple[slice, slice]]

    for region in selected_regions:
        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop

        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
    
        ry1, ry2 = max(0, center_y - rect_height // 2), min(binary_mask.shape[0], center_y + rect_height // 2)
        rx1, rx2 = max(0, center_x - rect_width // 2), min(binary_mask.shape[1], center_x + rect_width // 2)

        rect_mask[ry1:ry2, rx1:rx2] = True # Set largest rectangluar regions to True in a zeros_like (H, W) boolean mask

    # Set overlay for visualization
    overlay = np.zeros((*rect_mask.shape, 4), dtype=np.float32) # Overlay shape (H, W, 4) of zeros
    overlay[..., 0] = rect_mask * 1.0 # Red channel (RGB) - mask
    overlay[..., 3] = rect_mask * 0.4 # Alpha channel - 0.4

    # Visualization - PCA image + overlay
    plt.figure(figsize=(15, 5))
    plt.imshow(pca_image, cmap='grey')
    plt.imshow(overlay)
    plt.title("Original PCA with overlaid mask")
    plt.axis('off')

    # Number/label the regions on the plot
    for i, region in enumerate(selected_regions):
        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop
        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2
        plt.text(center_x, center_y, str(i+1), color='white', fontsize=12)

    plt.show()

    # Extract samples
    samples_dict = {}

    for i, region in enumerate(selected_regions):
        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop

        center_y, center_x = (y1 + y2) // 2, (x1 + x2) // 2

        adjusted_y1 = max(0, center_y - rect_height // 2)
        adjusted_y2 = min(cube.shape[1], center_y + rect_height // 2)
        adjusted_x1 = max(0, center_x - rect_width // 2)
        adjusted_x2 = min(cube.shape[2], center_x + rect_width //2 )

        sample_cube = cube[:, adjusted_y1:adjusted_y2, adjusted_x1: adjusted_x2] # Extract sample cubes, shape (B, H, W)
        samples_dict[i] = sample_cube # Add sample cubes to a dict, keyed by number in order of extraction (enumerate)
    
    return samples_dict

def extract_sample_cubes(cube: NDArray, selected_regions: list, rect_dims: tuple[int, int]) -> dict:
    """
    Extracts sample cubes from selected regions.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (B, H, W)
    selected_regions : list[tuple[slice, slice]]
        Regions selected from binary mask
    rect_dims : tuple[int, int]
        Height and width of rectangle.
    
    Returns
    -------
    dict[int, NDArray]
        Dictionary of extracted cubes with index keys (number in order of extraction)
    """
    samples_dict = {}
    rect_height, rect_width = rect_dims

    for i, region in enumerate(selected_regions):
        y1, y2 = region[0].start, region[0].stop
        x1, x2 = region[1].start, region[1].stop
        cy, cx = (y1 + y2) // 2, (x1 + x2) // 2

        ry1 = max(0, cy - rect_height // 2)
        ry2 = min(cube.shape[1], cy + rect_height // 2)
        rx1 = max(0, cx - rect_width // 2)
        rx2 = min(cube.shape[2], cx + rect_width // 2)

        samples_dict[i] = cube[:, ry1:ry2, rx1:rx2]
    return samples_dict

def extract_sample_cubes_from_masks(cube: NDArray, masks: list | NDArray, species_list: Optional[list] = None) -> dict:
    """
    Extract cubes from a hyperspectral cube using binary masks.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B)
    masks : list | NDArray
        List of masks or a single 2D boolean mask (H, W)
    species_list : Optional[list]
        Optional list of species names, same length as masks.

    Returns
    -------
    dict[int, NDArray]
        Dictionary of extracted cubes with index keys (number in order of extraction)
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