import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from scipy.ndimage import find_objects, generate_binary_structure, binary_closing

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
    mask = np.zeros(binary_mask_shape, dtype=bool)

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