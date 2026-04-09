"""
Utilities for cleaning dead pixels from hyperspectral data.

This module provides helper functions for cleaning data (for example dead pixels).

Note: This module is under active development and may change.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import median_filter


def detect_dead_and_outlier_pixels(
        cube: np.ndarray,
        lower_thresh: float = 0.1,
        upper_thresh: float | None = None,
        z_thresh: int = 3
    ) -> np.ndarray:
    """
    Detect dead pixels and outliers in a hyperspectral cube.  
    Dead pixel detection based on lower and upper threshold values set.  
    Outlier detection based on z-score, per band.

    Parameters
    ----------
    cube : np.ndarray
        HSI 3D array, expected shape (H, W, B).
    lower_thresh : float
        Values below this are considered dead.
    upper_thresh : float
        Values above this are considered dead (None -> use max value).
    z_thresh : int
        Threshold for outlier detection in std units.

    Returns
    -------
    np.ndarray
        Boolean mask, True = good pixel, False = dead/outlier
    """
    if upper_thresh is None:
        upper_thresh = np.max(cube)
    
    # Dead pixels
    dead_mask = (cube < lower_thresh) | (cube > upper_thresh)
    
    # Outliers (z-score based, per band)
    band_mean = np.mean(cube, axis=(0,1))
    band_std  = np.std(cube, axis=(0,1))
    
    # Broadcast to cube shape
    z_score = np.abs((cube - band_mean[None,None,:]) / (band_std[None,None,:] + 1e-8))
    outlier_mask = z_score > z_thresh
    
    # Combine masks: True = good, False = bad
    mask = ~(dead_mask | outlier_mask)
    return mask


def detect_line_defects(cube: np.ndarray, z_thresh: float = 5.0, min_fraction: float = 0.9) -> tuple[np.ndarray, list[tuple[int,int]]]:
    """
    Detect (col, band) defects where most rows at that (col,band) are extreme outliers.
    Returns a boolean array defect_mask of shape (rows, cols, bands) and a summary list of (col,band).
    
    Strategy
    --------
    1. For each band, compute the column-wise values averaged over rows (or L2 across rows).
    2. Compute a z-score across columns for that band; columns with high z indicate anomaly.
    3. Verify that the anomaly affects >= min_fraction of rows at that (col,band) by checking
    per-row deviation at that (col,band).
    
    Parameters
    ----------
    cube : ndarray
        HSI 3D array, expected shape (H, W, B)
    z_thresh : float
        z-score threshold for column anomaly at each band
    min_fraction : float
        In (0,1], fraction of rows that must be outliers to call it a line defect
    
    Returns
    -------
    defect_mask : np.ndarray
        Shape (H, W, B), True where pixel considered defective
    defects_list : list[tuple[int,int]]
        List of defects (col, band)
    """
    r, c, b = cube.shape
    defect_mask = np.zeros_like(cube, dtype=bool)
    defects_list = []
    
    # operate band-by-band
    for bi in range(b):
        band = cube[:, :, bi].astype(float)  # shape (r, c)
        # column summary: median across rows (robust) or mean
        col_med = np.median(band, axis=0)   # shape (c,)
        # robust z via median absolute deviation (less sensitive to extremes)
        med = np.median(col_med)
        mad = np.median(np.abs(col_med - med)) + 1e-9
        zcols = (col_med - med) / (1.4826 * mad)  # approximate z using MAD
        
        # candidate columns for this band
        cand_cols = np.where(np.abs(zcols) > z_thresh)[0]
        if cand_cols.size == 0:
            continue
        
        # verify that most rows in that (col,bi) are extreme relative to that row's distribution
        # compute per-row z using row stats across columns
        row_med = np.median(band, axis=1)        # (r,)
        row_mad = np.median(np.abs(band - row_med[:, None]), axis=1) + 1e-9  # (r,)
        # z for each (row,col)
        z_rows_cols = (band - row_med[:, None]) / (1.4826 * row_mad[:, None])
        
        for col in cand_cols:
            # fraction of rows exceeding threshold at this (row,col)
            frac = np.mean(np.abs(z_rows_cols[:, col]) > z_thresh)
            if frac >= min_fraction:
                defect_mask[:, col, bi] = True
                defects_list.append((int(col), int(bi)))
    return defect_mask, defects_list


def identify_dead_pixels(
    cube: np.ndarray,
    threshold: float = 30,
    kernel_size: tuple[int, int] = (3, 5),
    visualize: bool = True,
    verbose: bool = False
) -> dict[str, object]:
    """
    Identifies dead pixel locations (column and band index) based on median filter deviation.

    Parameters
    ----------
    cube : np.ndarray
        HSI 3D array, expected shape (H, W, B)
    threshold : float
        Multiplier applied to the standard deviation of the local residual.
        Typical working range: ~20-60 for current setup.
    kernel_size : tuple[int, int]
        Shape of median filter kernel size
    visualize : bool
        Whether to visualize the plots for verification
    verbose : bool
        Whether to print band and column indices as a table

    Returns
    -------
    mask : np.ndarray
        Mask across W and B dimensions
    mask_3d : np.ndarray
        A defect 3D mask broadcasted along spatial (row) dimension
    band_idx : np.ndarray
        Array of band indices
    col_idx : np.ndarray
        Array of column indices
    residual : np.ndarray
        Residuals of mean profiles - local medians
    """
    mean_profiles = np.mean(cube, axis=0).T

    local_median = median_filter(mean_profiles, size=kernel_size)
    local_residual = mean_profiles - local_median
    
    std = np.std(local_residual)
    thresh = threshold * (std + 1e-8)
    spikes = np.abs(local_residual) > thresh # spikes shape (B, W)
    mask_wb = spikes.T # (W, B)

    band_idx, col_idx = np.where(spikes)

    defect_mask_3d = spikes.T[None, :, :]
    defect_mask_3d = np.broadcast_to(defect_mask_3d, cube.shape)
    
    if verbose:
        print('Band  |  Column')
        print('---------------')
        for b, c in zip(band_idx, col_idx):
            print(f"{b:5d} | {c:5d}")
    
    if visualize:
        fig, axs = plt.subplots(1,2, figsize=(15,5))
        
        img = np.mean(cube, axis=0)  # (cols, bands)
        
        axs[0].imshow(img, cmap='gray', aspect='auto')
        axs[0].set_xlabel('Band index')
        axs[0].set_ylabel('Column index')
        axs[0].set_title('Detected spikes')
        axs[0].scatter(band_idx, col_idx, facecolors='none', edgecolors='red', linewidths=0.8, s=50)
        axs[0].set_aspect('auto')
        
        n_bands = cube.shape[2]
        norm = Normalize(vmin=0, vmax=n_bands - 1)
        cmap = plt.get_cmap('Reds')
        for i, band in enumerate(range(n_bands)):
            axs[1].plot(np.mean(cube[:, :, band], axis=0), color=cmap(norm(i)))
        axs[1].set_ylabel('Intensity')
        axs[1].set_xlabel('Column index')
        axs[1].set_title('All bands')
        axs[1].minorticks_on()
        axs[1].grid(which='major', lw=0.8, ls='-')
        axs[1].grid(which='minor', lw=0.5, ls='--')
        
        sm = ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=axs[1])
        cbar.set_label('Band index')
        
        plt.tight_layout()
        plt.show()

    return {
    "mask": mask_wb,
    "mask_3d": defect_mask_3d,
    "band_idx": band_idx,
    "col_idx": col_idx,
    "residual": local_residual
    }



def interpolate_dead_pixels(
    cube: np.ndarray,
    mask_3d: np.ndarray,
    method: str = "spectral",   # "spectral", "spatial", "hybrid"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Repair dead pixels in a hyperspectral cube.

    Parameters
    ----------
    cube : np.ndarray
        Input HSI cube of shape (H, W, B)
    mask_3d : np.ndarray (bool)
        Boolean mask of defective pixels (same shape as cube)
    method : str
        "spectral" → interpolate along bands
        "spatial"  → interpolate along columns
        "hybrid"   → spectral first, fallback to spatial

    Returns
    -------
    cube_fixed : np.ndarray
        Repaired cube
    repaired_mask : np.ndarray
        Boolean mask of successfully repaired pixels (repaired pixel = True)
    """
    valid_methods = {"spectral", "spatial", "hybrid"}
    if method not in valid_methods:
        raise ValueError(f"Invalid method: {method}, supported: 'spectral', 'spatial' and 'hybrid'.")

    cube_fixed = cube.astype(float, copy=True)
    repaired_mask = np.zeros_like(mask_3d, dtype=bool)

    H, W, B = cube.shape

    # -----------------------
    # SPECTRAL INTERPOLATION
    # -----------------------
    if method in ["spectral", "hybrid"]:
        left = np.roll(cube, shift=1, axis=2)
        right = np.roll(cube, shift=-1, axis=2)

        left_mask = np.roll(mask_3d, shift=1, axis=2)
        right_mask = np.roll(mask_3d, shift=-1, axis=2)

        # invalidate edges
        left[:, :, 0] = np.nan
        right[:, :, -1] = np.nan
        left_mask[:, :, 0] = True
        right_mask[:, :, -1] = True

        # only use valid neighbors
        left_valid = np.where(~left_mask, left, np.nan)
        right_valid = np.where(~right_mask, right, np.nan)

        interp_spec = np.nanmedian(np.stack([left_valid, right_valid], axis=0), axis=0)

        valid_spec = ~np.isnan(interp_spec)
        apply_spec = mask_3d & valid_spec

        cube_fixed[apply_spec] = interp_spec[apply_spec]
        repaired_mask[apply_spec] = True

    # -----------------------
    # SPATIAL INTERPOLATION
    # -----------------------
    if method in ["spatial", "hybrid"]:
        left = np.roll(cube, shift=1, axis=1)
        right = np.roll(cube, shift=-1, axis=1)

        left_mask = np.roll(mask_3d, shift=1, axis=1)
        right_mask = np.roll(mask_3d, shift=-1, axis=1)

        # invalidate edges
        left[:, 0, :] = np.nan
        right[:, -1, :] = np.nan
        left_mask[:, 0, :] = True
        right_mask[:, -1, :] = True

        left_valid = np.where(~left_mask, left, np.nan)
        right_valid = np.where(~right_mask, right, np.nan)

        interp_spat = np.nanmedian(np.stack([left_valid, right_valid], axis=0), axis=0)

        valid_spat = ~np.isnan(interp_spat)

        if method == "spatial":
            apply_spat = mask_3d & valid_spat
        else:  # hybrid
            apply_spat = mask_3d & ~repaired_mask & valid_spat

        cube_fixed[apply_spat] = interp_spat[apply_spat]
        repaired_mask[apply_spat] = True

    return cube_fixed, repaired_mask


class DeadPixelProcessor:
    """
    Identifies dead pixels and interpolates them.

    Initialization parameters
    -------------------------
    threshold : int
    kernel_size : tuple[int,int]
    method : Literal['spectral', 'spatial', 'hybrid']
    visualize : bool
    verbose : bool

    Methods
    -------
    - identify
    - interpolate
    - clean

    Usage
    -----
    dpp = DeadPixelProcessor()  
    cube_fixed, repaired_mask = dpp.clean(cube)
    """
    def __init__(self, threshold=30, kernel_size=(3,5), method='spectral', visualize=False, verbose=False):
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.method = method
        self.visualize = visualize
        self.verbose = verbose

    def identify(self, cube):
        result = identify_dead_pixels(cube, self.threshold, self.kernel_size, self.visualize, self.verbose)
        self.last_result = result
        return result['mask_3d']

    def interpolate(self, cube, mask):
        cube_fixed, repaired_mask = interpolate_dead_pixels(cube, mask, method=self.method)
        return cube_fixed, repaired_mask

    def clean(self, cube):
        mask = self.identify(cube)
        return self.interpolate(cube, mask)


def plot_defect_summary(cube, defect_mask, defects_list=None, band_example=None):
    """
    Quick diagnostic plots:
      - image of defect locations summed across bands (shows vertical lines)
      - a plot of band_example slice (if given) with defect overlay
    """
    r, c, b = cube.shape
    summed = defect_mask.any(axis=2).astype(int)  # (r,c) True where any band defect
    plt.figure(figsize=(6,4))
    plt.imshow(summed, aspect='auto')
    plt.title('Pixels with any detected defect (summed over bands)')
    plt.colorbar(label='defect flag')
    plt.xlabel('column'); plt.ylabel('row')
    plt.show()
    
    if band_example is not None:
        plt.figure(figsize=(6,3))
        plt.imshow(cube[:, :, band_example], aspect='auto')
        plt.title(f'Band {band_example} (example) with defect overlay')
        # overlay defects at that band
        overlay = np.zeros((r,c,4))
        overlay[...,3] = defect_mask[:, :, band_example].astype(float) * 0.6  # alpha
        plt.imshow(overlay)
        plt.colorbar()
        plt.show()
    
    if defects_list is not None and len(defects_list) > 0:
        print('Detected (col, band) defects (sample):', defects_list[:40])
