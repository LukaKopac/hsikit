import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy.ndimage import median_filter

def identify_dead_pixels(
    cube: np.ndarray,
    threshold: float = 30,
    kernel_size: tuple[int, int] = (3, 5),
    visualize: bool = True,
    verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identifies dead pixel locations (column and band index) based on median filter deviation.

    Parameters
    ----------
    cube : np.ndarray
        HSI 3D array, expected shape (H, W, B)
    threshold : float
        Multiplier applied to the standard deviation of the local residual.
        Typical working range: ~20–60 for current setup.
    kernel_size : tuple[int, int]
        Shape of median filter kernel size
    visualize : bool
        Whether to visualize the plots for verification
    verbose : bool
        Whether to print band and column indices as a table

    Returns
    -------
    band_idx : np.ndarray
        Array of band indices
    col_idx : np.ndarray
        Array of column indices
    """
    mean_profiles = np.mean(cube, axis=0).T

    local_median = median_filter(mean_profiles, size=kernel_size)
    local_residual = mean_profiles - local_median
    
    thresh = threshold * np.std(local_residual)
    spikes = np.abs(local_residual) > thresh # spikes shape (B, W)
    mask_wb = spikes.T # (W, B)

    band_idx, col_idx = np.where(spikes)

    mask_3d = spikes.T[None, :, :]
    mask_3d = np.broadcast_to(mask_3d, cube.shape)
    
    if verbose:
        print('Band  |  Column')
        print('---------------')
        for b, c in zip(band_idx, col_idx):
            print(f"{b}   |   {c}")
            print('---------------')
    
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
    "mask_3d": mask_3d,
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

    cube = cube.astype(float)
    cube_fixed = cube.copy()
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
    def __init__(self, threshold=5):
        ...

    def identify(self, cube):
        return mask

    def interpolate(self, cube, mask):
        ...

    def clean(self, cube, strategy="spectral"):
        mask = self.identify(cube)
        return self.interpolate(cube, mask)