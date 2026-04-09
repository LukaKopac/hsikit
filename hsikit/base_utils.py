"""
Utilities for extracting spectra from hyperspectral data.

This module provides helper functions for extracting spectra - static grid-based and interactive.

Note: This module is under active development and may change.
"""

import numpy as np

# Block average a cube
def block_average_cube(cube: np.ndarray, block_size: int = 5) -> np.ndarray:
    """
    Subsamples / block averages a cube.

    Parameters
    ----------
    cube : np.ndarray
        HSI 3D array, expected shape (H, W, B)
    block_size : int
        Size of block to average

    Returns
    -------
    np.ndarray
        Subsampled cube, shape (H/block_size, W/block_size, B)
    """
    H, W, B = cube.shape

    H_crop = H - (H % block_size)
    W_crop = W - (W % block_size)
    cube_cropped = cube[:H_crop, :W_crop, :]

    h_blocks = H_crop // block_size
    w_blocks = W_crop // block_size
    cube_blocks = cube_cropped.reshape(h_blocks, block_size, w_blocks, block_size, B)
    
    averaged = cube_blocks.mean(axis=(1, 3))

    return averaged


# Convert dictionary to X, y arrays
def dict2Xy(sample_dictionary: dict[str, list[np.ndarray] | np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """ 
    Converts a dictionary of [key:np.ndarray] or [key:list[np.ndarray]] to X (n_samples, B) and y (n_samples,) arrays.
    
    Parameters
    ----------
    sample_dictionary : dict
    
    Returns
    -------
    X : np.ndarray
    y : np.ndarray
    """
    
    X = []
    y = []
    feature_dim = None

    if not sample_dictionary:
        raise ValueError("Input dictionary is empty")

    for label, data in sample_dictionary.items():
        cubes = data if isinstance(data, list) else [data]

        for c in cubes:
            if not isinstance(c, np.ndarray):
                raise ValueError("Expected np.ndarray or list of np.ndarray")

            if c.ndim != 3:
                raise ValueError("Each array must be 3D")

            flattened = c.reshape(-1, c.shape[-1])

            if feature_dim is None:
                feature_dim = flattened.shape[1]
            elif flattened.shape[1] != feature_dim:
                raise ValueError("Inconsistent feature dimensions")

            X.append(flattened)
            y.extend([label] * flattened.shape[0])

    return np.vstack(X), np.array(y)

# Signal to noise ratio per band
def compute_snr_per_band(cube: np.ndarray, mask: None | np.ndarray = None) -> np.ndarray:
    """
    Compute SNR (signal-to-noise ratio) per band after optionally masking bad pixels.
    
    Parameters
    ----------
    cube : np.ndarray
        shape (H, W, B)
    mask : np.ndarray
        Boolean array (H, W), True = good pixel
    
    Returns
    -------
    np.ndarray
        Signal to noise ratio per band, length B
    """
    if mask is None:
        mask = np.ones_like(cube.shape[:2], dtype=bool)
    
    masked_cube = cube[mask] # shape (n_pixels, B)

    mean = masked_cube.mean(axis=0)
    std = masked_cube.std(axis=0)
    
    return mean / (std + 1e-8)


# VARIANCE RATIO - BETWEEN VS WITHIN CLASSES
def class_variance_ratio(X: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    Computes between classes to within classes variance ratio.

    Parameters
    ----------
    X : np.ndarray
        Samples spectra, expected shape (n_samples, B)
    y : np.ndarray
        Samples labels, expected shape (n_samples)

    Returns
    -------
    within_var : float
        Variance within individual classes
    between_var : float
        Variance between individual classes
    ratio : float
        Variance ratio: between_var / within_var
    """
    classes = np.unique(y)
    n_features = X.shape[1]
    mu_global = X.mean(axis=0)
    
    S_W = np.zeros((n_features, n_features))
    S_B = np.zeros((n_features, n_features))
    
    for cls in classes:
        X_i = X[y == cls]
        mu_i = X_i.mean(axis=0)
        S_W += (X_i - mu_i).T @ (X_i - mu_i)
        S_B += X_i.shape[0] * np.outer(mu_i - mu_global, mu_i - mu_global)
    
    within_var = np.trace(S_W)
    between_var = np.trace(S_B)
    
    ratio = between_var / within_var
    return within_var, between_var, ratio