"""
General utility functions:
- block_average_cube (subsample HSI cube, aggregates individual spatial pixels)
- dict2Xy (converts a dictionary to X matrix and y vector)
- snr_per_band (signal to noise ratio per band)
- class_variance_ratio (between classes to within classes variance ratio)
- spectral_outlier_analysis (Mahalanobis distance VS Orthogonal distance for outlier detection)

Note: This module is under active development and may change.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.covariance import EmpiricalCovariance

# Block average a cube
def block_average_cube(cube: np.ndarray, block_size: int = 5) -> np.ndarray:
    """
    Subsamples / block averages a cube.  
    The cube is cropped in the process to enforce 'block_size' parameter (if cube shape not divisible by it).

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
    cube_cropped = cube[:H_crop, :W_crop, :] # crop cube to enforce division by block_size

    h_blocks = H_crop // block_size
    w_blocks = W_crop // block_size # number of expected blocks in height and width
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
        Structured either [key:np.ndarray] or [key:list[np.ndarray]]
    
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
def snr_per_band(cube: np.ndarray, mask: None | np.ndarray = None) -> np.ndarray:
    """
    Compute SNR (signal-to-noise ratio) per band after optionally masking bad pixels.
    SNR = mean / std
    
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
    Computes between classes variance, within classes variance and their ratio.
    High ratio means:
    - samples within each class are tightly clustered
    - different classes are far apart

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
        S_W += (X_i - mu_i).T @ (X_i - mu_i) # scatter matrices
        S_B += X_i.shape[0] * np.outer(mu_i - mu_global, mu_i - mu_global)
    
    within_var = np.trace(S_W)
    between_var = np.trace(S_B)
    
    ratio = between_var / within_var
    return within_var, between_var, ratio

# Spectral outliers: Mahalanobis distance VS Orthogonal distance
def spectral_outlier_analysis(
    X,
    sample_labels,
    n_components=10,
    md_threshold=None,
    od_threshold=None,
    md_percentile=99.5,
    od_percentile=99.5,
    figsize=(8, 6),
    alpha=0.4,
):
    """
    PCA-based spectral outlier detection using Mahalanobis distance (MD)
    and orthogonal distance (OD).

    Parameters
    ----------
    X : ndarray, shape (n_spectra, n_bands)
        Spectral data.

    sample_labels : array-like, shape (n_spectra,)
        Sample/individual labels for each spectrum.

    n_components : int, default=10
        Number of PCA components used.

    md_threshold : float or None
        Explicit Mahalanobis-distance threshold.
        If None, md_percentile is used.

    od_threshold : float or None
        Explicit orthogonal-distance threshold.
        If None, od_percentile is used.

    md_percentile : float, default=99.5
        Percentile used to determine MD threshold if md_threshold is None.

    od_percentile : float, default=99.5
        Percentile used to determine OD threshold if od_threshold is None.

    figsize : tuple, default=(8, 6)
        Figure size.

    alpha : float, default=0.4
        Scatter point transparency.

    Returns
    -------
    results : dict
        Dictionary containing PCA model, distances, thresholds,
        and outlier masks.
    fig : matplotlib.figure.Figure
        Figure containing the MD vs OD plot.
    ax : matplotlib.axes.Axes
        Plot axes.
    """

    X = np.asarray(X)
    sample_labels = np.asarray(sample_labels)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_spectra, n_bands).")

    if len(X) != len(sample_labels):
        raise ValueError("X and sample_labels must contain the same number of spectra.")

    # ------------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------------

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)

    # ------------------------------------------------------------------
    # Mahalanobis distance in PCA score space
    # ------------------------------------------------------------------

    covariance = EmpiricalCovariance().fit(scores)

    md = np.sqrt(covariance.mahalanobis(scores))

    # ------------------------------------------------------------------
    # Orthogonal distance
    #
    # Reconstruct spectra from the retained PCA components and calculate
    # the Euclidean distance between the original and reconstructed spectra.
    # ------------------------------------------------------------------

    X_reconstructed = pca.inverse_transform(scores)

    residuals = X - X_reconstructed

    od = np.linalg.norm(residuals, axis=1)

    # ------------------------------------------------------------------
    # Determine thresholds
    # ------------------------------------------------------------------

    if md_threshold is None:
        md_threshold = np.percentile(md, md_percentile)

    if od_threshold is None:
        od_threshold = np.percentile(od, od_percentile)

    # ------------------------------------------------------------------
    # Outlier masks
    # ------------------------------------------------------------------

    md_outlier = md > md_threshold
    od_outlier = od > od_threshold

    # Either distance indicates an outlier
    outlier = md_outlier | od_outlier

    # Both distances indicate an outlier
    joint_outlier = md_outlier & od_outlier

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=figsize)

    unique_labels = np.unique(sample_labels)

    for label in unique_labels:
        idx = sample_labels == label

        ax.scatter(
            md[idx],
            od[idx],
            alpha=alpha,
            s=15,
            label=str(label),
        )

    # Thresholds
    ax.axvline(
        md_threshold,
        linestyle="--",
        linewidth=1.2,
        color="black",
    )

    ax.axhline(
        od_threshold,
        linestyle="--",
        linewidth=1.2,
        color="black",
    )

    ax.set_xlabel("Mahalanobis distance")
    ax.set_ylabel("Orthogonal distance")

    ax.set_title("Spectral outlier detection")

    ax.legend(
        title="Sample",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    results = {
        "pca": pca,
        "scores": scores,
        "reconstructed": X_reconstructed,
        "mahalanobis": md,
        "orthogonal": od,
        "md_threshold": md_threshold,
        "od_threshold": od_threshold,
        "md_outlier": md_outlier,
        "od_outlier": od_outlier,
        "outlier": outlier,
        "joint_outlier": joint_outlier,
        "sample_labels": sample_labels,
    }

    return results, fig, ax