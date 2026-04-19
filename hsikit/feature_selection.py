"""
Utilities for feature selection.

Note: This module is under active development and may change.
"""

import numpy as np
from numpy.typing import ArrayLike
import matplotlib.pyplot as plt
from matplotlib.figure import Figure, SubFigure
from matplotlib.axes import Axes
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from typing import Literal, Optional

# Based on binning
def adaptive_equalize_spectrum(
        intensities: ArrayLike,
        wavelengths: ArrayLike,
        n_bins: int = 10,
        method: Literal['intensity', 'count'] = 'intensity'
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Equalizes a spectrum by adaptively binning wavelengths.
    
    Parameters
    ----------
    intensities : array-like
        Intensity values of the spectrum.
    wavelengths : array-like
        Corresponding wavelength values.
    n_bins : int
        Number of bins to use.
    method : str
        - 'intensity' to equalize total intensity per bin,
        - 'count' to equalize number of points per bin.
    
    Returns:
    equalized_spectrum : np.ndarray
        Binned intensity values.
    bin_edges : np.ndarray
        Edges of the wavelength bins.
    """
    intensities = np.array(intensities)
    wavelengths = np.array(wavelengths)

    # Sort by wavelength
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    intensities = intensities[sort_idx]

    if method == 'intensity':
        # Cumulative intensity sum to define equal-intensity bins
        cumulative = np.cumsum(intensities)
        total_intensity = cumulative[-1]
        targets = np.linspace(0, total_intensity, n_bins + 1)
        bin_indices = np.searchsorted(cumulative, targets)
    elif method == 'count':
        # Equal number of samples per bin
        total_points = len(wavelengths)
        bin_indices = np.linspace(0, total_points, n_bins + 1, dtype=int)
    else:
        raise ValueError("Method must be 'intensity' or 'count'.")

    # Compute bin edges and equalized spectrum
    bin_edges = []
    equalized_spectrum = []

    for i in range(n_bins):
        start = bin_indices[i]
        end = bin_indices[i + 1]
        if end > start:
            bin_wavelengths = wavelengths[start:end]
            bin_intensities = intensities[start:end]
            bin_edges.append((bin_wavelengths[0], bin_wavelengths[-1]))
            equalized_spectrum.append(np.mean(bin_intensities))
        else:
            # If no data in bin (can happen with intensity method), skip
            continue

    # Convert bin_edges to array of edges
    bin_edges = np.array([edge[0] for edge in bin_edges] + [bin_edges[-1][1]])
    equalized_spectrum = np.array(equalized_spectrum)

    return equalized_spectrum, bin_edges


def adaptive_binning_by_gradient(
        intensities: ArrayLike,
        wavelengths: ArrayLike,
        n_bins: int = 20
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Adaptive binning based on spectral changes (derivative of the spectrum).
    
    Parameters
    ----------
    intensities : array-like
        Intensity values of the spectrum.
    wavelengths : array-like
        Corresponding wavelength values.
    n_bins : int
        Number of bins to produce.
        
    Returns
    -------
    binned_spectrum : np.ndarray
        Averaged intensity per bin.
    bin_edges : np.ndarray
        Wavelength edges of the bins.
    """
    intensities = np.array(intensities)
    wavelengths = np.array(wavelengths)

    # Sort the input by wavelength
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    intensities = intensities[sort_idx]

    # Compute normalized absolute gradient
    gradient = np.abs(np.gradient(intensities, wavelengths))
    weights = gradient / np.sum(gradient)  # Normalize to form a weight distribution

    # Compute cumulative sum of weights to define bin splits
    cumulative_weights = np.cumsum(weights)
    thresholds = np.linspace(0, 1, n_bins + 1)
    bin_edges_indices = np.searchsorted(cumulative_weights, thresholds)

    # Ensure valid bin indices
    bin_edges_indices[0] = 0
    bin_edges_indices[-1] = len(wavelengths) - 1

    # Extract bin edges and compute binned spectrum
    bin_edges = wavelengths[bin_edges_indices]
    binned_spectrum = []

    for i in range(n_bins):
        start = bin_edges_indices[i]
        end = bin_edges_indices[i + 1]
        if end > start:
            avg_intensity = np.mean(intensities[start:end])
            binned_spectrum.append(avg_intensity)

    return np.array(binned_spectrum), bin_edges

def compute_bin_centers(bin_edges: ArrayLike) -> np.ndarray:
    """
    Compute bin centers from bin edges.

    Parameters
    ----------
    bin_edges : array-like
        Array of bin edge values (length N+1 for N bins).

    Returns
    -------
    bin_centers : np.ndarray
        Array of bin centers (length N).
    """
    bin_edges = np.array(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers

def compute_binned_stats(
        intensities: ArrayLike,
        wavelengths: ArrayLike,
        bin_edges: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and standard deviation of intensities within each bin.
    
    Parameters
    ----------
    intensities : array-like
        Intensity values of the spectrum.
    wavelengths : array-like
        Corresponding wavelength values.
    bin_edges : array-like
        Edges of the wavelength bins (length N+1).
    
    Returns
    -------
    means : np.ndarray
        Mean intensity in each bin.
    stds : np.ndarray
        Standard deviation of intensity in each bin.
    """
    intensities = np.array(intensities)
    wavelengths = np.array(wavelengths)
    bin_edges = np.array(bin_edges)

    # Sort by wavelength
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    intensities = intensities[sort_idx]

    means = []
    stds = []

    for i in range(len(bin_edges) - 1):
        start_edge = bin_edges[i]
        end_edge = bin_edges[i + 1]
        mask = (wavelengths >= start_edge) & (wavelengths < end_edge)

        bin_intensities = intensities[mask]
        if len(bin_intensities) > 0:
            means.append(np.mean(bin_intensities))
            stds.append(np.std(bin_intensities))
        else:
            means.append(np.nan)
            stds.append(np.nan)

    return np.array(means), np.array(stds)

def plot_spectrum_with_bins(
        wavelengths: ArrayLike,
        intensities: ArrayLike,
        bin_edges: ArrayLike,
        title: str,
        show_centers: bool = False,
        ax: Optional[Axes] = None
    ) -> tuple[Figure | SubFigure, Axes]:
    """
    Plot the original spectrum and vertical lines for bin edges.

    Parameters:
        wavelengths (array-like): Wavelength values of the spectrum.
        intensities (array-like): Intensity values of the spectrum.
        bin_edges (array-like): Bin edge wavelengths.
        show_centers (bool): Whether to also show bin centers.
    """
    wavelengths = np.array(wavelengths)
    intensities = np.array(intensities)
    bin_edges = np.array(bin_edges)

    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    ax.plot(wavelengths, intensities, label='Original Spectrum', color='blue')

    for edge in bin_edges:
        ax.axvline(x=edge, color='red', linestyle='--', linewidth=1)

    if show_centers:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for center in bin_centers:
            ax.axvline(x=center, color='green', linestyle=':', linewidth=1)
        ax.legend(['Spectrum', 'Bin Edges', 'Bin Centers'])
    else:
        ax.legend(['Spectrum', 'Bin Edges'])

    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_title(title)
    ax.grid(True)

    return fig, ax

# Based on CARS

def rmse_cv(
        X: np.ndarray,
        y: np.ndarray,
        n_components: int = 10,
        n_splits: int = 5
    ) -> float:
    """
    Compute cross-validated Root Mean Squared Error (RMSE) using PLS regression.

    This function performs K-fold cross-validation with Partial Least Squares (PLS)
    regression and returns the mean RMSE across folds.

    Parameters
    ----------
    X : np.ndarray
        HSI 3D array, expected shape (n_samples, B)
    y : np.ndarray
        Labels vector, expected shape (n_samples,)
    n_components : int
        Number of PLS components to use in the regression model.
    n_splits : int
        Number of folds for K-fold cross-validation.
    
    Returns
    -------
    float
        Mean RMSE across all cross-validation folds.

    Notes
    -----
    - The data is shuffled before splitting using a fixed random state (42).
    - RMSE is computed as the square root of the mean squared error (MSE).
    - This function is typically used as an evaluation metric within
      iterative feature selection algorithms such as CARS.

    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses = []

    for train, test in kf.split(X):
        pls = PLSRegression(n_components=n_components)
        pls.fit(X[train], y[train])
        y_pred = pls.predict(X[test]).ravel()
        rmses.append(np.sqrt(mean_squared_error(y[test], y_pred)))

    return float(np.mean(rmses))

def CARS(X: np.ndarray,
         y: np.ndarray,
         n_components: int = 10,
         n_mc: int = 50,
         sample_ratio: float = 0.9,
         cv_splits: int = 5,
         random_state: int = 42,
         return_all: bool = True
    ) -> tuple[np.ndarray, list[float]] | tuple[np.ndarray, list[float], list[np.ndarray]]:
    """
    Competitive Adaptive Reweighted Sampling (CARS) for feature selection
    using Partial Least Squares (PLS) regression.

    CARS is an iterative variable selection method that combines Monte Carlo
    sampling with PLS regression coefficients to progressively eliminate
    less informative variables. At each iteration, variables are ranked by
    importance and a decreasing subset is retained based on an exponential
    decay schedule. Model performance is evaluated using cross-validated RMSE.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Predictor matrix.
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Response vector or matrix.
    n_components : int, default=10
        Maximum number of PLS components.
    n_mc : int, default=50
        Number of Monte Carlo sampling iterations.
    sample_ratio : float, default=0.9
        Fraction of samples to use in each Monte Carlo iteration.
    cv_splits : int, default=5
        Number of folds for cross-validation when evaluating subsets.
    random_state : int, default=42
        Seed for reproducibility.
    return_all : bool, default=True
        If True, return RMSE trajectory and variable subsets for all iterations.

    Returns
    -------
    best_vars : ndarray
        Indices of the selected variables corresponding to the lowest RMSE.
    rmse_list : list of float
        RMSE values at each iteration.
    var_list : list of ndarray, optional
        List of variable index subsets at each iteration. Returned only if
        `return_all=True`.

    Notes
    -----
    - Variable importance is estimated from absolute PLS regression coefficients.
    - The number of retained variables decreases exponentially over iterations.
    - The final subset is chosen based on minimum cross-validated RMSE.
    - Ensures that the number of variables is always sufficient for the chosen
      number of PLS components.
    """
    rng = np.random.default_rng(random_state)

    n_samples, n_vars = X.shape
    variables = np.arange(n_vars)

    rmse_list = []
    var_list = []

    decay = np.log(n_vars) / n_mc

    for i in range(n_mc):
        # Monte Carlo sampling of samples
        idx = rng.choice(n_samples, size=int(sample_ratio * n_samples), replace=False)
        X_mc = X[idx][:, variables]
        y_mc = y[idx]

        # ---- PLS
        n_comp = min(n_components, X_mc.shape[1] - 1)
        pls = PLSRegression(n_components=n_comp)
        pls.fit(X_mc, y_mc)

        # ---- Variable importance
        coef = np.abs(pls.coef_).flatten()

        # ---- Number of variables to keep
        k = int(n_vars * np.exp(-decay * i))
        k = max(k, n_comp + 1)

        # ---- Keep top-k variables
        keep_idx = np.argsort(coef)[-k:]
        variables = variables[keep_idx]

        # ---- Evaluate subset
        rmse = rmse_cv(X[:, variables], y, n_components=min(n_components, len(variables) - 1), n_splits=cv_splits)

        rmse_list.append(rmse)
        var_list.append(variables.copy())

    # ---- Best subset
    best_iter = np.argmin(rmse_list)
    best_vars = var_list[best_iter]

    if return_all:
        return best_vars, rmse_list, var_list
    else:
        return best_vars, rmse_list