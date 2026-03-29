"""
Utilities for preprocessing hyperspectral data.

This module provides helper functions for normalizing HSI cubes and common preprocessing - SNV, MSC, SG derivatives.

Note: This module is under active development and may change.
"""

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from scipy.signal import savgol_filter

# Normalization

def normalize_min_max(
    cube: NDArray,
    return_params: bool = False
) -> NDArray | tuple[NDArray, NDArray, NDArray]:
    """
    Min-max normalizes a hypercube.


    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B)
    return_params : bool
        Whether to return min and max values per band.

    Returns
    -------
    NDArray or tuple[NDArray, NDArray, NDArray]
        If return_params is False, returns normalized cube (H, W, B).
        If True, returns a tuple:
            (normalized_cube, min_vals, max_vals),
            where min_vals and max_vals are NDArray of shape (B,)
    """
    h, w, b = cube.shape
    flat_cube = cube.reshape(-1, b)
    min_vals = flat_cube.min(axis=0)
    max_vals = flat_cube.max(axis=0)
    norm_X = (flat_cube - min_vals) / (max_vals - min_vals + 1e-8)
    norm_cube = norm_X.reshape(h, w, b)

    if return_params:
        return norm_cube, min_vals, max_vals
    else:
        return norm_cube

def normalize_mean_std(cube: NDArray, return_params: bool = False) -> NDArray | tuple[NDArray, NDArray, NDArray]:
    """
    Standardizes a hypercube using mean and std.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B)
    return_params : bool
        Whether to return mean and std values per band.

    Returns
    -------
    NDArray or tuple[NDArray, NDArray, NDArray]
        If return_params is False, returns standardized cube (H, W, B).
        If True, returns a tuple:
            (standardized_cube, mean_vals, std_vals),
            where mean_vals and std_vals are NDArray of shape (B,).
    """
    h, w, b = cube.shape
    X = cube.reshape(-1, b)
    mean_vals = X.mean(axis=0)
    std_vals = X.std(axis=0) + 1e-8
    norm_X = (X - mean_vals) / std_vals
    norm_cube = norm_X.reshape(h, w, b)
    if return_params:
        return norm_cube, mean_vals, std_vals
    else:
        return norm_cube

# Standard preprocessing classes (SNV, MSC, SG derivatives)

class SNV(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate (SNV) preprocessing class.
    Each spectrum is centered and scaled individually.
    """
    def fit(self, X, y=None):
        X = check_array(X)
        return self
    
    def transform(self, X, y=None):
        X = check_array(X)
        X = X.copy()

        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)

        stds[stds == 0] = 1e-12

        Xsnv = (X - means) / stds
        return Xsnv

class MSC(BaseEstimator, TransformerMixin):
    """
    Multiplicative Scatter Correction (MSC) preprocessing class.
    Mean spectrum as reference if None.
    """
    def __init__(self, reference=None):
        self.reference = reference

    def fit(self, X, y=None):

        X = check_array(X)
        X = X.copy()

        # Mean centering
        X_centered = X - X.mean(axis=1, keepdims=True)

        # Set reference
        if self.reference is None:
            self.reference_ = X_centered.mean(axis=0)
        else:
            self.reference_ = np.asarray(self.reference)
            if self.reference_.shape[0] != X.shape[1]:
                raise ValueError("reference spectrum must match number of features")

        return self
    
    def transform(self, X, y=None):
        check_is_fitted(self, "reference_")

        X = check_array(X)
        X = X.copy()

        # Mean centering
        X_centered = X - X.mean(axis=1, keepdims=True)

        r = self.reference_

        # Center reference
        r_mean = r.mean()
        r_centered = r - r_mean

        # Precompute denominator (scalar)
        denom = np.dot(r_centered, r_centered)

        # Compute slopes (vector of shape[n_samples])
        slopes = (X_centered @ r_centered) / denom
        slopes[slopes == 0] = 1e-12

        # Compute intercepts
        X_means = X_centered.mean(axis=1)
        intercepts = X_means - slopes * r_mean

        # Apply correction
        Xmsc = (X_centered - intercepts[:, np.newaxis]) / slopes[:, np.newaxis]

        return Xmsc
    
class SavitzkyGolay(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay preprocessing class.
    Define window_length, polyorder and deriv when initializing.
    """
    def __init__(self, window_length=11, polyorder=2, deriv=0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        X = check_array(X)

        # Basic parameter validation
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        
        if self.window_length <= self.polyorder:
            raise ValueError("window_length must be > polyorder")
        
        return self
    
    def transform(self, X, y=None):
        X = check_array(X)

        # Apply filter along feature axis
        Xsg = savgol_filter(X, window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv, axis=1)

        return Xsg