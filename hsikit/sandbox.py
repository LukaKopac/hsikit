"""
Interesting functions/ideas - copied from various projects.
Non-sorted, not reviewed, can be non-functional
Good ideas to implement:
---------------------------
- Detect and repair dead pixels and outliers

- Distances between spectra
    - euclidian, manhattan, cosine dissimilarity, SAM, correlation distance, Mahalanobis
    - Pairwise spectral distances (between samples of different wood species)
    - distance matrix

- Spectral similarity network based on distance matrix
    - networkx package/library

- Isomap, k-means, kNN, t-sne, MDS (can be based on distance matrix)

- PCA density scatter (hexbin), PCA RGB image

- HSI cube using plotly (faster for data exploration)

- Manual preprocessing/feature selection - based on bin edges

- Robust SNV

- Visualizations using actual wavelengths instead of bands

- PLS-DA, soft PLS-DA

- MNF

- CARS feature selection

- Global normalization for visualization (image plotting) functions
...

"""

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import chi2
from scipy import sparse

from typing import Literal

# ----------------------------------------------------------------- FEATURE SELECTION -------------------------------------

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
    
    Returns
    -------
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


# ----------------------------------------------------------------- CLEANING / EXPLORATORY -----------------------------------------------

# Reflectance to absorbance - Lambert-Beer or Kubelka-Munk
def reflectance_to_absorbance(R, method='lambert'):
    """
    Convert reflectance spectra to absorbance (Lambert-Beer) or Kubelka-Munk units.

    Parameters
    ----------
    R : array-like
        Reflectance values (0 < R <= 1)
    method : str
        - 'lambert' for Lambert-Beer
        - 'kubelka' for Kubelka-Munk

    Returns
    -------
    absorbance : array-like
        Transformed values
    """
    R = np.array(R)
    
    if np.any((R <= 0) | (R > 1)):
        raise ValueError("Reflectance values must be between 0 and 1 (exclusive 0).")
    
    if method == 'lambert':
        return -np.log10(R)
    elif method == 'kubelka':
        return (1 - R)**2 / (2 * R)
    else:
        raise ValueError("Method must be 'lambert' or 'kubelka'.")


# Baseline correction
def asls_baseline(y, lam=1e6, p=0.001, niter=10):
    """
    Asymmetric Least Squares baseline correction.

    Parameters
    ----------
    y : 1D array
        Spectrum (absorbance)
    lam : float
        Smoothness parameter (higher = smoother baseline)
    p : float
        Asymmetry (small = baseline under peaks)
    niter : int
        Number of iterations
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2)) # type: ignore[arg-type]
    w = np.ones(L)
    y = np.asarray(y)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D @ D.T
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z

# ----------------------------------------------------------------- PREPROCESSING -----------------------------------------------

def robust_snv(X):
    median = np.nanmedian(X, axis=1, keepdims=True)
    iqr = np.nanpercentile(X, 75, axis=1, keepdims=True) - np.nanpercentile(X, 25, axis=1, keepdims=True)
    iqr[iqr == 0] = 1e-6 # Avoid division by zero
    return (X - median) / iqr

def rnv(X):
    medians = np.median(X, axis=1, keepdims=True)
    mad = np.median(np.abs(X - medians), axis=1, keepdims=True)
    mad[mad == 0] = 1e-6
    return (X - medians) / mad


# ----------------------------------------------------------------- PLS-DA -----------------------------------------------

class SoftPLSDA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2, alpha=0.05, gamma=0.05):
        self.n_components = n_components
        self.alpha = alpha
        self.gamma = gamma

    def fit(self, X, y):
        # One-hot encode y
        self._ohencoder = OneHotEncoder(sparse_output=False)
        y_ohe = self._ohencoder.fit_transform(y.reshape(-1, 1))

        self._classes = np.asarray(self._ohencoder.categories_[0])
        n_classes = len(self._classes)

        # Class masks
        self._class_mask = np.stack([y == cls for cls in self._classes], axis=1)  # shape (n_samples, n_classes)

        # Scale X
        self._x_scaler = StandardScaler(with_mean=True, with_std=True)
        X_scaled = self._x_scaler.fit_transform(X)

        # Center Y only (no scaling)
        self._y_scaler = StandardScaler(with_mean=True, with_std=False)
        y_scaled = self._y_scaler.fit_transform(y_ohe)

        # Fit PLS regression
        self._pls = PLSRegression(n_components=self.n_components, max_iter=5000, tol=1.0e-9, scale=False)
        self._pls.fit(X_scaled, y_scaled)

        # Predict Y_hat for training
        y_hat_train = self._y_scaler.inverse_transform(self._pls.predict(X_scaled))

        # PCA on Y_hat
        self._pca = PCA(n_components=n_classes - 1, random_state=0)
        self._T_train = self._pca.fit_transform(y_hat_train)
        self._class_centers = self._pca.transform(np.eye(n_classes))  # class centers

        # Compute within-class scatter matrices (vectorized)
        self._S = np.zeros((n_classes, self._T_train.shape[1], self._T_train.shape[1]))
        epsilon = 1e-6
        for i in range(n_classes):
            diffs = self._T_train[self._class_mask[:, i]] - self._class_centers[i]
            self._S[i] = (diffs.T @ diffs) / np.sum(self._class_mask[:, i])

        # Precompute inverse scatter matrices for Mahalanobis
        self._S_inv = np.array([
            np.linalg.pinv(self._S[i] + epsilon * np.eye(self._S[i].shape[0]))
            for i in range(n_classes)
        ])

        # Critical Mahalanobis distance for class assignment
        self._d2_crit = chi2.ppf(1.0 - self.alpha, n_classes - 1)
        self._d2_out = np.array([
            chi2.ppf((1.0 - self.gamma) ** (1.0 / np.sum(self._class_mask[:, i])), n_classes - 1)
            for i in range(n_classes)
        ])

        # Training outliers (vectorized)
        diffs_all = self._T_train[:, np.newaxis, :] - self._class_centers[np.newaxis, :, :]  # (n_samples, n_classes, n_features)
        # Correct einsum: 'sci,cij->sc' where s=sample, c=class, i,j=features
        self._d2_train = np.einsum('sci,cij,scj->sc', diffs_all, self._S_inv, diffs_all)
        self._outliers_train = np.array([
            self._d2_train[j, i] > self._d2_out[i]
            for j in range(X.shape[0])
            for i in range(n_classes)
            if self._class_mask[j, i]
        ])
        return self

    def predict(self, X):
        # Scale X
        X_scaled = self._x_scaler.transform(X)
        y_hat_test = self._y_scaler.inverse_transform(self._pls.predict(X_scaled))
        T_test = self._pca.transform(y_hat_test)

        # Vectorized Mahalanobis distances
        diffs = T_test[:, np.newaxis, :] - self._class_centers[np.newaxis, :, :]  # (n_samples, n_classes, n_features)
        d2 = np.einsum('sci,cij,scj->sc', diffs, self._S_inv, diffs)  # (n_samples, n_classes)

        # Assign classes within critical distance
        predictions = [
            [self._classes[i] for i, dist in enumerate(row) if dist < self._d2_crit] or ["NOT_ASSIGNED"]
            for row in d2
        ]
        return predictions

    def get_outliers_train(self):
        return np.array(self._outliers_train)

# ----------------------------------------------------------------- MNF -----------------------------------------------

class MNF:
    def __init__(self, n_components=None):
        self.n_components = n_components

    def _estimate_noise(self, cube):
        "Estimate noise via spatial differences."
        H, W, B = cube.shape

        diff_h = cube[:, 1:, :] - cube[:, :-1, :]
        diff_v = cube[1:, :, :] - cube[:-1, :, :]

        noise = np.concatenate([
            diff_h.reshape(-1, B),
            diff_v.reshape(-1, B)
        ], axis=0)

        return noise

    def fit(self, cube, mask=None):
        """
        Fit MNF transform.

        Parameters
        ----------
        cube : np.ndarray
            HSI 3D array, expected shape (H, W, B)
        mask : np.ndarray
            Optional boolean mask of valid pixels, expected shape (H, W).
        """

        H, W, B = cube.shape
        X = cube.reshape(-1, B)

        if mask is not None:
            X = X[mask.reshape(-1)]

        # mean center
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        # noise covariance
        noise = self._estimate_noise(cube)
        Cn = np.cov(noise, rowvar=False)
    
        # Eigen-decomposition of noise covariance
        Ln, En = np.linalg.eigh(Cn)
    
        # Whitening matrix and whiten data
        self.whitening_ = En @ np.diag(1.0 / np.sqrt(Ln + 1e-12)) @ En.T
        Xw = Xc @ self.whitening_

        # Covariance of whitened data
        Cx = np.cov(Xw, rowvar=False)

        # PCA on whitened data
        L, E = np.linalg.eigh(Cx)

        # Sort descending
        idx = np.argsort(L)[::-1]
        self.eigenvalues_ = L[idx]
        self.components_ = E[:, idx]

        if self.n_components is not None:
            self.components_ = self.components_[:, :self.n_components]

        return self

    def transform(self, cube):
        "Apply MNF transform to cube."

        H, W, B = cube.shape

        X = cube.reshape(-1, B)

        Xc = X - self.mean_

        Xw = Xc @ self.whitening_

        Xmnf = Xw @ self.components_

        return Xmnf.reshape(H, W, -1)

    def fit_transform(self, cube, mask=None):
        "Fit MNF and transform cube"
        self.fit(cube, mask)
        return self.transform(cube)