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
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import chi2
from scipy import sparse

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
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)

    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D @ D.T
        z = sparse.linalg.spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return z


# ----------------------------------------------------------------- FEATURE-SELECTION -----------------------------------------------
# Based on binning
def plot_spectrum_with_bins(wavelengths, intensities, bin_edges, title, show_centers=False, savefig=False):
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

    plt.figure(figsize=(10, 5))
    plt.plot(wavelengths, intensities, label='Original Spectrum', color='blue')

    for edge in bin_edges:
        plt.axvline(x=edge, color='red', linestyle='--', linewidth=1)

    if show_centers:
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for center in bin_centers:
            plt.axvline(x=center, color='green', linestyle=':', linewidth=1)
        plt.legend(['Spectrum', 'Bin Edges', 'Bin Centers'])
    else:
        plt.legend(['Spekter', 'Robovi razredov'])

    plt.xlabel('Valovna dolžina (nm)')
    plt.ylabel('Intenziteta')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    if savefig:
        plt.savefig(title+'.png', dpi=300)
    plt.show()


def adaptive_equalize_spectrum(intensities, wavelengths, n_bins=10, method='intensity'):
    """
    Equalizes a spectrum by adaptively binning wavelengths.
    
    Parameters:
        intensities (array-like): Intensity values of the spectrum.
        wavelengths (array-like): Corresponding wavelength values.
        n_bins (int): Number of bins to use.
        method (str): 'intensity' to equalize total intensity per bin,
                      'count' to equalize number of points per bin.
    
    Returns:
        equalized_spectrum (np.ndarray): Binned intensity values.
        bin_edges (np.ndarray): Edges of the wavelength bins.
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

def compute_bin_centers(bin_edges):
    """
    Compute bin centers from bin edges.

    Parameters:
        bin_edges (array-like): Array of bin edge values (length N+1 for N bins).

    Returns:
        bin_centers (np.ndarray): Array of bin centers (length N).
    """
    bin_edges = np.array(bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers

def adaptive_binning_by_gradient(intensities, wavelengths, n_bins=20):
    """
    Adaptive binning based on spectral changes (derivative of the spectrum).
    
    Parameters:
        intensities (array-like): Intensity values of the spectrum.
        wavelengths (array-like): Corresponding wavelength values.
        n_bins (int): Number of bins to produce.
        
    Returns:
        binned_spectrum (np.ndarray): Averaged intensity per bin.
        bin_edges (np.ndarray): Wavelength edges of the bins.
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


def compute_binned_stats(intensities, wavelengths, bin_edges):
    """
    Compute mean and standard deviation of intensities within each bin.
    
    Parameters:
        intensities (array-like): Intensity values of the spectrum.
        wavelengths (array-like): Corresponding wavelength values.
        bin_edges (array-like): Edges of the wavelength bins (length N+1).
    
    Returns:
        means (np.ndarray): Mean intensity in each bin.
        stds (np.ndarray): Standard deviation of intensity in each bin.
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


# Based on CARS

def rmse_cv(X, y, n_components=10, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses = []

    def mean_squared_error():
        pass

    for train, test in kf.split(X):
        pls = PLSRegression(n_components=n_components)
        pls.fit(X[train], y[train])
        y_pred = pls.predict(X[test]).ravel()
        rmses.append(np.sqrt(mean_squared_error(y[test], y_pred)))

    return np.mean(rmses)

def CARS(X, y, n_components=10, n_mc=50, sample_ratio=0.9, cv_splits=5, random_state=42, return_all=True):
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
        coef = np.abs(pls.coef_).ravel()

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

        self._classes = self._ohencoder.categories_[0]
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