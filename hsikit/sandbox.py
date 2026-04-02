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

- Subsampling/block-average 5x5, 10x10...

- Visualizations using actual wavelengths instead of bands

- Class variance ratio - between classes vs within classes

- PLS-DA, soft PLS-DA

- MNF

- CARS feature selection

- Global normalization for visualization (image plotting) functions
...

TODO
----
- deduplicate - keep latest versions
- unify cube shape convention (H, W, B)
- Undefined dependencies
- ...

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
def detect_line_defects(cube, z_thresh=5.0, min_fraction=0.9):
    """
    Detect (col, band) defects where most rows at that (col,band) are extreme outliers.
    Returns a boolean array defect_mask of shape (rows, cols, bands) and a summary list of (col,band).
    
    Strategy:
      - For each band, compute the column-wise values averaged over rows (or L2 across rows).
      - Compute a z-score across columns for that band; columns with high z indicate anomaly.
      - Verify that the anomaly affects >= min_fraction of rows at that (col,band) by checking
        per-row deviation at that (col,band).
    
    Parameters:
      cube : ndarray (r,c,b)
      z_thresh : float, z-score threshold for column anomaly at each band
      min_fraction : float in (0,1], fraction of rows that must be outliers to call it a line defect
    
    Returns:
      defect_mask : bool ndarray (r,c,b) True where pixel considered defective
      defects_list : list of tuples (col, band)
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


def repair_line_defect_spatial(cube, defect_mask, cols_bands=None):
    """
    Repair detected (col,band) defects by replacing each defective (col,band) with the
    median of its spatial neighbors for the same band (left & right columns).
    
    If neighbors are also defective, expands to 2-column radius median.
    
    Parameters:
      cube : ndarray (r,c,b)
      defect_mask : bool ndarray (r,c,b) marking defective pixels to repair
      cols_bands : optional list of (col,band) tuples to process (if None, derive from mask)
    
    Returns:
      cube_fixed : ndarray same shape as cube
      repaired_mask : bool ndarray marking pixels actually replaced
    """
    r, c, b = cube.shape
    cube_fixed = cube.copy().astype(float)
    repaired_mask = np.zeros_like(defect_mask, dtype=bool)
    
    if cols_bands is None:
        # find unique (col,band) from mask
        idx = np.where(np.any(defect_mask, axis=0))  # gives (cols, bands) grid boolean
        # simpler: iterate through columns and bands where any row True
        cols, bands = np.where(np.any(defect_mask, axis=0))
        cols_bands = list(zip(cols.tolist(), bands.tolist()))
    
    for col, bi in cols_bands:
        # determine neighbor columns to use
        left = col - 1 if col - 1 >= 0 else None
        right = col + 1 if col + 1 < c else None
        neighbor_vals = []
        if left is not None:
            neighbor_vals.append(cube[:, left, bi])
        if right is not None:
            neighbor_vals.append(cube[:, right, bi])
        if len(neighbor_vals) == 0:
            continue  # nothing to do (single-column image)
        neighbor_stack = np.stack(neighbor_vals, axis=0)  # (n_neighbors, r)
        # if neighbors include NaNs or other defects, take median across available neighbors
        rep = np.median(neighbor_stack, axis=0)  # shape (r,)
        # apply replacement only at rows marked defective
        rows_to_replace = defect_mask[:, col, bi]
        cube_fixed[rows_to_replace, col, bi] = rep[rows_to_replace]
        repaired_mask[rows_to_replace, col, bi] = True
    
    return cube_fixed, repaired_mask


def repair_line_defect_spectral(cube, defect_mask, cols_bands=None):
    """
    Repair detected (col,band) defects by spectral interpolation using adjacent bands
    for the same pixel (row,col). Uses median of band-1 and band+1 (or nearest available).
    
    This is useful when spatial neighbors are also defective or when spectral continuity is preferred.
    
    Returns cube_fixed, repaired_mask.
    """
    r, c, b = cube.shape
    cube_fixed = cube.copy().astype(float)
    repaired_mask = np.zeros_like(defect_mask, dtype=bool)
    
    if cols_bands is None:
        cols, bands = np.where(np.any(defect_mask, axis=0))
        cols_bands = list(zip(cols.tolist(), bands.tolist()))
    
    for col, bi in cols_bands:
        rows = np.where(defect_mask[:, col, bi])[0]
        for row in rows:
            # find available spectral neighbors
            if bi == 0:
                val = cube[row, col, bi+1]
            elif bi == b-1:
                val = cube[row, col, bi-1]
            else:
                val = np.median([cube[row, col, bi-1], cube[row, col, bi+1]])
            cube_fixed[row, col, bi] = val
            repaired_mask[row, col, bi] = True
    return cube_fixed, repaired_mask


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


def detect_dead_and_outlier_pixels(cube: np.ndarray, lower_thresh: float = 0.1, upper_thresh: float | None = None, z_thresh: int = 3):
    """
    Detect dead pixels and outliers in a hyperspectral cube.

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


def compute_snr_per_band(cube, mask=None):
    """
    Compute SNR per band after masking bad pixels.
    
    Parameters:
        cube: np.ndarray, shape (H, W, B)
        mask: boolean array of same shape, True = good pixel
    
    Returns:
        snr: np.ndarray, length B
    """
    if mask is None:
        mask = np.ones_like(cube, dtype=bool)
    
    B = cube.shape[2]
    snr = np.zeros(B)
    
    for b in range(B):
        band_pixels = cube[:,:,b][mask[:,:,b]]
        if band_pixels.size == 0:
            snr[b] = 0
        else:
            snr[b] = np.mean(band_pixels) / (np.std(band_pixels) + 1e-8)
    
    return snr


def block_average_cube(cube, block_size=5):
    H, W, B = cube.shape

    H_crop = H - (H % block_size)
    W_crop = W - (W % block_size)
    cube_cropped = cube[:H_crop, :W_crop, B]

    h_blocks = H_crop // block_size
    w_blocks = W_crop // block_size
    cube_blocks = cube_cropped.reshape(h_blocks, block_size, w_blocks, block_size, B)
    
    averaged = cube_blocks.mean(axis=(1, 3))

    return averaged


# VARIANCE RATIO - BETWEEN VS WITHIN CLASSES

def class_variance_ratio(X, y):
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

# Reflectance to absorbance - Lambert-Beer or Kubelka-Munk
def reflectance_to_absorbance(R, method='lambert'):
    """
    Convert reflectance spectra to absorbance (Lambert-Beer) or Kubelka-Munk units.

    Parameters:
    R : array-like
        Reflectance values (0 < R <= 1)
    method : str
        'lambert' for Lambert-Beer, 'kubelka' for Kubelka-Munk

    Returns:
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

def robust_snv(cube):
    median = np.nanmedian(cube, axis=0)
    iqr = np.nanpercentile(cube, 75, axis=0) - np.nanpercentile(cube, 25, axis=0)
    iqr[iqr == 0] = 1 # Avoid division by zero if IQR is 0
    normalized_cube = (cube - median) / iqr
    return normalized_cube

def rnv(cube):
    H, W, B = cube.shape
    pixels = cube.reshape(-1, B)

    medians = np.median(pixels, axis=1, keepdims=True)
    mad = np.median(np.abs(pixels - medians), axis=1, keepdims=True)

    mad[mad == 0] = 1e-6

    normalized = (pixels - medians) / mad
    return normalized.reshape(H, W, B)


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