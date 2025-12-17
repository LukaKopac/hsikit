import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'

# Basic utility, processing, scaling, normalization, PCA

def convert_to_reflectance(cube):
    """
    Divide all values in a hyperspectral cube by 10,000.

    Parameters:
        cube (np.ndarray): 3D hyperspectral data (H, W, B).

    Returns:
        scaled_cube (np.ndarray): Cube with values divided by 10,000.
    """
    return cube.astype(np.float32) / 10000.0

def normalize_min_max(cube, return_params=False):
    """
    Min-max normalize a 3D hyperspectral cube

    Parameters:
    - cube (array): 3D array, expected shape (H, W, B)
    - return_params (bool): whether to return min and max values

    Returns:
    - norm_cube (array): normalized cube
    - min_vals (): minimum values per band (B,)
    - max_vals (): maximum values per band (B,)
    """
    h, w, b = cube.shape
    X = cube.reshape(-1, b)
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    norm_X = (X - min_vals) / (max_vals - min_vals + 1e-8)
    norm_cube = norm_X.reshape(h, w, b)
    if return_params:
        return norm_cube, min_vals, max_vals
    else:
        return norm_cube

def normalize_mean_std(cube, return_params=False):
    """
    Standardize a 3D hyperspectral cube using mean and std.

    Parameters:
    - cube (array): 3D array, expected shape (H, W, B)
    - return_params (bool): whether to return mean and std values

    Returns:
    - norm_cube (array): standardized cube
    - mean_vals (): mean values per band (B,)
    - std_vals (): std deviation values per band (B,)
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

def apply_pca(cube, n_components=3, mask=None, return_model=False):
    """
    Apply PCA on a 3D hyperspectral cube.

    Parameters:
    - cube (array): 3D array, expected shape (H, W, B)
    - n_components (int): number of components used for PCA
    - mask (array or None): optional boolean mask of shape (h, w)
    - return_model (bool): whether to return the fitter PCA model

    Returns:
    - pca_cube (array): PCA-transformed cube of shape (H, W, n_components)
    - pca (): trained PCA object
    """
    h, w, b = cube.shape
    X = cube.reshape(-1, b)

    if mask is not None:
        mask_flat = mask.flatten()
        X_masked = X[mask_flat]
        pca = PCA(n_components=n_components)
        X_pca_masked = pca.fit_transform(X_masked)
        X_pca = np.zeros((h * w, n_components))
        X_pca[:] = np.nan
        X_pca[mask_flat] = X_pca_masked
    else:
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
    pca_cube = X_pca.reshape(h, w, n_components)
    
    if return_model:
        return pca_cube, pca
    else:
        return pca_cube

# Plotting and visualization

def plot_band_image(cube, band_index, title=None, cmap='gray', show_grid=False):
    """
    Plot a single band/component from a 3D hyperspectral cube.

    Parameters:
    - cube (np.ndarray): 3D array (H, W, B).
    - band_index (int): Index of the band/component to display.
    - title (str): Optional plot title.
    - cmap (str): Matplotlib colormap (default: 'gray').
    - show_grid (bool): If True, overlays a pixel grid for coordinate reference.
    """
    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    im = ax.imshow(cube[:, :, band_index], cmap=cmap)
    plt.title(title or f'Band {band_index}')
    plt.axis('on')

    if show_grid:
        grid_spacing = 50
        h, w, _ = cube.shape
        ax.set_xticks(np.arange(0, w, grid_spacing))
        ax.set_yticks(np.arange(0, h, grid_spacing))
        ax.grid(which='both', color='white', linestyle='--', linewidth=0.7)
        ax.tick_params(length=0)
    else:
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_rgb_composite(cube, bands=(30, 20, 10), title=None, scale_each_band=True):
    """
    Create and display an RGB composite from 3 bands of a hyperspectral cube.

    Parameters:
        cube (np.ndarray): 3D hyperspectral cube (H, W, B).
        bands (tuple): (R, G, B) band indices to use.
        title (str): Optional plot title.
        scale_each_band (bool): Whether to scale each band individually to [0, 1].
    """
    h, w, b = cube.shape
    r, g, b = bands
    rgb = np.stack([cube[:, :, r], cube[:, :, g], cube[:, :, b]], axis=-1)

    if scale_each_band:
        # Normalize each band independently to [0, 1]
        rgb_min, rgb_max = np.nanmin(rgb), np.nanmax(rgb)
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)

    rgb = np.clip(rgb, 0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title(title or f'RGB Composite (Bands {r}, {g}, {b})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_spectra(cube, coords=None, wavelengths=None, labels=None, average=False, window_size=1, title=None):
    """
    Plot spectral profiles from specific pixels, the average spectrum, or both.

    Parameters:
    - cube (np.ndarray): 3D hyperspectral cube (H, W, B).
    - coords (list of (row, col)): Optional list of pixel locations.
    - wavelengths (list or np.ndarray): Optional wavelength values (length B).
    - labels (list of str): Optional labels for each pixel.
    - average (bool): If True, include the average spectrum of the whole cube.
    - window_size (int): Size of the square area around each coord to average (must be odd, default=1).
    - title (str): Optional plot title.
    """
    h, w, b = cube.shape
    x_axis = wavelengths if wavelengths is not None else np.arange(b)

    def get_area_spectrum(center_row, center_col, size):
        half = size // 2
        row_min = max(center_row - half, 0)
        row_max = min(center_row + half + 1, h)
        col_min = max(center_col - half, 0)
        col_max = min(center_col + half + 1, w)
        area = cube[row_min:row_max, col_min:col_max, :]
        return area.reshape(-1, b).mean(axis=0)
    
    plt.figure(figsize=(8, 5))

    if coords:
        for i, (row, col) in enumerate(coords):
            spectrum = get_area_spectrum(row, col, window_size)
            label = labels[i] if labels else f'Pixel ({row}, {col})'
            if window_size > 1:
                label += f' (avg {window_size}x{window_size})'
            plt.plot(x_axis, spectrum, label=label)

    if average:
        avg_spectrum = cube.reshape(-1, b).mean(axis=0)
        plt.plot(x_axis, avg_spectrum, label='Average Spectrum', linewidth=2.5, color='black')

    plt.xlabel('Wavelength' if wavelengths is not None else 'Band Index')
    plt.ylabel('Reflectance / Intensity')
    plt.title(title or 'Spectral Profiles')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_mean_spectrum_with_std(cube, mask=None, wavelengths=None, title=None, color='blue'):
    """
    Plot the mean spectrum with ±1 standard deviation as a shaded area.

    Parameters:
    - cube (np.ndarray): 3D hyperspectral data (H, W, B).
    - mask (np.ndarray or None): Optional binary mask (H, W) to restrict region of interest.
    - wavelengths (list or np.ndarray): Optional list of wavelengths (length B).
    - title (str): Plot title.
    - color (str): Line/shading color (default: 'blue').
    """
    h, w, b = cube.shape
    x_axis = wavelengths if wavelengths is not None else np.arange(b)

    if mask is not None:
        masked_cube = cube[mask]
    else:
        masked_cube = cube.reshape(-1, b)

    mean_spectrum = masked_cube.mean(axis=0)
    std_spectrum = masked_cube.std(axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, mean_spectrum, label='Mean Spectrum', color=color)
    plt.fill_between(x_axis, 
                     mean_spectrum - std_spectrum, 
                     mean_spectrum + std_spectrum, 
                     color=color, alpha=0.3, label='±1 Std Dev')
    plt.xlabel('Wavelength' if wavelengths is not None else 'Band Index')
    plt.ylabel('Reflectance / Intensity')
    plt.title(title or 'Mean Spectrum ± Std Dev')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_spectral_histogram(cube, band=None, mask=None, bins=100, log_scale=False, title=None):
    """
    Plot a histogram of reflectance/intensity values from a single band or entire cube.

    Parameters:
    - cube (np.ndarray): 3D hyperspectral data (H, W, B).
    - band (int or None): If specified, shows histogram for that band; otherwise flattens all bands.
    - mask (np.ndarray or None): Optional binary mask (H, W) to limit pixels.
    - bins (int): Number of histogram bins.
    - log_scale (bool): Whether to use logarithmic y-axis.
    - title (str): Plot title.
    """
    if mask is not None:
        data = cube[mask]
    else:
        data = cube.reshape(-1, cube.shape[2])  # (N, B)

    if band is not None:
        values = data[:, band]
        label = f'Band {band}'
    else:
        values = data.ravel()
        label = 'All Bands'

    plt.figure(figsize=(7, 4))
    plt.hist(values, bins=bins, color='gray', alpha=0.8, edgecolor='black', label=label)
    if log_scale:
        plt.yscale('log')

    plt.xlabel('Intensity / Reflectance')
    plt.ylabel('Pixel Count')
    plt.title(title or f'Histogram of {label}')
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()

def vis_3D_slices(data_cube, spacing=10, num_slices=10, mask=None, stride=3, cmap=plt.cm.gray):
    """
    3D visualization of hyperspectral slices.

    Parameters:
    - data_cube: ndarray (H, W, B)
    - spacing: distance between slices on z-axis
    - num_slices: how many bands to visualize
    - mask: optional binary mask (H, W)
    - stride: meshgrid stride for rendering
    - cmap: matplotlib colormap
    """
    h, w, b = data_cube.shape

    total_bands = data_cube.shape[2]
    if num_slices is None or num_slices >= b:
        slices = np.arange(b)
    else:
        slices = np.linspace(0, b - 1, num_slices, dtype=int)

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, band in enumerate(slices):
        img = data_cube[:, :, band]
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        z = np.full_like(x, (len(slices) -1 - i) * spacing)
    
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
    
        if mask is not None:
            img = np.where(mask, img_norm, np.nan)
            alpha = mask.astype(float)
        else:
            img = img_norm
            alpha = np.ones_like(img_norm)

        rgba = cmap(img)
        rgba[..., -1] = alpha
        
        ax.plot_surface(x, y, z, rstride=stride, cstride=stride, facecolors=rgba, shade=False)
        print(f"Rendered band {band}")
    
    ax.set_box_aspect((w, h, spacing * len(slices)))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_zlim(0, spacing * len(slices))
    ax.set_xlabel('Spatial axis')
    ax.set_ylabel('Spatial axis')
    ax.set_zlabel('Spectral axis')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.view_init(elev=45, azim=45)
    plt.tight_layout()
    plt.show()

def vis_3D_slices_interactive(data_cube, spacing=1, stride=3, num_slices=20, save=False):
    """
    Interactive 3D visualization of hyperspectral slices.

    Parameters:
    - data_cube (ndarray): shape (H, W, B)
    - spacing (int/float): distance between slices on Z axis
    - stride (int): downsample spatial resolution by this factor
    - num_slices (int or None): number of spectral bands to visualize. If None, show all bands.
    - save (bool): if True, save to "3D_stack.svg"
    """
    downsampled = data_cube[::stride, ::stride, :]
    h, w, bands = downsampled.shape
    
    if num_slices is None or num_slices >= bands:
        band_indices = np.arange(bands)
    else:
        band_indices = np.linspace(0, bands - 1, num_slices, dtype=int)
    
    x, y = np.meshgrid(np.arange(w) * stride, np.arange(h) * stride)
    
    surfaces = []    
    for i in band_indices:
        z_offset = (bands - 1 - i) * spacing
        z = np.full((h, w), z_offset)
        surfaces.append(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=downsampled[:, :, i],
                colorscale='gray',
                showscale=False,
                cmin=downsampled.min(),
                cmax=downsampled.max(),
                opacity=1.0
            )
        )

    fig = go.Figure(data=surfaces)
    fig.update_layout(
        title='3D hyperspectral cube',
        scene=dict(
            xaxis=dict(showticklabels=False, title='Spatial axis'),
            yaxis=dict(showticklabels=False, title='Spatial axis'),
            zaxis=dict(showticklabels=False, title='Spectral axis', range=[0, bands * spacing]),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        scene_camera=dict(eye=dict(x=1.6, y=1.6, z=1.5))
    )

    if save:
        fig.write_image("3D_stack.svg")

    fig.show()