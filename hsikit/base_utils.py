import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'notebook'

from typing import Optional, Literal

# Basic utility, processing, scaling, normalization, PCA

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

def apply_pca(
    cube: NDArray,
    n_components: int = 3,
    mask: Optional[NDArray] = None,
    return_model: bool = False
) -> NDArray | tuple[NDArray, PCA]:
    """
    Applies PCA on a hypercube.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    n_components : int
        Number of PCA components to retain.
    mask : NDArray | None
        Optional boolean mask of shape (H, W) to select pixels for PCA.
    return_model : bool
        Whether to return the fitted PCA object.

    Returns
    -------
    NDArray or tuple[NDArray, object]
        If return_model is False, returns PCA-transformed cube of shape (H, W, n_components)
        If True, returns a tuple (pca_cube, pca), containing the transformed cube and fitted PCA object.
    """
    h, w, b = cube.shape
    flat_cube = cube.reshape(-1, b)

    if mask is not None:
        mask_flat = mask.flatten()
        X_masked = flat_cube[mask_flat]
        pca = PCA(n_components=n_components)
        X_pca_masked = pca.fit_transform(X_masked)
        flat_pca = np.full((h * w, n_components), np.nan)
        flat_pca[mask_flat] = X_pca_masked
    else:
        pca = PCA(n_components=n_components)
        flat_pca = pca.fit_transform(flat_cube)
        
    pca_cube = flat_pca.reshape(h, w, n_components)
    
    if return_model:
        return pca_cube, pca
    return pca_cube

# Plotting and visualization

def plot_band_image(cube: NDArray, band_index: int, title: Optional[str] = None, cmap: str = 'gray', show_grid: bool = False) -> None:
    """
    Plots a single specified band image from a hypercube.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    band_index : int
        Index of the band/component to display.
    title : str
        Optional plot title.
    cmap : str
        Matplotlib colormap (default: 'gray').
    show_grid : bool
        If True, overlays a pixel grid for coordinate reference.

    Returns
    -------
    None
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

def plot_rgb_composite(cube: NDArray, bands: tuple = (50, 150, 250), title: Optional[str] = None, scale_each_band: bool = True) -> None:
    """
    Create and display an RGB composite from selected 3 bands of a hypercube.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    bands : tuple
        Band indices to use for (R, G, B).
    title : str
        Optional plot title.
    scale_each_band : bool
        Whether to scale each band individually to [0, 1].

    Returns
    -------
    None
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

def plot_spectra(
    cube: NDArray,
    coords: Optional[list[tuple[int, int]]] = None,
    wavelengths: Optional[list | NDArray] = None,
    labels: Optional[list[str]] = None,
    average: bool = True,
    window_size: int = 1,
    title: Optional[str] = None
) -> None:
    """
    Plot spectral profiles from specific pixels of a hypercube, the average spectrum, or both.  
    Default plots only the average spectrum of the whole cube.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    coords : Optional[list[tuple[int, int]]]
        Optional list of pixel locations (row, col).
    wavelengths : Optional[list | NDArray]
        Optional wavelength values (length B).
    labels : Optional[list[str]]
        Optional labels for each pixel.
    average : bool
        If True, include the average spectrum of the whole cube.
        Default is True.
    window_size : int
        Size of the square area around each coord to average (must be odd, default=1).
    title : Optional[str]
        Optional plot title.

    Returns
    -------
    None
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

def plot_mean_spectrum_with_std(
    cube: NDArray,
    mask: Optional[NDArray] = None,
    wavelengths: Optional[list | NDArray] = None,
    title: Optional[str] = None,
    color: str = 'blue'
) -> None:
    """
    Plot the mean spectrum of a hypercube with ±1 standard deviation as a shaded area.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    mask : Optional[NDArray]
        Optional binary mask (H, W) to restrict region of interest.
    wavelengths : Optional[list | NDArray]
        Optional list of wavelengths (length B).
    title : Optional[str]
        Plot title.
    color : str
        Matplotlib line/shading color (default: 'blue').

    Returns
    -------
    None
    """
    b = cube.shape[-1]
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

def plot_spectral_histogram(
    cube: NDArray,
    band: Optional[int] = None,
    mask: Optional[NDArray] = None,
    bins: int = 100,
    log_scale: bool = False,
    title: Optional[str] = None
) -> None:
    """
    Plot a histogram of reflectance/intensity values from a single band or entire hypercube.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    band : Optional[int]
        If specified, shows histogram for that band; otherwise flattens all bands.
    mask : Optional[NDArray]
        Optional binary mask (H, W) to limit pixels.
    bins : int
        Number of histogram bins.
    log_scale : bool
        Whether to use logarithmic y-axis.
    title : Optional[str]
        Plot title.

    Returns
    -------
    None
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

def plot_3D_slices(
    cube: NDArray,
    spacing: float = 10,
    num_slices: int = 10,
    mask: Optional[NDArray] = None,
    stride: int = 3,
    cmap: str | Colormap = "gray"
) -> None:
    """
    3D visualization of hyperspectral slices.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    spacing : float
        Distance between slices on z-axis.
    num_slices : int
        Number of bands to visualize.
    mask : Optional[NDArray]
        Optional binary mask (H, W).
    stride : int
        Meshgrid stride for rendering.
    cmap : str | Colormap
        matplotlib colormap (default "gray").

    Returns
    -------
    None
    """
    h, w, b = cube.shape

    if mask is not None and mask.shape != (h, w):
        raise ValueError(f"Mask must have shape ({h}, {w})")

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    if num_slices is None or num_slices >= b:
        bands = np.arange(b)
    else:
        bands = np.linspace(0, b - 1, num_slices, dtype=int)

    x, y = np.meshgrid(np.arange(w), np.arange(h))

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, band in enumerate(bands):
        img = cube[:, :, band]
    
        den = img.max() - img.min()
        img_norm = np.zeros_like(img) if den == 0 else (img - img.min()) / den
    
        if mask is not None:
            alpha = (mask > 0).astype(float)
        else:
            alpha = np.ones_like(img_norm)

        rgba = cmap(img_norm)
        rgba[..., -1] = alpha
        
        z = np.full_like(x, (len(bands) -1 - i) * spacing)

        ax.plot_surface(x, y, z, rstride=stride, cstride=stride, facecolors=rgba, shade=False, lw=0, antialiased=False)
        print(f"Rendered band {band}")
    
    ax.set_box_aspect((w, h, spacing * len(bands)))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_zlim(0, spacing * len(bands))
    ax.set_xlabel('Spatial axis')
    ax.set_ylabel('Spatial axis')
    ax.set_zlabel('Spectral axis')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_proj_type('persp')
    ax.view_init(elev=45, azim=45)

    plt.tight_layout()
    plt.show()

def plot_3D_slices_interactive(
    cube: NDArray,
    spacing: float = 1,
    stride: int = 3,
    num_slices: Optional[int] = 20,
    title: Optional[str] = None,
    save: bool = False
) -> None:
    """
    Interactive 3D visualization of hyperspectral slices.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    spacing : float
        Distance between slices on Z axis.
    stride : int
        Downsample spatial resolution by this factor.
    num_slices : Optional[int]
        Number of spectral bands to visualize. If None, show all bands.
    title : Optional[str]
        Optional plot title and file name if saved (if None: 'HSI 3D stack').
    save : bool
        if True, save as title or 'HSI_3D_stack.svg' if title is None

    Returns
    -------
    None
    """
    downsampled = cube[::stride, ::stride, :]
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
        title=title if title else 'HSI 3D stack',
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
        fig.write_image(title if title else 'HSI_3D_stack' + ".svg")

    fig.show()


def plot_hsi_cube(
    cube: NDArray,
    cmap: str = 'jet',
    top_face_mode: Literal['single', 'mean', 'rgb'] = 'mean',  # options: 'single', 'mean', 'rgb'
    single_band_index: int = 0,   # used if top_face_mode=='single'
    rgb_bands: tuple[int, int, int] = (0, 1, 2),  # used if top_face_mode=='rgb'
    rstride: int = 2,
    cstride: int = 2,
    ax: Axes | None = None
) -> tuple[Figure, Axes]:
    """
    Render a hyperspectral data cube as a 3D box with five colored faces.
    Values will be normalized per face for visualization.

    Parameters
    ----------
    cube : NDArray
        HSI 3D array, expected shape (H, W, B).
    cmap : str
        Name of a Matplotlib colormap used for scalar face coloring ('grey', 'viridis', 'jet', 'plasma', 'magma').
    top_face_mode : Literal['single', 'mean', 'rgb']
        Strategy used to color the top face:
        - 'single': visualize a single spectral band
        - 'mean': average across all bands
        - 'rgb': RGB composite from three selected bands
    single_band_index : int
        Band index used when top_face_mode='single'.
    rgb_bands : tuple[int, int, int]
        Band indices used when top_face_mode='rgb'.
    rstride : int
        Row sampling stride for surface plotting (reduces rendering density).
    cstride : int
        Column sampling stride for surface plotting (reduces rendering density).
    ax : Axes | None
        Existing 3D axes to plot into. If None, a new figure and axes are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created or associated figure.
    ax : matplotlib.axes.Axes
        The 3D axes containing the plot.
    """
    rows, cols, bands = cube.shape

    def normalize(data):
        dmin, dmax = data.min(), data.max()
        if dmax == dmin:
            return np.zeros_like(data)
        return np.clip((data - dmin) / (dmax - dmin), 0, 1)

    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    cmap_obj = plt.get_cmap(cmap)

    # --- Left face ---
    x_left = np.zeros((rows, bands))
    y_left = np.repeat(np.arange(rows)[:, None], bands, axis=1)
    z_left = np.tile(np.arange(bands), (rows, 1))
    left_face = normalize(cube[:, 0, :])
    left_face_rgb = cmap_obj(left_face)[:, :, :3]
    ax.plot_surface(x_left, y_left, z_left, facecolors=left_face_rgb,
                    rstride=rstride, cstride=cstride, linewidth=1, alpha=1, shade=False)

    # --- Right face ---
    x_right = np.full((rows, bands), cols)
    y_right = y_left
    z_right = z_left
    right_face = normalize(cube[:, -1, :])
    right_face_rgb = cmap_obj(right_face)[:, :, :3]
    ax.plot_surface(x_right, y_right, z_right, facecolors=right_face_rgb,
                    rstride=rstride, cstride=cstride, linewidth=1, alpha=1, shade=False)

    # --- Back face ---
    x_back, z_back = np.meshgrid(np.arange(cols), np.arange(bands), indexing='ij')
    y_back = np.zeros_like(x_back)
    back_face = normalize(cube[0, :, :])
    back_face_rgb = cmap_obj(back_face)[:, :, :3]
    ax.plot_surface(x_back, y_back, z_back, facecolors=back_face_rgb,
                    rstride=rstride, cstride=cstride, linewidth=1, alpha=1, shade=False)

    # --- Front face ---
    y_front = np.full_like(x_back, rows)
    front_face = normalize(cube[-1, :, :])
    front_face_rgb = cmap_obj(front_face)[:, :, :3]
    ax.plot_surface(x_back, y_front, z_back, facecolors=front_face_rgb,
                    rstride=rstride, cstride=cstride, linewidth=1, alpha=1, shade=False)

    # --- Top face ---
    x_top, y_top = np.meshgrid(np.arange(cols), np.arange(rows), indexing='xy')

    if top_face_mode == 'single':
        z_top = np.full_like(x_top, bands)
        top_face = normalize(cube[:, :, single_band_index])
        top_face_rgb = plt.get_cmap(cmap)(top_face)[:, :, :3]

    elif top_face_mode == 'mean':
        z_top = np.full_like(x_top, bands)
        top_face = normalize(np.mean(cube, axis=2))
        top_face_rgb = cmap_obj(top_face)[:, :, :3]

    elif top_face_mode == 'rgb':
        z_top = np.full_like(x_top, bands)
        r = normalize(cube[:, :, rgb_bands[0]])
        g = normalize(cube[:, :, rgb_bands[1]])
        b = normalize(cube[:, :, rgb_bands[2]])
        top_face_rgb = np.stack([r, g, b], axis=2)

    else:
        raise ValueError("Invalid top_face_mode. Choose 'single', 'mean', or 'rgb'.")

    ax.plot_surface(x_top, y_top, z_top, facecolors=top_face_rgb,
                    rstride=rstride, cstride=cstride, linewidth=1, alpha=1, shade=False)

    # --- Aesthetics ---
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_zlim(0, bands)
    ax.set_box_aspect((cols, rows, bands))
    ax.view_init(elev=30, azim=45)
    ax.axis('off')

    plt.tight_layout()

    return fig, ax