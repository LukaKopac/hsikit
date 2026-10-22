"""
Input/output utilities for hyperspectral imaging (HSI) data.

Handles reading raw files, parsing metadata, and saving processed data.

Note: This module is under active development and may change.
"""

import numpy as np
from numpy.typing import NDArray
import tifffile

from collections.abc import Iterator
from typing import Any, Literal

from hsikit.binary_masks import rect_extraction, mask_top_contrast
import csv
from PIL import Image

import os
from pathlib import Path
import gc

def find_hsi_basepaths(root_folder: str, suffix: str = "_refl") -> list[str]:
    """
    Finds HSI basepaths in a parent/root folder. Compatible with load_hsi_raw() and used in load_hsi_batch().

    Parameters
    ----------
    root_folder : str
        Path to the root folder.
    suffix : str
        Suffix of the wanted files, default "_refl".

    Returns
    -------
    list[str]
        A list of HSI basepaths.
    """
    basepaths = []
    for dirpath, _, filenames in os.walk(root_folder):
        filenames_lower = [f.lower() for f in filenames]  # for raw match

        for f in filenames:
            if f.lower().endswith('.hdr') and suffix.lower() in f.lower():
                base = f[:-4]  # remove '.hdr'
                raw_name = base + '.raw'

                # Check if matching .raw file exists (case-insensitive)
                if raw_name.lower() in filenames_lower:
                    full_path = os.path.join(dirpath, base)
                    basepaths.append(full_path)
    return basepaths

def load_hsi_raw(
        base_path: str,
        return_metadata: bool = False, 
        scale_to_reflectance: bool = True, 
        verbose: bool = False
    ) -> NDArray | tuple[NDArray, dict]:
    """
    Loads a hyperspectral data cube from a .raw + .hdr pair.

    Parameters
    ----------
    base_path : str 
        Path to file without extension (e.g., 'folder/file' for 'folder/file.raw').
    return_metadata : bool
        If True, also returns the metadata dictionary.
    scale_to_reflectance : bool
        If True (default), divide values by 10000 to convert to reflectance.
    verbose : bool
        If True, prints 'Loaded HSI data (shape)'.

    Returns
    -------
    NDArray or tuple[NDArray, dict]
        If return_metadata is False, returns the hyperspectral cube
        (H, W, B). If True, returns a tuple (cube, metadata).
    """
    hdr_path = base_path + ".hdr"
    raw_path = base_path + ".raw"

    with open(hdr_path, 'r') as f:
        header_lines = f.readlines()

    header_dict = {}
    for line in header_lines:
        if "=" in line:
            key, value = line.strip().split("=", 1)
            header_dict[key.strip().lower()] = value.strip().strip('{}')

    # Get required metadata
    samples = int(header_dict.get("samples", 0))
    lines = int(header_dict.get("lines", 0))
    bands = int(header_dict.get("bands", 0))
    interleave = header_dict.get("interleave", "bil").lower()
    data_type = int(header_dict.get("data type", 0))

    # Map ENVI data types to NumPy dtypes
    data_type_map = {
        1: np.uint8,
        2: np.int16,
        3: np.int32,
        4: np.float32,
        5: np.float64,
        12: np.uint16,
    }

    dtype = data_type_map.get(data_type)
    if dtype is None:
        raise ValueError(f"Unsupported data type {data_type} in {hdr_path}")

    # Load and reshape .raw data according to interleave
    flat_data = np.fromfile(raw_path, dtype=dtype)

    if scale_to_reflectance:
        flat_data = flat_data.astype(np.float32, copy=False)
        flat_data /= 10000.0

    if interleave == 'bil':
        cube = flat_data.reshape((lines, bands, samples))
        cube = np.transpose(cube, (0, 2, 1))
    elif interleave == 'bsq':
        cube = flat_data.reshape((bands, lines, samples))
        cube = np.transpose(cube, (1, 2, 0))
    elif interleave == 'bip':
        cube = flat_data.reshape((lines, samples, bands))
    else:
        raise ValueError(f"Unsupported interleave format: {interleave}")
    
    if verbose:
        print(f"Loaded HSI data cube: {lines} rows x {samples} cols x {bands} bands")

    if return_metadata:
        return cube, header_dict
    else:
        return cube

def load_wavelengths(hdr_file_path: str, verbose: bool = False) -> NDArray:
    """
    Loads wavelengths from .hdr file and return them in a numpy array.

    Parameters
    ----------
    hdr_file_path : str
        Path to the .hdr file.
    verbose : bool
        Whether to print min, max and number of wavelengths.

    Returns
    -------
    NDArray
        1D array of wavelengths.
    """
    with open(hdr_file_path, 'r') as file:
        hdr_data = file.read()
    wavelengths_str = hdr_data.split('wavelength = {')[1].strip('}')
    wavelengths_list = wavelengths_str.split(',')
    wavelengths = [float(value.strip()) for value in wavelengths_list]
    wavelengths = np.array(wavelengths)

    if verbose:
        print("Number of wavelengths:", len(wavelengths))
        print("Max wavelength:", np.max(wavelengths))
        print("Min wavelength:", np.min(wavelengths))
        
    return wavelengths

def load_sample_mapping(txt_path: str) -> dict[str, list[str]]:
    """
    Loads species mapping from a text file.

    Expected format
    ---------------
    scene01: sp1, sp2, sp3...
    scene02: sp4, sp5, sp6...
    ...
    
    Parameters
    ----------
    txt_path : str
        Path to the mapping file.

    Returns
    -------
    dict[str, list[str]]
        Dictionary with scene names as keys and list of species as values:
        {scene_name: [species1, species2, ...], ...}
    """
    mapping = {}
    with open(txt_path, "r") as f:
        for line in f:
            if ":" in line:
                scene, species_str = line.strip().split(":", 1)
                species = [s.strip() for s in species_str.split(",")]
                mapping[scene.strip()] = species
    return mapping

def load_hsi_batch(
        root_folder: str,
        suffix: str = "_refl",
        return_metadata: bool = False,
        return_wavelengths: bool = False,
        return_names: bool = False,
        verbose: bool = True
    ) -> dict[str, object]:
    """
    Batch-loads all hypersprectral cubes from a folder and returns a dictionary containing loaded data, along with optional metadata.
    
    Parameters
    ----------
    root_folder : str
        Path to root folder containing .hdr/.raw pairs.
    suffix : str
        Filename suffix to filter (default "refl").
    return_metadata : bool
        if True, include metadata dicts for each cube.
    return_wavelengths : bool
        if True, include wavelengths from the first .hdr.
    return_names : bool
        if True, include list of cube names.
    verbose : bool
        if True, prints the loaded files and their shape.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``"cubes"``
            List of loaded hyperspectral cubes.

        ``"metadata"``, optional
            Metadata for each cube. Included only if ``return_metadata=True``.

        ``"wavelengths"``, optional
            Wavelength vector from the first cube. Included only if ``return_wavelengths=True``.

        ``"names"``, optional
            Cube filenames without extensions. Included only if ``return_names=True``.
    """
    basepaths = find_hsi_basepaths(root_folder, suffix=suffix)

    cubes = []
    names = []
    metadata_dict = {}

    for base in basepaths:
        filename = Path(base).stem
        names.append(filename)
        
        if return_metadata:
            cube, meta = load_hsi_raw(base, return_metadata=True)
            metadata_dict[filename] = meta
        else:
            cube = load_hsi_raw(base, return_metadata=False)
            assert isinstance(cube, np.ndarray)

        cubes.append(cube)
        if verbose:
            print(f"Loaded {filename} {cube.shape}")

    result: dict[str, object] = {"cubes": cubes}

    if return_metadata:
        result["metadata"] = metadata_dict

    if return_wavelengths and basepaths:
        result["wavelengths"] = load_wavelengths(basepaths[0] + ".hdr")

    if return_names:
        result["names"] = names

    return result

# Load and process HSI cubes sequentially (one by one) to reduce RAM workload
# For larger datasets
def _iter_hsi_cubes(
    root_folder: str | Path,
    suffix: str = "_refl",
    return_metadata: bool = False,
    verbose: bool = True
) -> Iterator[tuple[str, NDArray, dict[str, Any] | None]]:
    """Yield one HSI cube at a time as (name, cube, metadata)"""

    basepaths = find_hsi_basepaths(root_folder, suffix=suffix)

    for base in basepaths:
        name = Path(base).stem

        if return_metadata:
            cube, metadata = load_hsi_raw(base, return_metadata=True)
        else:
            cube = load_hsi_raw(base, return_metadata=False)
            metadata = None
    
        if verbose:
            print(f"Loaded {name}: {cube.shape}, {cube.dtype}")
    
        yield name, cube, metadata
    
        # Release generator references before loading the next cube
        del cube, metadata

def _assign_grid_locations(
    coords: list[tuple[int, int]],
    rect_dims: tuple[int, int] | list[tuple[int, int]],
    tolerance: int = 5,
) -> list[tuple[int, int]]:
    """Assign 1-based grid locations from ROI coordinates and dimensions."""

    if not coords:
        return []

    coords_array = np.asarray(coords, dtype=float)

    dims_array = np.asarray(rect_dims, dtype=float)

    # Allow either a single shared size or individual sizes.
    if dims_array.shape == (2,):
        dims_array = np.tile(dims_array, (len(coords), 1))

    if dims_array.shape != (len(coords), 2):
        raise ValueError(
            "rect_dims must be a (height, width) tuple or "
            "one (height, width) tuple per coordinate."
        )

    # Calculate individual ROI centres.
    row_centres = coords_array[:, 0] + dims_array[:, 0] / 2
    col_centres = coords_array[:, 1] + dims_array[:, 1] / 2

    def cluster_positions(values):
        order = np.argsort(values)
        labels = np.empty(len(values), dtype=int)

        cluster = 0
        cluster_values = [values[order[0]]]
        labels[order[0]] = cluster

        for idx in order[1:]:
            centre = np.mean(cluster_values)

            if abs(values[idx] - centre) > tolerance:
                cluster += 1
                cluster_values = [values[idx]]
            else:
                cluster_values.append(values[idx])

            labels[idx] = cluster

        return labels

    row_ids = cluster_positions(row_centres)
    col_ids = cluster_positions(col_centres)

    locations = [
        (int(row + 1), int(col + 1))
        for row, col in zip(row_ids, col_ids)
    ]

    if len(set(locations)) != len(locations):
        raise ValueError(
            "Multiple ROIs mapped to the same grid location. "
            "Check the tolerance or ROI coordinates."
        )

    return locations

def _save_sample_cubes(
    cube: NDArray,
    binary_mask: NDArray,
    output_folder: str | Path,
    cube_name: str,
    mode: Literal["row", "column"] = "column",
    min_frac: float = 0.9,
    rect_dims: tuple[int, int] | None = None,
    auto_size_mode: Literal["min", "minmin", "fit_all"] = "min",
    margin: int = 0,
    min_area: int = 10,
    fallback_dims: tuple[int, int] = (32, 32),
) -> list[Path]:
    """Extract and save individual specimen HSI cubes."""

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    binary_mask = np.asarray(binary_mask, dtype=bool)

    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D cube, got {cube.shape}")

    if binary_mask.shape != cube.shape[:2]:
        raise ValueError(
            f"Mask shape {binary_mask.shape} does not match "
            f"cube spatial dimensions {cube.shape[:2]}"
        )

    _, coords, dims = rect_extraction(
        binary_mask=binary_mask,
        rect_dims=rect_dims,
        mode=mode,
        min_frac=min_frac,
        visualize=False,
        return_masks=False,
        auto_size_mode=auto_size_mode,
        margin=margin,
        min_area=min_area,
        fallback_dims=fallback_dims,
    )

    grid_locations = _assign_grid_locations(
        coords=coords,
        rect_dims=dims,
        tolerance=5,
    )

    saved_paths = []

    for (top, left), (height, width), (row, col) in zip(
        coords, dims, grid_locations
    ):
        sample_cube = cube[
            top:top + height,
            left:left + width,
            :
        ]

        sample_mask = binary_mask[
            top:top + height,
            left:left + width
        ]

        if sample_cube.shape[:2] != (height, width):
            continue

        # Remove background pixels from the extracted cube.
        sample_cube = sample_cube.copy()
        sample_cube[~sample_mask] = 0

        sample_id = f"{cube_name}_sample_r{row:02d}_c{col:02d}"
        output_path = output_folder / f"{sample_id}.npz"

        np.savez_compressed(
            output_path,
            cube=sample_cube,
            origin=np.asarray([top, left], dtype=np.int32),
            sample_id=np.asarray(sample_id),
            source_cube=np.asarray(cube_name),
        )

        saved_paths.append(output_path)

    return saved_paths

def process_hsi_folder_to_samples(
    root_folder: str | Path,
    output_folder: str | Path,
    suffix: str = "_refl",
    mode: Literal["row", "column"] = "column",
    min_frac: float = 0.9,
    rect_dims: tuple[int, int] | None = None,
    auto_size_mode: Literal["min", "minmin", "fit_all"] = "min",
    margin: int = 0,
    min_area: int = 10,
    fallback_dims: tuple[int, int] = (32, 32),
    verbose: bool = True,
) -> list[dict]:
    """Process HSI acquisitions sequentially and save specimen cubes
    plus one full-scene foreground mask per acquisition.
    """

    output_folder = Path(output_folder)
    sample_folder = output_folder / "sample_cubes"
    sample_folder.mkdir(parents=True, exist_ok=True)

    log_path = output_folder / "processing_log.csv"
    results = []

    write_header = (
        not log_path.exists() or log_path.stat().st_size == 0
    )

    with log_path.open("a", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "source_cube",
                "status",
                "n_samples",
                "error",
            ],
        )

        if write_header:
            writer.writeheader()

        for name, cube, _ in _iter_hsi_cubes(
            root_folder,
            suffix=suffix,
            return_metadata=False,
            verbose=verbose,
        ):
            try:
                # 1. Generate the full-scene foreground mask.
                foreground_mask = mask_top_contrast(
                    cube,
                    visualize=False,
                    manual_max_band=20,
                )

                # 2. Save the individual specimen cubes.
                paths = _save_sample_cubes(
                    cube=cube,
                    binary_mask=foreground_mask,
                    output_folder=sample_folder,
                    cube_name=name,
                    mode=mode,
                    min_frac=min_frac,
                    rect_dims=rect_dims,
                    auto_size_mode=auto_size_mode,
                    margin=margin,
                    min_area=min_area,
                    fallback_dims=fallback_dims,
                )

                # 3. Save one mask for the entire scene.
                mask_image = Image.fromarray(foreground_mask.astype(np.uint8) * 255)
                mask_image.save(output_folder / f"{name}_mask.png")

                result = {
                    "source_cube": name,
                    "status": "success",
                    "n_samples": len(paths),
                    "error": "",
                }

                if verbose:
                    print(
                        f"{name}: saved {len(paths)} specimen cubes "
                        "and full-scene mask"
                    )

            except Exception as exc:
                result = {
                    "source_cube": name,
                    "status": "error",
                    "n_samples": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }

                if verbose:
                    print(f"ERROR processing {name}: {result['error']}")

            finally:
                # Release large arrays before loading the next acquisition.
                if "foreground_mask" in locals():
                    del foreground_mask

                del cube

            writer.writerow(result)
            log_file.flush()
            results.append(result)

    return results

# Lazy load individual sample cubes, calculate median/mean spectrum per sample and return X (n_samples, n_bands), y (n_samples,) arrays
def load_sample_spectra(
    folder: str | Path,
    statistic: Literal['median', 'mean'] = 'median',
    label_key: str = 'sample_id',
    pattern: str ='*.npz',
    verbose: bool = True
) -> tuple[NDArray, NDArray]:
    """
    Lazily load individual sample cubes and calculate one spectrum per sample.

    Parameters
    ----------
    folder : str or Path
        Folder containing individual sample .npz files
    statistic : {'mean', 'median'}
        Statistic calculated over the spatial dimensions of each cube.
    label_key : str
        Key used to obtain the sample label from each .npz file.
    pattern : str
        Glob pattern for selecting sample files.
    verbose : bool
        Print processing progress.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_bands)
        Mean or median spectrum for each sample.
    y : np.ndarray, shape (n_samples,)
        Sample labels.
    """
    folder = Path(folder)

    if statistic not in {'mean', 'median'}:
        raise ValueError("statistic must be 'mean' or 'median'")

    files = sorted(folder.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {folder}")

    spectra = []
    labels = []

    for i, file in enumerate(files, start=1):
        with np.load(file, allow_pickle=False) as data:
            cube = data['cube']

            if statistic == 'median':
                spectrum = np.median(cube, axis=(0,1))
            else:
                spectrum = np.mean(cube, axis=(0,1))

            label = data[label_key].item()

            # Keep only the spectrum and label, not the cube
            spectra.append(spectrum)
            labels.append(label)

        del cube, spectrum

        if verbose:
            print(f"\rProcessed {i}/{len(files)} samples", end="")

    if verbose:
        print()

    X = np.stack(spectra)
    y = np.asarray(labels)

    del spectra, labels
    gc.collect()

    return X, y


# Export hyperspectral cube as a TIFF stack of images (can be opened with ImageJ/FIJI)
def export_tiff_stack(cube: NDArray, filename: str, verbose: bool = True) -> None:
    """
    Exports hyperspectral cube as a TIFF stack of images.

    Parameters
    ----------
    cube : NDArray
        Expected shape (H, W, B)
    filename : str
        File name or file path
    
    Returns
    -------
    None
        Saves data to disk and doesn't return anything.
    """
    transposed_data = np.transpose(cube, (2, 0, 1))
    filename = str(Path(filename).with_suffix(".tif"))
    tifffile.imwrite(filename, transposed_data)
    if verbose:
        print(f"Successfully saved the cube ({cube.shape}) as a tiff stack")