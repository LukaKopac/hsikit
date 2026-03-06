import numpy as np
from numpy.typing import NDArray
import tifffile

import os
from pathlib import Path

def load_hsi_raw(base_path: str, return_metadata: bool = False, verbose: bool = False) -> NDArray | tuple[NDArray, dict]:
    """
    Loads a hyperspectral data cube from a .raw + .hdr pair.

    Parameters
    ----------
    base_path : str 
        Path to file without extension (e.g., 'folder/file' for 'folder/file.raw').
    return_metadata : bool
        If True, also returns the metadata dictionary.
    verbose : bool
        If True, prints 'Loaded HSI data (shape)'

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

def load_wavelengths(hdr_file_path: str) -> NDArray:
    """
    Loads wavelengths from .hdr file and return them in a numpy array.

    Parameters
    ----------
    hdr_file_path : str
        Path to the .hdr file

    Returns
    -------
    NDArray
        1D array of wavelengths
    """
    with open(hdr_file_path, 'r') as file:
        hdr_data = file.read()
    wavelengths_str = hdr_data.split('wavelength = {')[1].strip('}')
    wavelengths_list = wavelengths_str.split(',')
    wavelengths = [float(value.strip()) for value in wavelengths_list]
    wavelengths = np.array(wavelengths)
    print("Number of wavelengths:", len(wavelengths))
    print("Max wavelength:", np.max(wavelengths))
    print("Min wavelength:", np.min(wavelengths))
    return wavelengths

def find_hsi_basepaths(root_folder: str, suffix: str = "_refl") -> list[str]:
    """
    Finds HSI basepaths in a parent/root folder. Compatible with load_hsi_raw function.

    Parameters
    ----------
    root_folder : str
        Path to the root folder
    suffix : str
        Suffix of the wanted files, default "_refl"

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

def load_sample_mapping(txt_path: str) -> dict[str, list[str]]:
    """
    Loads species mapping from a text file.

    Expected format
    ---------------
    scene01: sp1, sp2, sp3...
    scene02: sp4, sp5, sp6...
    
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

def batch_load_hsi(root_folder: str,
                   suffix: str = "refl",
                   return_metadata: bool = False,
                   return_wavelengths: bool = False,
                   return_names: bool = False
) -> dict[str, object]:
    """
    Batch-loads all hypersprectral cubes from a folder.
    
    Parameters
    ----------
    root_folder : str
        Path to root folder containing .hdr/.raw pairs
    suffix : str
        Filename suffix to filter (default "refl")
    return_metadata : bool
        if True, include metadata dicts for each cube
    return_wavelengths : bool
        if True, include wavelengths from the first .hdr
    return_names : bool
        if True, include list of cube names

    Returns
    -------
    dict
        {
            "cubes": list[NDArray],
            "metadata": dict[str, dict] | None,
            "wavelengths": NDArray | None,
            "names": list[str] | None
        }
    """
    basepaths = find_hsi_basepaths(root_folder, suffix=suffix)

    cubes = []
    names = []
    metadata_dict = {} if return_metadata else None
    wavelengths = None

    for base in basepaths:
        filename = Path(base).stem
        names.append(filename)
        
        if return_metadata:
            cube, meta = load_hsi_raw(base, return_metadata=True)
            metadata_dict[filename] = meta
        else:
            cube = load_hsi_raw(base, return_metadata=False)

        cubes.append(cube)
        print(f"Loaded {filename} {cube.shape}")

    if return_wavelengths and basepaths:
        hdr_path = basepaths[0] + ".hdr"
        wavelengths = load_wavelengths(hdr_path)

    return {
        "cubes": cubes,
        "metadata": metadata_dict,
        "wavelengths": wavelengths if return_wavelengths else None,
        "names": names if return_names else None
    }

def export_tiff_stack(cube: NDArray, filename: str, verbose: bool = True) -> None:
    """
    Exports hyperspectral cube as a TIFF stack.

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