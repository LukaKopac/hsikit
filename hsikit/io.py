import numpy as np
import tifffile
import os
from pathlib import Path

def load_hsi_raw(base_path, return_metadata=False, verbose=False):
    """"
    Loads a hyperspectral data cube from a .raw + .hdr pair.

    Parameters:
    - base_path (str): Path to file without extension (e.g., 'folder/file' for 'folder/file.raw').
    - return_metadata (bool): If True, also returns the metadata dictionary.
    - verbose (bool): If True, prints 'Loaded HSI data (shape)'

    Returns:
    - cube (np.ndarray): 3D array (rows x cols x bands).
    - metadata (dict, optional): Return tuple of cube and dictionary of metadata if return_metadata=True.
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

def load_wavelengths(hdr_file_path: str):
    """
    Loads wavelengths from .hdr file and returns them in a numpy array.

    Parameters:
    - hdr_file_path (str): path to the .hdr file

    Returns:
    - Wavelengths vector (ndarray): 1D array
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

def find_hsi_basepaths(root_folder, suffix="_refl"):
    """
    Finds hsi basepaths in a parent/root folder. Compatible with load_hsi_raw function.

    Parameters:
    - root_folder (str): path to the root folder
    - suffix (str): suffix of the wanted files, default "_refl"

    Returns:
    - Array of basepaths
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

def load_sample_mapping(txt_path):
    """
    Load species mapping from a text file.
    Expected format:
        scene01: sp1, sp2, sp3
        scene02: sp4, sp5, sp6...
    Parameters:
    - txt_path (str): path to the mapping file

    Returns:
    - mapping (dict): {scene_name: [species1, species2, ...], ...}
    """
    mapping = {}
    with open(txt_path, "r") as f:
        for line in f:
            if ":" in line:
                scene, species_str = line.strip().split(":", 1)
                species = [s.strip() for s in species_str.split(",")]
                mapping[scene.strip()] = species
    return mapping

def batch_load_hsi(root_folder, suffix="refl", return_metadata=False, return_wavelengths=False, return_names=False):
    """
    Batch-loads all hypersprectral cubes from a folder.
    
    Parameters:
    - root_folder (str): Path to root folder containing .hdr/.raw pairs
    - suffix (str): Filename suffix to filter (default "_refl)
    - return_metadata (bool): if True, returns metadata dicts for each cube
    - return_wavelengths (bool): if True, returns wavelengths from the first .hdr

    Returns:
    - cubes (list of np.ndarray): list of hyperspectral cubes (rows x cols x bands)
    - metadata_list (list of dict, optional): list of metadata dicts if return_metadata=True
    - wavelengths (np.ndarray, optional): array of wavelengths if return_wavelengths=True
    """
    basepaths = find_hsi_basepaths(root_folder, suffix=suffix)

    cubes = []
    names = []
    metadata_dict = {}
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

    results = [cubes]
    
    if return_metadata:
        results.append(metadata_dict)
    if return_wavelengths:
        results.append(wavelengths)
    if return_names:
        results.append(names)

    if len(results) == 1:
        return results[0]
    return tuple(results)

def export_tiff_stack(hsi_cube, filename):
    """
    Exports hyperspectral cube as a TIFF stack.

    Parameters:
    - hsi_cube (3D array): expected shape (h, w, b)
    - filename (str): file name or file path
    """
    transposed_data = np.transpose(hsi_cube, (2, 0, 1))
    tifffile.imwrite(filename+".tif", transposed_data)