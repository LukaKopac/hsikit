import numpy as np
import tifffile
import os
from pathlib import Path

def import_hsi_raw(base_path, return_metadata=False):
    """"
    Imports a hyperspectral data cube from a .raw + .hdr pair.

    Parameters:
    - base_path (str): Path to file without extension (e.g., 'folder/file' for 'file.raw' and 'file.hdr').
    - return_metadata (bool): If True, also returns the metadata dictionary.

    Returns:
    - cube (np.adarray): 3D array (rows x cols x bands).
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

    #print(f"Imported HSI data cube: {lines} rows x {samples} cols x {bands} bands")

    if return_metadata:
        return cube, header_dict
    else:
        return cube

def import_wavelengths(hdr_file_path: str):
    """
    Imports wavelengths from .hdr file and returns them in a numpy array.

    Parameters:
    - hdr_file_path (str): path to the .hdr file

    Returns:
    - Array of wavelengths
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
    Finds hsi basepaths in a parent/root folder. Compatible with import_hsi_raw function.

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

def import_sample_mapping(txt_path):
    """
    Load species mapping from a text file.
    Returns a dict: {scene_name: [species1, species2, ...]}
    """
    mapping = {}
    with open(txt_path, "r") as f:
        for line in f:
            if ":" in line:
                scene, species_str = line.strip().split(":", 1)
                species = [s.strip() for s in species_str.split(",")]
                mapping[scene.strip()] = species
    return mapping

def batch_import_hsi(root_folder, suffix="refl", return_metadata=False, return_wavelengths=False, return_names=False):
    """
    Batch-imports all hypersprectral cubes from a folder.
    
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
            cube, meta = import_hsi_raw(base, return_metadata=True)
            metadata_dict[filename] = meta
        else:
            cube = import_hsi_raw(base, return_metadata=False)

        cubes.append(cube)
        print(f"Imported {filename} {cube.shape}")

    if return_wavelengths and basepaths:
        hdr_path = basepaths[0] + ".hdr"
        wavelengths = import_wavelengths(hdr_path)

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

class HSIImporter:
    def __init__(self, root_folder, suffix="_refl", mapping_file=None):
        self.root_folder = root_folder
        self.suffix = suffix
        self.mapping_file = mapping_file
        self.cubes = {}
        self.metadata = {}
        self.wavelengths = None
        self.basepaths = []
        self.scenes = {}

    def batch_import(self, return_metadata=False, return_wavelengths=False, auto_mapping=True):
        self.basepaths = find_hsi_basepaths(self.root_folder, suffix=self.suffix)
        self.cubes = {}
        self.metadata = {}
        self.wavelengths = None
        self.scenes = {}

        # --- auto-detect mapping.txt if not specified ---
        if self.mapping_file is None:
            candidate = os.path.join(self.root_folder, "mapping.txt")
            if os.path.exists(candidate):
                self.mapping_file = candidate
                print(f"Detected mapping file: {self.mapping_file}")

        # load global mapping file once
        mapping = {}
        if auto_mapping and self.mapping_file and os.path.exists(self.mapping_file):
            mapping = import_sample_mapping(self.mapping_file)
            print(f"Loaded mapping file: {self.mapping_file} ({len(mapping)} scenes)")
        elif auto_mapping:
            print("⚠️ No mapping file found.")

        # loop through scenes
        for base in self.basepaths:
            basename = os.path.basename(base)
            if basename.endswith(self.suffix):
                scene_name = basename[:-len(self.suffix)]
            else:
                scene_name = basename

            # load cube (+ metadata if requested)
            if return_metadata:
                cube, meta = import_hsi_raw(base, return_metadata=True)
                self.metadata[scene_name] = meta
            else:
                cube = import_hsi_raw(base, return_metadata=False)
                meta = {}

            self.cubes[scene_name] = cube
            self.scenes[scene_name] = {
                "cube": cube,
                "metadata": meta,
                "mapping": mapping.get(scene_name, []),
                "masks": [],
                "combined_mask": None
            }
            print(f"Imported: {basename} -> shape {cube.shape}")

            if scene_name in mapping:
                print(f" → Mapping found for {scene_name} ({len(mapping[scene_name])} entries)")
            else:
                print(f" → No mapping for {scene_name}")

        # load wavelengths (only once)
        if return_wavelengths and self.basepaths:
            self.wavelengths = import_wavelengths(self.basepaths[0] + ".hdr")
            print(f"Wavelengths loaded: {len(self.wavelengths)} bands")

        return self

    def add_masks(self, scene_name, masks):
        """Attach sample masks to a scene and compute combined mask."""
        if scene_name not in self.scenes:
            raise ValueError(f"Scene {scene_name} not found")
        self.scenes[scene_name]["masks"] = masks
        self.scenes[scene_name]["combined_mask"] = np.any(masks, axis=0)
        print(f"Masks added for {scene_name} ({len(masks)} samples)")

    def batch_add_masks(self, mask_func, **kwargs):
        """
        Apply a mask generation function to all scenes in the importer.

        Parameters:
        - mask_func: function that takes a cube (ndarray) and returns a list of masks
        - kwargs: extra arguments passed to mask_func
        """
        for scene_name, scene in self.scenes.items():
            cube = scene["cube"]
            masks = mask_func(cube, **kwargs)
            self.add_masks(scene_name, masks)
        return self

    def extract_species_samples(self):
        """Return dictionary: species -> list of sample cubes (background removed)."""
        species_dict = {}

        for scene_name, scene in self.scenes.items():
            cube = scene["cube"]
            masks = scene.get("masks", [])
            mapping = scene.get("mapping", [])

            if not masks or not mapping:
                print(f"⚠️ Skipping {scene_name}: no masks or mapping")
                continue

            for mask, species in zip(masks, mapping):
                sample = cube * mask[:, :, None]  # apply mask to all bands
                if species not in species_dict:
                    species_dict[species] = []
                species_dict[species].append(sample)

        print(f"Extracted samples for {len(species_dict)} species")
        return species_dict

    def summary(self):
        print("\n📦 HSIImporter summary")
        print(f"Root folder: {self.root_folder}")
        print(f"Suffix: {self.suffix}")
        print(f"Number of scenes: {len(self.scenes)}")

        for name, scene in self.scenes.items():
            print(f" - {name}: cube shape {scene['cube'].shape}, "
                  f"{len(scene['mapping'])} mapping entries, "
                  f"{len(scene['masks'])} masks")

        if self.metadata:
            print("Metadata loaded for scenes")
            all_keys = set()
            for scene in self.scenes.values():
                all_keys.update(scene["metadata"].keys())
            print("Available metadata keys:", ", ".join(sorted(all_keys)))
        else:
            print("No metadata loaded")

        if self.wavelengths is not None:
            print(f"Wavelengths: {len(self.wavelengths)} bands "
                  f"(range {self.wavelengths.min()}–{self.wavelengths.max()})")
        else:
            print("No wavelengths loaded")

class HSIProcessor:
    def __init__(self, folder, mapping_file=None):
        self.folder = Path(folder)
        self.cubes = [] # list of np.arrays (raw data)
        self.meta = [] # metadata per cube
        self.wl = None # wavelengths (shared)
        self.reflectance = [] # list of reflectance cubes
        self.masks = [] # list of per cube masks
        self.rect_masks = [] # list of fixed rect masks (list of lists)
        self.coords = [] # coordinates from rect masks
        if mapping_file is None:
            mapping_candidates = list(self.folder.glob('mapping*'))
            if not mapping_candidates:
                raise FileNotFoundError(f'No mapping file found in {self.folder}')
            self.mapping_file = mapping_candidates[0]
        else:
            self.mapping_file = Path(mapping_file)
        self.translation = {}
        self.mapping = {}
        self.samples_dict = {}

    def load(self, suffix='refl'):
        self.cubes, self.meta, self.wl = batch_import_hsi(
            root_folder=self.folder,
            suffix=suffix,
            return_wavelengths=True,
            return_metadata=True
        )
        return self

    def load_mapping(self):
        with open(self.mapping_file, 'r') as f:
            text = f.read()

        abbrev_text, scenes_text = text.split('\nscene', 1)

        for pair in abbrev_text.strip().split(', '):
            k, v = pair.split('=')
            self.translation[k.strip()] = v.strip()

        for block in scenes_text.strip().split('\nscene'):
            if not block.strip():
                continue
            scene_id, species_str = block.split(':', 1)
            scene_id = f'scene{scene_id.strip()}'
            species_list = [self.translation.get(word.strip(), word.strip())
                           for word in species_str.split(',')]
            self.mapping[scene_id] = species_list

        return self

    def to_reflectance(self):
        self.reflectance = [hsikit.base_utils.convert_to_reflectance(np.array(c)) for c in self.cubes]
        return self

    def compute_masks(self, min_size=1800, visualize=False):
        self.masks = [
            hsikit.bg_removal.mask_top_contrast(c, min_size=min_size, visualize=visualize)
            for c in self.reflectance
        ]
        return self

    def add_rectangles(self, width=60, height=140, min_frac=0.9):
        self.rect_masks = []
        self.coords = []
        for mask in self.masks:
            rects, coords = hsikit.bg_removal.fixed_rect_mask(mask, width, height, min_frac=min_frac)
            self.rect_masks.append(rects)
            self.coords.append(coords)
        return self

    def extract_samples(self):
        self.samples_dict = {}
        for cube_id, (cube, rects) in enumerate(zip(self.reflectance, self.rect_masks)):
            scene_id = f'scene{cube_id+1:02d}'
            species_list = self.mapping[scene_id]

            samples = hsikit.bg_removal.extract_sample_cubes_from_masks(cube, rects, species_list=species_list)

            for specie, cubes_list in samples.items():
                if specie not in self.samples_dict:
                    self.samples_dict[specie] = []
                self.samples_dict[specie].extend(cubes_list)
        return self

    def summary(self, print_shapes=False):
        print("=== HSI Processor Summary ===")
        print(f"Total cubes loaded: {len(self.cubes)}")
        print(f"Total masks computed: {len(self.masks)}")
        print(f"Total rectangle masks: {len(self.rect_masks)}")

        print("\nSamples per species:")
        for species, cubes in self.samples_dict.items():
            line = f"  {species}: {len(cubes)} samples"
            if print_shapes and cubes:
                unique_shapes = {c.shape for c in cubes}
                line += f", shapes: {unique_shapes}"
            print(line)
        
        print("=============================")

def export_tiff_stack(hsi_cube, filename):
    """
    Exports hyperspectral cube as a TIFF stack.

    Parameters:
    - hsi_cube (3D array): expected shape (h, w, b)
    - filename (str): file name or file path
    """
    transposed_data = np.transpose(hsi_cube, (2, 0, 1))
    tifffile.imwrite(filename+".tif", transposed_data)