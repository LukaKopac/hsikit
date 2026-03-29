"""
Pipelines for background removal and sample extraction.

Note: This module is under active development and may change.
"""

import numpy as np

from pathlib import Path
import re
import os

from hsikit.hsi_io import load_hsi_batch, find_hsi_basepaths, load_sample_mapping, load_hsi_raw, load_wavelengths
from hsikit.binary_masks import mask_top_contrast, mask_top_contrastV2, fixed_rect_extraction
from hsikit.masking_utility import extract_sample_cubes_from_masks



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

    def batch_load(self, return_metadata=False, return_wavelengths=False, auto_mapping=True):
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
            mapping = load_sample_mapping(self.mapping_file)
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
                cube, meta = load_hsi_raw(base, return_metadata=True)
                self.metadata[scene_name] = meta
            else:
                cube = load_hsi_raw(base, return_metadata=False)
                meta = {}

            self.cubes[scene_name] = cube
            self.scenes[scene_name] = {
                "cube": cube,
                "metadata": meta,
                "mapping": mapping.get(scene_name, []),
                "masks": [],
                "combined_mask": None
            }
            print(f"Loaded: {basename} -> shape {cube.shape}")

            if scene_name in mapping:
                print(f" → Mapping found for {scene_name} ({len(mapping[scene_name])} entries)")
            else:
                print(f" → No mapping for {scene_name}")

        # load wavelengths (only once)
        if return_wavelengths and self.basepaths:
            self.wavelengths = load_wavelengths(self.basepaths[0] + ".hdr")
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
        self.meta = {} # metadata per cube
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
        self.cubes, self.meta, self.wl = load_hsi_batch(
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

    def compute_masks(self, min_size=1800, manual_max_band=None, visualize=False):
        self.masks = [
            mask_top_contrast(c, min_size=min_size, manual_max_band=manual_max_band, visualize=visualize)
            for c in self.reflectance
        ]
        return self

    def add_rectangles(self, width=60, height=140, min_frac=0.9):
        self.rect_masks = []
        self.coords = []
        for mask in self.masks:
            rects, coords = fixed_rect_extraction(mask, (width, height), mode='column', min_frac=min_frac)
            self.rect_masks.append(rects)
            self.coords.append(coords)
        return self

    def extract_samples(self):
        self.samples_dict = {}

        filenames = list(self.meta.keys())
        
        for cube, rects, filename in zip(self.reflectance, self.rect_masks, filenames):
            match = re.search(r'scene\d+', filename)
            if not match:
                raise ValueError(f'Could not extract scene ID from filename: {filename}')
            scene_id = match.group(0)

            if scene_id not in self.mapping:
                raise KeyError(f"{scene_id} not found in mapping file (from filename '{filename}')")
            
            species_list = self.mapping[scene_id]
            samples = extract_sample_cubes_from_masks(cube, rects, species_list=species_list)

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

        print("=============================")
        print("\nSamples per species:")
        for species, cubes in self.samples_dict.items():
            line = f"  {species}: {len(cubes)} samples"
            if print_shapes and cubes:
                unique_shapes = {c.shape for c in cubes}
                line += f", shapes: {unique_shapes}"
            print(line)
        
        print("=============================")

class HSIProcessorV2:
    """
    High-level pipeline for loading, preprocessing, masking, and extracting samples
    from a collection of hyperspectral image (HSI) cubes.

    This class provides an end-to-end workflow for hyperspectral data stored in a
    folder, including:

    - Batch loading of raw hyperspectral cubes and metadata.
    - Parsing a scene-to-species mapping file (abbreviation expansion + scene blocks).
    - Conversion of raw radiance cubes to reflectance.
    - Automatic mask generation using a top-contrast heuristic.
    - Placement of fixed rectangular masks for sampling (column-wise layout).
    - Extraction of per-species sample cubes based on scene mapping.
    - Summary reporting.

    The processor assumes a consistent folder structure and a “mapping” file that
    defines species assignments for each scene. The resulting `samples_dict` is
    suitable for downstream machine-learning tasks (e.g., pixel-wise or patch-based
    classification).

    Parameters
    ----------
    folder : str or Path
        Directory containing hyperspectral scenes. Each scene must be importable by
        `hsikit.io.batch_import_hsi`.
    mapping_file : str or Path, optional
        Path to a text mapping file that associates scene IDs (e.g. “scene01”) with
        species labels. If not provided, the processor searches `folder` for a file
        whose name begins with ``mapping``.

    Attributes
    ----------
    folder : Path
        Root directory of the hyperspectral dataset.
    cubes : list of np.ndarray
        Raw hyperspectral cubes loaded from disk.
    meta : list of dict
        Metadata objects returned by the importer for each cube.
    wl : np.ndarray
        Wavelengths shared across all imported cubes.
    reflectance : list of np.ndarray
        Reflectance-converted hyperspectral cubes (same ordering as `cubes`).
    masks : list of np.ndarray
        Binary foreground masks computed per cube.
    rect_masks : list of list of np.ndarray
        For each cube, a list of binary rectangle masks generated by
        `fixed_rect_mask_columnwise`.
    coords : list of list of tuple
        Coordinates associated with each set of rectangular masks (as returned by
        `fixed_rect_mask_columnwise`).
    mapping_file : Path
        Path to the species mapping file.
    translation : dict
        Mapping of abbreviations to full species names parsed from the mapping file.
    mapping : dict
        Dictionary of scene IDs → list of species names, derived from the mapping file.
    samples_dict : dict[str, list[np.ndarray]]
        Dictionary of species → list of extracted sample cubes.

    Methods
    -------
    load(suffix='refl')
        Imports all scenes from the folder, reading both the cubes and metadata.
    load_mapping()
        Parses the species mapping file into `translation` and `mapping`.
    compute_masks(min_size=1800, ycrop=0, xcrop=0, manual_max_band=None, visualize=False)
        Computes one foreground mask per cube using a top-contrast heuristic.
    add_rectangles(width=50, height=90, min_frac=0.9)
        Generates fixed rectangular sample regions for each cube mask.
    extract_samples()
        Extracts per-species hyperspectral samples according to the mapping file and
        accumulated rectangular masks.
    summary(print_shapes=False)
        Prints counts and (optionally) shape information for extracted samples.

    Notes
    -----
    The typical workflow is:

    >>> processor = (
    ...     HSIProcessorV2(folder)
    ...     .load()
    ...     .load_mapping()
    ...     .compute_masks()
    ...     .add_rectangles()
    ...     .extract_samples()
    ... )
    >>> processor.summary()

    All returned methods follow a fluent interface pattern (returning ``self``),
    enabling chained workflows.
    """
    def __init__(self, folder, mapping_file=None):
        self.folder = Path(folder)
        self.cubes = [] # list of np.arrays (raw data)
        self.cube_names = []
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
        self.cubes, self.meta, self.wl, self.cube_names = load_hsi_batch(
            root_folder=self.folder,
            suffix=suffix,
            return_wavelengths=True,
            return_metadata=True,
            return_names=True
        )
        assert len(self.cubes) == len(self.cube_names)
        
        return self

    def load_mapping(self):
        with open(self.mapping_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        if lines and '=' in lines[0]:
            for pair in lines[0].split(','):
                if '=' in pair:
                    k, v = pair.split('=', 1)
                    self.translation[k.strip()] = v.strip()
            lines = lines[1:]

        for line in lines:
            if ':' not in line:
                continue
            scene_id_part, species_part = line.split(':', 1)
            scene_id = scene_id_part.strip()
            species_list = [self.translation.get(word.strip(), word.strip()) for word in species_part.split(',')]
            self.mapping[scene_id] = species_list

        return self

    def compute_masks(self, min_size=1800, ycrop=0, xcrop=0, manual_max_band=None, visualize=False):
        self.masks = [
            mask_top_contrastV2(c, min_size=min_size, ycrop=ycrop, xcrop=xcrop, manual_max_band=manual_max_band, visualize=visualize, cube_name=name)
            for c, name in zip(self.reflectance, self.cube_names)
        ]
        return self

    def add_rectangles(self, width=50, height=90, min_frac=0.9):
        self.rect_masks = []
        self.coords = []
        for mask in self.masks:
            rects, coords = fixed_rect_extraction(mask, (width, height), mode='column', min_frac=min_frac)
            self.rect_masks.append(rects)
            self.coords.append(coords)
        return self

    def extract_samples(self):
        self.samples_dict = {}

        for cube, rects, scene_id in zip(self.reflectance, self.rect_masks, self.cube_names):
            species_list = self.mapping[scene_id]

            samples = extract_sample_cubes_from_masks(cube, rects, species_list=species_list)

            for specie, cubes_list in samples.items():
                self.samples_dict.setdefault(specie, []).extend(cubes_list)

        return self

    def summary(self, print_shapes=False):
        print("=== HSI Processor Summary ===")
        print(f"Total cubes loaded: {len(self.cubes)}")
        print(f"Total masks computed: {len(self.masks)}")
        print(f"Total rectangle masks: {len(self.rect_masks)}")

        print("=============================")
        print("\nSamples per species:")
        for species, cubes in self.samples_dict.items():
            line = f"  {species}: {len(cubes)} samples"
            if print_shapes and cubes:
                unique_shapes = {c.shape for c in cubes}
                line += f", shapes: {unique_shapes}"
            print(line)
        
        print("=============================")