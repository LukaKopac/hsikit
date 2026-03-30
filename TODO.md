# TODO

This is a todo file for Python package **hsikit**.

## New content

Tasks related to new content and functionality.

### Related to *temp_bg_classes.py*
- [ ] Add use of other automatic masking options  
- [ ] Add the option to not use mapping file/different kind of mapping file/list of labels  

### Related to *binary_masks.py*
- [ ] **Add shadow percentile option to SAM mask**
- [ ] **Add manual interactive masking options (pixel/wider pixel area eg. 5x5, rectangle, polygon)**  

### Other
- [ ] **Outlier and dead pixel detection/removal (*base_utils.py*)**  
- [ ] Add cube subsampling options such as 5x5 or 10x10 pixel mean/median aggregation  
- [ ] Convert dictionary of cubes to X (n_samples, n_features) and y (n_samples) matrices (keep track of individual samples)  

## Fixes

Tasks related to fixes, cleaning and organization.

### General
- [ ] Reorganize the package structure - supporting/internal folders and modules  

### Related to *temp_bg_classes.py*
- [ ] **Clean and organize temp_bg_classes.py - merge into single class**  

### Related to *base_utils.py*
- [ ] Consider renaming/reorganizing into two modules (utils + visualization)  

## DONE

- [x] Add SAM masking function  
- [x] Direct spectral extraction (mean/median per area)  
- [x] Add {preprocessing} options (SNV, MSC, SG 1st and 2nd derivative)  
- [x] Restructure plotting functions - add (fig, ax) return  
- [x] Full cube min-max normalization (instead of per surface) in plot_hsi_cube() function  
- [x] Add docstrings and type hints for functions  
- [x] Split masking into *binary_masks.py* and *masking_utility.py*  
- [x] Rename functions from "import..." to "load..."  
- [x] Create GitHub organization TODO.md  
- [x] Add requirements.txt  
- [x] Init project repo and publish project on GitHub (link: [hsikit](https://github.com/LukaKopac/hsikit))  
