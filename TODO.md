# TODO

This is a todo file for Python package **hsikit**.

## New content

Tasks related to new content and functionality.

### Related to *temp_bg_classes.py*
- [ ] Add use of other automatic masking options  
- [ ] Add the option to not use mapping file or use a different kind of mapping file/list of labels  

### Related to *binary_masks.py*
- [ ] Add manual interactive masking options (pixel/wider pixel area eg. 5x5, rectangle, polygon)  

## Fixes

Tasks related to fixes, cleaning and organization.

### General
- [ ] **Review and organize *sandbox.py* functions/utility**  
- [ ] Reorganize the package structure - supporting/internal folders and modules  

### Related to *temp_bg_classes.py*
- [ ] **Review and organize temp_bg_classes.py - merge into single class**   

## DONE

- [x] Add a module that focuses on data cleaning/cleansing
- [x] **Outlier and dead pixel detection/removal**  
- [x] Add cube subsampling options such as 5x5 or 10x10 pixel mean/median aggregation  
- [x] Convert dictionary of cubes to X (n_samples, n_features) and y (n_samples) matrices (keep track of individual samples)  
- [x] Add shadow percentile option to SAM mask
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
