# TODO

This is a todo file for Python package **hsikit**.

## New content

Tasks related to new content and functionality.

### Related to *temp_bg_classes.py*
- [ ] Add other automatic masking options  
- [ ] Add the option to not use mapping file/different kind of mapping file/list of labels  
- [ ] Add an option to either return dictionary or directly X matrix (n_samples, n_features)  
- [ ] Add subsampling options such as 5x5 or 10x10 pixel mean/median aggregation  

### Related to *preprocessing.py*
- [ ] Add {preprocessing} options (SNV, MSC, SG 1st and 2nd derivative)  

### Related to *masking_utility.py*
- [ ] Add manual interactive masking options (pixel/wider pixel area eg. 5x5, rectangle, polygon)  

### Other
- [ ] Outlier and dead pixel detection/removal

## Fixes

Tasks related to fixes, cleaning and organization.

### Related to *temp_bg_classes.py*
- [ ] Clean and organize temp_bg_classes.py - merge into one class  

### Related to *base_utils.py*
- [ ] Consider renaming/reorganizing into two modules (utils + visualization)  
- [ ] Possibly remove certain utility functions that aren't vital/useful  

### Related to *README.md*
- [ ] Write an actual README.md file / restructure it  

## DONE

- [x] Split masking into *binary_masks.py* and *masking_utility.py*  
- [x] Rename functions from "import..." to "load..."  
- [x] Create GitHub organization todo.md  
- [x] Add requirements.txt  
- [x] Init project repo and publish project on GitHub (link: [text](https://github.com/LukaKopac/hsikit))  