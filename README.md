# heliospheric_imager_analysis
---

This module includes functions designed to process and analyse the Heliospheric Imager (HI) data from STEREO-A and STEREO-B. Provided with a path to the Heliospheric Imager data, it provides functions to search for HI1 and HI2 data between set dates, and functions to produce standard images, or differenced images, with optional star field suppression and smoothing. These functions are largely a wrappers around Sunpy maps and other Sunpy features.

When installed, it is necessary to edit `config.dat` to provide a path to where the HI data are stored locally. This assumes the data are stored in the same directory structure as distributed by the UK Solar System Data Center (https://www.ukssdc.ac.uk/solar/stereo/data.html): Level > Background type > Craft > Img > Camera > Day > HI data. If your HI data are not stored in this structure, you may need to edit `images.find_hi_files()` to make it work with your filesystem. 