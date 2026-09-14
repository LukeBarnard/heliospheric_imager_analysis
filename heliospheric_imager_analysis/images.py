"""
This module contains functions for processing and analyzing Heliospheric Imager (HI) data from
the STEREO mission.
"""
from pathlib import Path
import copy
import os
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
import scipy.signal as signal
import sunpy.map as smap
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans
import astropy.units as u
from skimage.measure import label
from sunkit_image.coalignment import phase_cross_correlation_coalign
from sunpy.coordinates import frames, get_body_heliographic_stonyhurst, get_horizons_coord

def find_hi_files(hi_path, t_start, t_stop, craft="sta", camera="hi1", background_type=1):
    """
    Function to find a subset of the STEREO Heliospheric imager data.
    :param hi_path: Path to the top level directory of the STEREO HI data that matches UKSSDC
                    structure.
    :param t_start: Datetime giving start time of the data window requested
    :param t_stop: Datetime giving stop time of the data window requested
    :param craft: String ['sta', 'stb'] to select data from either STEREO-A or STEREO-B.
    :param camera: String ['hi1', 'hi2'] to select data from either HI1 or HI2.
    :param background_type:  Integer [1, 11] to decide between selecting one or eleven day background subtraction.
    :return:
    """
    hi_path = Path(hi_path)

    # Check the input arguments:
    if craft not in {'sta', 'stb'}:
        print("Error: camera should be set to either 'sta', or 'stb'. Defaulting to 'stb'")
        camera = 'sta'

    if camera not in {'hi1', 'hi2'}:
        print("Error: camera should be set to either 'hi1', or 'hi2'. Defaulting to 'hi1'")
        camera = 'hi1'

    if not isinstance(background_type, int):
        print("Error: background_type should be an integer, either 1 or 11. Defaulting to 1")
        background_type = 1

    if background_type not in {1, 11}:
        print("Error: background_type is invalid. Should be either 1, or 11. Defaulting to 1")
        background_type = 1

    # Work out the right directory names to get to the right part of data tree
    background_tag = "L2"

    # Get path up to craft
    if craft == 'sta':
        craft_tag = 'a'
    elif craft == 'stb':
        craft_tag = 'b'

    # Get path up to craft
    if camera == 'hi1':
        camera_tag = 'hi_1'
    elif camera == 'hi2':
        camera_tag = 'hi_2'

    search_path = hi_path / background_tag / craft_tag / "img" / camera_tag

    # Use t_start/stop to get list of days to get data
    day_list = [t.strftime('%Y%m%d') for t in
                pd.date_range(t_start.date(), t_stop.date(), freq='1D')]

    all_files = []
    for day in day_list:
        path = search_path / day
        all_files.extend(path.glob(f"*{background_type:02d}.fts"))

    # Out_files contains all files on dates corresponding to t_start/stop. Now restrict to the exact time window.
    t_min = t_start.strftime('%Y%m%d_%H%M%S')
    t_max = t_stop.strftime('%Y%m%d_%H%M%S')
    out_files = []
    for file_path in all_files:
        # Get the filename without full path
        file_name = os.path.basename(file_path)
        # HI files follow the naming convention of yyyymmdd_hhmmss_datatag.fts.
        # So the first 15 elements give a time string.
        time_tag = file_name[:15]
        if (time_tag >= t_min) and (time_tag <= t_max):
            out_files.append(file_path)

    return out_files


def suppress_star_field(himap, thresh=97.5):
    """
    Function to suppress bright stars in the HI field of view. Is purely data based and does not use
    star-maps. It looks for high-gradient peaks by calculating the Laplacian of the image. This has
    only been developed with HI1 data - unsure how it will behave with HI2.
    :param himap: A sunpy map of the HI image to suppress the star field in.
    :param thresh: Float value containing the percentile threshold used to identify the large
                   gradients associated with stars. This means valid thresh values must lie in the
                   range 0-100, and should normally be high e.g. 97.5.
    :return himap_sm: A HI sunpy map with the star field suppressed.
    """
    # Check inputs
    if not isinstance(thresh, (float, int)):
        print("Error: Invalid thresh, should be float or int. Defaulting to 97.5")
        thresh = 97.5
    elif (thresh < 0) or (thresh > 100):
        print("Error: thresh = {} is invalid, should be in range 0-100. Defaulting to 97.5".format(
            thresh))
        thresh = 97.5

    himap_out = copy.deepcopy(himap)
    img = himap_out.data.copy()
    # Get del2 of image, to find horrendous gradients
    del2 = np.abs(ndimage.laplace(img))
    # Find threshold of data, excluding NaNs
    thresh2 = np.percentile(del2[np.isfinite(del2)], thresh)
    abv_thresh = del2 > thresh2

    # `star_mask` should include the whole star, not just high-Laplacian pixels.
    star_mask = ndimage.binary_dilation(abv_thresh, iterations=2)

    original_bad = np.isnan(img)
    to_fill = star_mask & ~original_bad

    work = img.copy()
    work[to_fill] = np.nan

    # Tune this to the apparent stellar PSF width.
    kernel = Gaussian2DKernel(x_stddev=2)
    out_img = interpolate_replace_nans(work, kernel)

    # Do not interpolate pixels that were invalid in the original image.
    out_img[original_bad] = np.nan

    himap_out = himap._new_instance(out_img, himap.meta.copy())
    return himap_out


def get_approx_star_field(himap, ignore_cmes=False):
    """This function returns a binary array that provides a rough estimate of the locations of stars in the HI1 fov.
     All points above a fixed threshold are 1s, all points below are 0s. Used in the align_image, which is based
    on template matching against the background star-field.
    :param himap: A sunpy map of a HI image.
    :param ignore_cmes: If True, remove contiguous regions covering more than 1% of the image area.
    :return img_stars: A binary image showing estimated locations of stars.
    """
    img_stars = himap.data.copy()
    img_stars[~np.isfinite(img_stars)] = 0
    img_stars[img_stars < np.nanpercentile(img_stars, 97.5)] = 0
    img_stars[img_stars != 0] = 1

    if ignore_cmes:
        labels = label(img_stars.astype(bool))
        region_sizes = np.bincount(labels.ravel())
        large_labels = np.flatnonzero(region_sizes > 0.01 * img_stars.size)
        large_labels = large_labels[large_labels != 0]
        img_stars[np.isin(labels, large_labels)] = 0

    return img_stars


def align_image(src_map, dst_map):
    """
    Function to align two hi images. src_map is shifted by interpolation into the coordinates of dst_map. The
    transformation required to do this is calculated by pattern matching an approximation of the star field between
    frames in a subset of the HI image.
    :param src_map: A SunPy Map of the HI image to shift the coordinates of
    :param dst_map: A SunPy Map of the HI image to match coordinates against
    :return out_img: Array of src_map image shifted into coordinates of dst_map
    """
    stars_src = get_approx_star_field(src_map, ignore_cmes=True)
    stars_dst = get_approx_star_field(dst_map, ignore_cmes=True)

    affine_params = phase_cross_correlation_coalign(stars_src, stars_dst)
    x_shift, y_shift = affine_params.translation
    to_shift = (y_shift, x_shift)

    id_bad = ~np.isfinite(src_map.data)
    src_img = src_map.data.copy()
    src_img[id_bad] = np.nanmedian(src_img)

    # A bilinear output can depend on adjacent source pixels.
    id_bad_for_shift = ndimage.binary_dilation(id_bad, structure=np.ones((3, 3), dtype=bool))

    src_img_shft = ndimage.shift(src_img, to_shift, order=1, mode="constant", cval=np.nan,
                                 prefilter=False)

    id_bad_shft = ndimage.shift(id_bad_for_shift, to_shift, order=0, mode="constant", cval=True,
                                prefilter=False).astype(bool)

    src_img_shft[id_bad_shft] = np.nan

    src_map_out = src_map._new_instance(src_img_shft, src_map.meta)

    return src_map_out


def get_image_plain(hi_file, star_suppress=False):
    """
    A function to load in a HI image file and return this as a SunPy Map object. Will optionally suppress the star field
    using hi_processing.filter_stars().
    :param hi_file: String, full path to a HI image file (in fits format).
    :param star_suppress: Bool, True or False on whether star suppression should be performed. Default false
    :return:
    """
    # Check inputs.
    if not os.path.exists(hi_file):
        print("Error: Path to file does not exist.")

    if not isinstance(star_suppress, bool):
        print("Error: star_suppress should be True or False. Defaulting to False")
        star_suppress = False

    hi_map = smap.Map(hi_file)
    if star_suppress:
        hi_map = suppress_star_field(hi_map)

    return hi_map


def get_image_diff(file_c, file_p, star_suppress=False, align=True, smoothing=False):
    """
    Function to produce a differenced image from HI data. Differenced image is calculated as
    Ic - Ip, loaded from file_c and file_p, respectively. It will optionally perform star field
    suppression and also image alignment. It is currently only configured to do differences of
    consecutive images. Will return a blank frame if images are separated by more than the
    nominal image cadence for hi1 or hi2, or come from different detectors.
    :param file_c: String, full path to the current image file.
    :param file_p: String, full path to the previous image file.
    :param star_suppress: Bool, True or False on whether star suppression should be performed.
                          Default False
    :param align: Bool, True or False depending on whether to align the images before differencing.
    :param smoothing: Bool, True or False depending on whether the differenced image should be
    smoothed with a median filter (5x5)
    :return:
    """
    if not os.path.exists(file_c):
        print("Error: Invalid path to file_c.")

    if not os.path.exists(file_p):
        print("Error: Invalid path to file_p.")

    if not isinstance(star_suppress, bool):
        print("Error: star_suppress should be True or False. Defaulting to False")
        star_suppress = False

    if not isinstance(align, bool):
        print("Error: align should be True or False. Defaulting to False")
        star_suppress = True

    if not isinstance(smoothing, bool):
        print("Error: align should be True or False. Defaulting to False")
        smoothing = False

    hi_c = smap.Map(file_c)

    hi_p = smap.Map(file_p)

    # Set flag to produce diff images, unless data checks fail.
    produce_diff_flag = True

    # Check data from same instrument
    if hi_c.nickname != hi_p.nickname:
        print("Error: Trying to differnece images from {0} and {1}.".format(hi_c.nickname, hi_p.nickname))
        produce_diff_flag = False

    # Check the images are only 1 image apart.
    if hi_c.detector == "HI1":
        # Get typical cadence of HI1 images
        cadence = pd.Timedelta(minutes=40)
        cadence_tol = pd.Timedelta(minutes=5)
    elif hi_c.detector == "HI2":
        # Get typical cadence of HI2 images
        cadence = pd.Timedelta(minutes=120)
        cadence_tol = pd.Timedelta(minutes=5)

    img_dt = hi_c.date - hi_p.date

    if np.abs((img_dt - cadence)) > cadence_tol:
        print("Error: Differenced images time difference is {0}, while typical cadence is {1}.".format(img_dt, cadence))
        print(" Returning a blank frame")
        produce_diff_flag = False

    if produce_diff_flag:
        # Align image p with image c,
        hi_p = align_image(hi_p, hi_c)
        id_bad_p = np.isnan(hi_p.data)
        id_bad_c = np.isnan(hi_c.data)

        if star_suppress:
            hi_c = suppress_star_field(hi_c)
            hi_p = suppress_star_field(hi_p)

        # Get difference image,
        diff_image = hi_c.data - hi_p.data

        # Apply some median smoothing.
        id_bad_diff = np.isnan(diff_image)
        if smoothing:
            diff_image = signal.medfilt2d(diff_image, (5, 5))

        id_bad_all = id_bad_diff | id_bad_c | id_bad_p
        diff_image[id_bad_all] = np.nan
    else:
        diff_image = hi_c.data.copy()*np.nan

    hi_c_diff = hi_c._new_instance(diff_image, hi_c.meta)
    return hi_c_diff


def get_all_hpc_coords(himap):
    """
    Function to get the helioprojective coordinates of a Heliospheric Imager map.
    :param himap: Heliospheric Imager map.
    :return
    lon: longitude
    lat: latitude
    """
    x = np.arange(0, himap.dimensions[0].value, 1) * u.pix
    y = np.arange(0, himap.dimensions[1].value, 1) * u.pix
    x, y = np.meshgrid(x, y)
    hpc = himap.pixel_to_world(x, y)
    return hpc.Tx, hpc.Ty

def get_all_hpr_coords(himap):
    """
    Function to get the helioprojective coordinates of a Heliospheric Imager map.
    :param himap: Heliospheric Imager map.
    :return
    el: Elongation angle
    pa: Position angle
    """
    x = np.arange(0, himap.dimensions[0].value, 1) * u.pix
    y = np.arange(0, himap.dimensions[1].value, 1) * u.pix
    x, y = np.meshgrid(x, y)
    hpc = himap.pixel_to_world(x, y)
    hpr = hpc.transform_to('helioprojectiveradial')

    return hpr.theta, hpr.psi


def get_body_hpr_coord(himap, body):
    """
    Function to get the helioprojective radial coordinates of a body in a Heliospehric Imager map.
    :param himap: A Heliospheric Imager map
    :param body: String name of a body e.g. Earth
    :return
    el: Elongation angle
    pa: Position angle
    """
    planets = ['Mercury', 'Venus', 'Earth', 'Mars']
    missions = ['STEREO-A', 'STEREO-B', 'Parker Solar Probe', 'Solar Orbiter', 'BepiColombo']
    bodies = planets + missions
    if body not in bodies:
        raise ValueError(f"{body} not in list of available bodies: {bodies}")

    if body in planets:
        body = get_body_heliographic_stonyhurst(body, himap.date)
    elif body in missions:
        body = get_horizons_coord(body, himap.date)

    body_hpr = body.transform_to(
        frames.HelioprojectiveRadial(
            observer=himap.coordinate_frame.observer,
            obstime=himap.coordinate_frame.obstime))

    pa = body_hpr.psi.to(u.deg).value
    el = body_hpr.theta.to(u.deg).value
    return el, pa