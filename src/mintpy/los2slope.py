#!/usr/bin/env python3
#########################################################################
# Program is part of MintPy                                             #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi                      #
# Author: Author: Ashok Dahal, Alexander Handwerger, Eric Fielding.     #
# Mar 2024 Based on previous code by Antonio Valentino, Zhang Yunjun.   #
#########################################################################

import numpy as np
from osgeo import gdal
from scipy.ndimage import gaussian_filter

from mintpy.utils import ptime, readfile, writefile

################################################################################


def median_filter(dem, kernel_size):
    """
    Apply median filtering to a given Digital Elevation Model (DEM).

    Parameters:
    - dem: 2D NumPy array representing the DEM
    - kernel_size: Size of the median filter kernel (odd integer)

    Returns:
    - filtered_dem: 2D NumPy array representing the filtered DEM
    """
    # Ensure the kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1

    # Apply median filtering
    filtered_dem = np.zeros_like(dem)
    padded_dem = np.pad(dem, kernel_size // 2, mode="reflect")  # Reflect padding
    for i in range(filtered_dem.shape[0]):
        for j in range(filtered_dem.shape[1]):
            window = padded_dem[i : i + kernel_size, j : j + kernel_size]
            filtered_dem[i, j] = np.median(window)

    return filtered_dem


def smooth_dem(dem, sigma):
    """
    Remove artifacts from a given Digital Elevation Model (DEM) using Gaussian smoothing.

    Parameters:
    - dem: 2D NumPy array representing the DEM
    - sigma: Standard deviation for Gaussian smoothing

    Returns:
    - smoothed_dem: 2D NumPy array representing the gaussian smoothed dem
    """
    # Apply Gaussian smoothing
    smoothed_dem = gaussian_filter(dem, sigma=sigma)

    return smoothed_dem


def get_slp_asp(dem_array, meta):
    # Create a virtual dataset from the NumPy array
    driver = gdal.GetDriverByName("MEM")
    cols, rows = dem_array.shape
    geotransform = (
        float(meta["X_FIRST"]),
        float(meta["X_STEP"]),
        0,
        float(meta["Y_FIRST"]),
        0,
        float(meta["Y_STEP"]),
    )
    dataset = driver.Create("dem.tif", rows, cols, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection("EPSG:4326")
    band = dataset.GetRasterBand(1)
    band.WriteArray(dem_array)

    # Calculate slope, save it as slope.tif as DEMProcessing does not allow mem rasters
    slope_ds = gdal.DEMProcessing("slope.tif", dataset, "slope", scale=111120)
    slope_array = slope_ds.ReadAsArray()

    # Calculate aspect save it as aspect.tif as DEMProcessing does not allow mem rasters
    aspect_ds = gdal.DEMProcessing(
        "aspect.tif",
        dataset,
        "aspect",
        zeroForFlat=True,
    )
    aspect_array = aspect_ds.ReadAsArray()

    # Close all rasters
    slope_ds = None
    aspect_ds = None
    dataset = None
    return slope_array, aspect_array


def get_design_matrix4slope(los_inc_angle, los_az_angle, slope, aspect):
    """Design matrix G to convert asc/desc range displacement into slope direction.
    Only 2D approach is implemented.

    Project displacement from LOS to slope direction components:

    Math for 2D:
        dLOS =   dslope * G
        where, G = (sin(inc_angle) * cos(az_angle - aspect) * cos(slope) + cos(inc_angle) * sin(slope))

    Parameters: los_inc_angle - 1/2D np.ndarray in size of (num_file, window_y, window_x), LOS incidence angle in degree.
                los_az_angle  - 1/2D np.ndarray in size of (num_file, window_y, window_x), LOS azimuth angle in degree, measured in anti-clockwise direction from east.
                slope         - 1/2D np.ndarray, slope of the terrain in degree.
                aspect        - 1/2D np.ndarray, aspect of the terrain in degree, measured in anti-clockwise direction from east.
    Returns:    G             - 2D matrix
    """
    # 2D approach: not currently working, needs math re-check
    # aspect = -aspect
    # vh = np.sin(np.deg2rad(los_inc_angle))*np.cos(np.deg2rad(los_az_angle-aspect))
    # vu = np.cos(np.deg2rad(los_inc_angle))
    # ch = np.cos(np.deg2rad(slope))
    # cu = np.sin(np.deg2rad(slope))
    # G = vh*ch + vu*cu

    # # 3d Approach
    aspect = (
        -aspect + 90
    )  # Shift hillsope aspect to 0 east anti-clockwise, same as the azimuth angle.

    ve = np.sin(np.deg2rad(los_inc_angle)) * np.sin(np.deg2rad(los_az_angle)) * -1
    vn = np.sin(np.deg2rad(los_inc_angle)) * np.cos(np.deg2rad(los_az_angle))
    vu = np.cos(np.deg2rad(los_inc_angle))

    ce = (np.sin(np.deg2rad(aspect)) * np.sin(np.deg2rad(slope))) * -1
    cn = np.sin(np.deg2rad(aspect)) * np.cos(np.deg2rad(slope))
    cu = np.cos(np.deg2rad(slope))

    G = ve * ce + vu * cu + vn * cn
    return G


def los2slope(dlos, los_inc_angle, los_az_angle, slope, aspect, inps, step=20):
    """Project asc / desc LOS data onto downslope direction.
    Parameters: dlos          - 2D np.ndarray in size of (length, width), LOS displacement in meters.
                los_inc_angle - 1/2D np.ndarray in size of (length, width), LOS incidence angle in degree.
                los_az_angle  - 1/2D np.ndarray in size of ( length, width), LOS azimuth   angle in degree, measured in anti-clockwise direction from east.
                slope         - 1/2D np.ndarray, slope of the terrain in degree.
                aspect        - 1/2D np.ndarray, aspect of the terrain in degree, measured in anti-clockwise direction from east.
                slp_thresh    - float [inps], slope angle threshold below which the displacement is masked out
                asp_thresh    - float [inps], hillslope aspect threshold at which aspect the displacement is masked out, usually 0.0, meaning flat location.
                scaling_factor_thresh - float [inps], threshold for multiplication scaling factor in unit vector, this avoids very large, unrealistic values in asymptotic regions.
                step          - int, geometry step size
    Returns:    dslope        - 2D np.ndarray in size of (length, width), slope direction displacement in meters.
    """
    # initiate output
    (length, width) = dlos.shape
    dlos = -dlos  # convert los to make downslope positive
    dslope = np.zeros((length, width), dtype=np.float32) * np.nan
    scale_factor = np.zeros((length, width), dtype=np.float32) * np.nan

    # 2D incidence / azimuth angle --> invert [window-by-window to speed up]
    if los_inc_angle.ndim == 2:
        num_row = np.ceil(length / step).astype(int)
        num_col = np.ceil(width / step).astype(int)

        print(
            f"projecting asc/desc onto slope direction in windows of {step}x{step} ..."
        )
        prog_bar = ptime.progressBar(maxValue=num_row)
        for i in range(num_row):
            y0, y1 = step * i, min(step * (i + 1), length)
            for j in range(num_col):
                x0, x1 = step * j, min(step * (j + 1), width)

                # calculate the median geometry for the local window
                los_inc_angle_window = los_inc_angle[y0:y1, x0:x1]
                los_az_angle_window = los_az_angle[y0:y1, x0:x1]

                # calculate the terrain parameters
                slope_window = slope[y0:y1, x0:x1]
                aspect_window = aspect[y0:y1, x0:x1]

                G = get_design_matrix4slope(
                    los_inc_angle_window,
                    los_az_angle_window,
                    slope_window,
                    aspect_window,
                )
                dlos_window = dlos[y0:y1, x0:x1]
                dslope_window = np.divide(dlos_window, G, where=G != 0.0)

                # create a mask to mask out regions with errors
                slp_mask = slope_window > float(inps.slp_thresh)
                aspect_mask = aspect_window != float(inps.asp_thresh)
                velocity_mask = dlos_window != 0.0
                sf_thresh = 1 / float(inps.scaling_factor_thresh)
                sfactor_mask = np.abs(G) > sf_thresh
                mask = slp_mask & aspect_mask & velocity_mask & sfactor_mask
                dslope_window[~mask] = np.nan

                # store result in original array
                dslope[y0:y1, x0:x1] = dslope_window
                scale_factor[y0:y1, x0:x1] = 1 / G

            prog_bar.update(i + 1, suffix=f"{i+1}/{num_row}")
        prog_bar.close()

    else:
        raise ValueError(
            f"un-supported incidence angle matrix dimension ({los_inc_angle.ndim})!"
        )

    return dslope, scale_factor


def los2slope_ts(dlos, los_inc_angle, los_az_angle, slope, aspect, inps, step=20):
    """Project asc / desc LOS data onto downslope direction.
    Parameters: dlos          - 3D np.ndarray in size of (time, length, width), LOS displacement in meters.
                los_inc_angle - 1/2D np.ndarray in size of (length, width), LOS incidence angle in degree.
                los_az_angle  - 1/2D np.ndarray in size of (length, width), LOS azimuth   angle in degree, measured in anti-clockwise direction from east.
                slope         - 1/2D np.ndarray, slope of the terrain in degree.
                aspect        - 1/2D np.ndarray, aspect of the terrain in degree, measured in anti-clockwise direction from east.
                slp_thresh    - float [inps], slope angle threshold below which the displacement is masked out
                asp_thresh    - float [inps], hillslope aspect threshold at which aspect the displacement is masked out, usually 0.0, meaning flat location.
                scaling_factor_thresh - float [inps], threshold for multiplication scaling factor in unit vector, this avoids very large, unrealistic values in asymptotic regions.
                step          - int, geometry step size
    Returns:    dslope        - 2D np.ndarray in size of (length, width), slope direction displacement in meters.
    """
    # initiate output
    (timestep, length, width) = dlos.shape
    dlos = -dlos  # convert los to make downslope positive
    dslope = np.zeros((timestep, length, width), dtype=np.float32) * np.nan
    scale_factor = np.zeros((length, width), dtype=np.float32) * np.nan
    # 2D incidence / azimuth angle --> invert [window-by-window to speed up]
    if los_inc_angle.ndim == 2:
        num_row = np.ceil(length / step).astype(int)
        num_col = np.ceil(width / step).astype(int)

        print(
            f"projecting asc/desc onto downslope direction in windows of {step}x{step} ..."
        )
        prog_bar = ptime.progressBar(maxValue=num_row)
        for i in range(num_row):
            y0, y1 = step * i, min(step * (i + 1), length)
            for j in range(num_col):
                x0, x1 = step * j, min(step * (j + 1), width)

                # calculate the median geometry for the local window
                los_inc_angle_window = los_inc_angle[y0:y1, x0:x1]
                los_az_angle_window = los_az_angle[y0:y1, x0:x1]

                # calculate the terrain parameters
                slope_window = slope[y0:y1, x0:x1]
                aspect_window = aspect[y0:y1, x0:x1]

                G = get_design_matrix4slope(
                    los_inc_angle_window,
                    los_az_angle_window,
                    slope_window,
                    aspect_window,
                )
                scale_factor[y0:y1, x0:x1] = 1 / G
                # create a mask to mask out regions with errors
                dlos_window = dlos[:, y0:y1, x0:x1]
                G = np.tile(G, (timestep, 1, 1))
                dslope_window = np.divide(dlos_window, G, where=G != 0.0)
                slp_mask = slope_window > float(inps.slp_thresh)
                slp_mask = np.tile(slp_mask, (timestep, 1, 1))
                aspect_mask = aspect_window != float(inps.asp_thresh)
                aspect_mask = np.tile(aspect_mask, (timestep, 1, 1))
                velocity_mask = dlos_window != 0.0
                sf_thresh = 1 / float(inps.scaling_factor_thresh)
                sfactor_mask = np.abs(G) > sf_thresh
                mask = slp_mask & aspect_mask & velocity_mask & sfactor_mask

                dslope_window[~mask] = np.nan
                # store result in original array, take median if multiple input files with overlap exists
                dslope[:, y0:y1, x0:x1] = dslope_window

            prog_bar.update(i + 1, suffix=f"{i+1}/{num_row}")
        prog_bar.close()

    else:
        raise ValueError(
            f"un-supported incidence angle matrix dimension ({los_inc_angle.ndim})!"
        )

    return dslope, scale_factor


def run_los2slope(inps):
    """Project asc / desc LOS files onto downslope direction file(s).
    Parameters: inps         - namespace, input parameters
    Returns:    inps.outfile - str(s) output file(s)
    """

    ## 1. calculate the image size
    atr = readfile.read_attribute(inps.file, datasetName=inps.ds_name)
    f_type = atr["FILE_TYPE"]
    ## 2. read LOS data, elevation and geometry
    dlos = readfile.read(inps.file, datasetName=inps.ds_name)[0]

    if inps.geom_file:
        los_inc_angle = readfile.read(inps.geom_file, datasetName="incidenceAngle")[0]
        los_az_angle = readfile.read(inps.geom_file, datasetName="azimuthAngle")[0]
        dem = readfile.read(inps.geom_file, datasetName="height")[0]
        if inps.median_kernel:
            dem = median_filter(dem, int(inps.median_kernel))
        if inps.gaussian_sigma:
            dem = smooth_dem(dem, float(inps.gaussian_sigma))

        attrs_dem = readfile.read_attribute(inps.geom_file)
        slope, aspect = get_slp_asp(dem, attrs_dem)
        aspect[aspect == -9999] = np.nan
        slope[slope == -9999] = np.nan
    else:
        raise ValueError(
            "Input geometry file must be provided to calculate the slope and aspect"
        )

    ## 3. project LOS displacements onto downslope direction displacements
    print("---------------------")
    # print(f'{dlos.shape}, {los_inc_angle.shape}, {los_az_angle.shape}, {slope.shape}, {aspect.shape},')
    if f_type == "velocity":
        dslope, scale_factor = los2slope(
            dlos, los_inc_angle, los_az_angle, slope, aspect, inps
        )
    elif f_type == "timeseries":
        dslope, scale_factor = los2slope_ts(
            dlos, los_inc_angle, los_az_angle, slope, aspect, inps
        )
    else:
        raise ValueError(
            "Input filetype not supported, only timeseries and velocity data are supported now."
        )

    ## 4. write outputs
    print("---------------------")

    # Save scale factor
    driver = gdal.GetDriverByName("GTiff")
    cols, rows = dem.shape
    geotransform = (
        float(attrs_dem["X_FIRST"]),
        float(attrs_dem["X_STEP"]),
        0,
        float(attrs_dem["Y_FIRST"]),
        0,
        float(attrs_dem["Y_STEP"]),
    )
    dataset = driver.Create("scale_factor.tif", rows, cols, 1, gdal.GDT_Float32)
    dataset.SetGeoTransform(geotransform)
    dataset.SetProjection("EPSG:4326")
    band = dataset.GetRasterBand(1)
    band.WriteArray(scale_factor)
    dataset = None

    # Update attributes
    atr = atr.copy()
    if inps.ds_name and atr["FILE_TYPE"] in ["ifgramStack", "timeseries", "HDFEOS"]:
        atr["FILE_TYPE"] = "displacement"

    # use ref_file for time-series file writing
    ref_file = inps.file if atr["FILE_TYPE"] == "timeseries" else None

    print(f"write asc/desc/slope datasets into {inps.outfile}")
    dsDict = {}
    if f_type == "velocity":
        dsDict["velocitySlope"] = dslope
    elif f_type == "timeseries":
        dsDict["displacementSlope"] = dslope

    writefile.write(dsDict, out_file=inps.outfile, metadata=atr, ref_file=ref_file)

    return inps.outfile
