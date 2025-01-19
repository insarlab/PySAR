"""Utilities to read files."""
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, 2013               #
############################################################
# Recommend import:
#   from mintpy.utils import readfile


import datetime as dt
import glob
import os
import re
import sys
import warnings
import xml.etree.ElementTree as ET
from typing import Union

import h5py
import numpy as np
from numpy.typing import DTypeLike

from mintpy.objects import (
    DSET_UNIT_DICT,
    HDFEOS,
    geometry,
    giantIfgramStack,
    giantTimeseries,
    ifgramStack,
    sensor,
    timeseries,
)
from mintpy.utils import ptime, utils0 as ut

SPEED_OF_LIGHT = 299792458  # meters per second


STD_METADATA_KEYS = {
    # ROI_PAC/MintPy attributes
    'ALOOKS'             : ['azimuth_looks'],
    'RLOOKS'             : ['range_looks'],
    'AZIMUTH_PIXEL_SIZE' : ['azimuthPixelSize', 'azimuth_pixel_spacing', 'az_pixel_spacing', 'azimuth_spacing'],
    'RANGE_PIXEL_SIZE'   : ['rangePixelSize', 'range_pixel_spacing', 'rg_pixel_spacing', 'range_spacing'],
    'CENTER_LINE_UTC'    : ['center_time'],
    'DATA_TYPE'          : ['dataType', 'data_type', 'image_format'],
    'EARTH_RADIUS'       : ['earthRadius', 'earth_radius_below_sensor', 'earth_radius'],
    'HEADING'            : ['HEADING_DEG', 'heading', 'centre_heading'],
    'HEIGHT'             : ['altitude', 'SC_height'],
    'BANDS'              : ['number_bands', 'bands'],
    'INTERLEAVE'         : ['scheme', 'interleave'],
    'LENGTH'             : ['length', 'FILE_LENGTH', 'lines', 'azimuth_lines', 'nlines', 'az_samp',
                            'interferogram_azimuth_lines', 'num_output_lines'],
    'LAT_REF1'           : ['first_near_lat'],
    'LON_REF1'           : ['first_near_long'],
    'LAT_REF2'           : ['first_far_lat'],
    'LON_REF2'           : ['first_far_long'],
    'LAT_REF3'           : ['last_near_lat'],
    'LON_REF3'           : ['last_near_long'],
    'LAT_REF4'           : ['last_far_lat'],
    'LON_REF4'           : ['last_far_long'],
    'ORBIT_DIRECTION'    : ['passDirection', 'pass'],
    'NO_DATA_VALUE'      : ['NoDataValue'],
    'PLATFORM'           : ['spacecraftName', 'sensor', 'mission'],
    'POLARIZATION'       : ['polarization'],
    'PRF'                : ['prf', 'pulse_repetition_frequency'],
    'STARTING_RANGE'     : ['startingRange', 'near_range_slc', 'near_range', 'slant_range_to_first_pixel'],
    'WAVELENGTH'         : ['wavelength', 'Wavelength', 'radarWavelength', 'radar_wavelength'],
    'WIDTH'              : ['width', 'Width', 'samples', 'range_samp', 'interferogram_width', 'num_samples_per_line'],
    # from PySAR [MintPy<=1.1.1]
    'REF_DATE'           : ['ref_date'],
    'REF_LAT'            : ['ref_lat'],
    'REF_LON'            : ['ref_lon'],
    'REF_X'              : ['ref_x'],
    'REF_Y'              : ['ref_y'],
    'SUBSET_XMIN'        : ['subset_x0'],
    'SUBSET_XMAX'        : ['subset_x1'],
    'SUBSET_YMIN'        : ['subset_y0'],
    'SUBSET_YMAX'        : ['subset_y1'],
    # from Gamma geo-coordinates - degree / meter
    'X_FIRST'            : ['corner_lon', 'corner_east'],
    'Y_FIRST'            : ['corner_lat', 'corner_north'],
    'X_STEP'             : ['post_lon', 'post_east'],
    'Y_STEP'             : ['post_lat', 'post_north'],

    # HDF-EOS5 attributes
    'beam_swath'     : ['swathNumber'],
    'first_frame'    : ['firstFrameNumber'],
    'last_frame'     : ['lastFrameNumber'],
    'relative_orbit' : ['trackNumber'],
}

## data type conventions [use numpy as reference]
# 1 - ENVI
# reference: https://subversion.renater.fr/efidir/trunk/efidir_soft/doc/Programming_C_EFIDIR/header_envi.pdf
DATA_TYPE_ENVI2NUMPY = {
    '1' : 'uint8',
    '2' : 'int16',
    '3' : 'int32',
    '4' : 'float32',
    '5' : 'float64',
    '6' : 'complex64',
    '9' : 'complex128',
    '12': 'uint16',
    '13': 'uint32',
    '14': 'int64',
    '15': 'uint64',
}

DATA_TYPE_NUMPY2ENVI = {
    'uint8'     : '1',
    'int16'     : '2',
    'int32'     : '3',
    'float32'   : '4',
    'float64'   : '5',
    'complex64' : '6',
    'complex128': '9',
    'uint16'    : '12',
    'uint32'    : '13',
    'int64'     : '14',
    'uint64'    : '15',
}

# 2 - GDAL
# link: https://gdal.org/api/raster_c_api.html#_CPPv412GDALDataType
DATA_TYPE_GDAL2NUMPY = {
    1 : 'uint8',
    2 : 'uint16',
    3 : 'int16',
    4 : 'uint32',
    5 : 'int32',
    6 : 'float32',
    7 : 'float64',
    8 : 'cint16',       # for translation purpose only, as numpy does not support complex int
    9 : 'cint32',       # for translation purpose only, as numpy does not support complex int
    10: 'complex64',
    11: 'complex128',
    12: 'uint64',
    13: 'int64',
    14: 'int8',         # GDAL >= 3.7
}

DATA_TYPE_NUMPY2GDAL = {
    "uint8"     : 1,    # GDT_Byte
    "uint16"    : 2,    # GDT_UInt16
    "int16"     : 3,    # GDT_Int16
    "uint32"    : 4,    # GDT_UInt32
    "int32"     : 5,    # GDT_Int32
    "float32"   : 6,    # GDT_Float32
    "float64"   : 7,    # GDT_Float64
    "cint16"    : 8,    # GDT_CInt16, for translation purpose only, as numpy does not support complex int
    "cint32"    : 9,    # GDT_CInt32, for translation purpose only, as numpy does not support complex int
    "complex64" : 10,   # GDT_CFloat32
    "complex128": 11,   # GDT_CFloat64
    "uint64"    : 12,   # GDT_UInt64 (GDAL >= 3.5)
    "int64"     : 13,   # GDT_Int64  (GDAL >= 3.5)
    "int8"      : 14,   # GDT_Int8   (GDAL >= 3.7)
}

# 3 - ISCE
DATA_TYPE_ISCE2NUMPY = {
    'byte'  : 'uint8',
    'short' : 'int16',
    'int'   : 'int32',
    'float' : 'float32',
    'double': 'float64',
    'cfloat': 'complex64',
}

DATA_TYPE_NUMPY2ISCE = {
    'uint8'    : 'BYTE',
    'int16'    : 'SHORT',
    'int32'    : 'INT',
    'float32'  : 'FLOAT',
    'float64'  : 'DOUBLE',
    'complex64': 'CFLOAT',
}

# 4 - GAMMA
DATA_TYPE_GAMMA2NUMPY = {
    'fcomplex' : 'complex64',
    'scomplex' : 'complex32',
}

# single file (data + attributes) supported by GDAL
# .cos file - TerraSAR-X complex SAR data (https://gdal.org/drivers/raster/cosar.html)
GDAL_FILE_EXTS = ['.tiff', '.tif', '.grd', '.cos']

ENVI_BAND_INTERLEAVE = {
    'BAND' : 'BSQ',
    'LINE' : 'BIL',
    'PIXEL': 'BIP',
}

ENVI_BYTE_ORDER = {
    '0': 'little-endian',
    '1': 'big-endian',
}


SPECIAL_STR2NUM = {
    'yes'   : True,
    'true'  : True,
    'no'    : False,
    'false' : False,
    'none'  : None,
    'nan'   : np.nan,
}

GMTSAR_SENSOR_ID2NAME = {
    1 : 'ers',
    2 : 'ers',
    3 : 'rs1',
    4 : 'env',
    5 : 'alos',
    6 : 'tsx',
    7 : 'csk',
    8 : 'csk',
    9 : 'rs2',
    10: 'sen',
    11: 'gf3',
    12: 'lt1',
}


#########################################################################
def numpy_to_gdal_dtype(np_dtype: DTypeLike) -> int:
    """Convert NumPy dtype to GDAL dtype.

    Modified from dolphin.utils.numpy_to_gdal_type().

    Parameters: np_dtype  - DTypeLike, NumPy dtype to convert.
    Returns:    gdal_code - int, GDAL type code corresponding to `np_dtype`.
    """
    from osgeo import gdal_array, gdalconst
    np_dtype = np.dtype(np_dtype)

    # convert dtype using the GDAL function/attribute
    if np.issubdtype(bool, np_dtype):
        gdal_code = gdalconst.GDT_Byte
    else:
        gdal_code = gdal_array.NumericTypeCodeToGDALTypeCode(np_dtype)

    # if provided dtype is not support GDAL, e.g. np.dtype('>i4')
    if gdal_code is None:
        raise TypeError(f"dtype {np_dtype} not supported by GDAL.")

    return gdal_code


def gdal_to_numpy_dtype(gdal_dtype: Union[str, int]) -> np.dtype:
    """Convert GDAL dtype to NumPy dtype.

    Modified from dolphin.utils.gdal_to_numpy_type().

    Parameters: gdal_dtype - str/int, GDAL dtype to convert.
    Returns:    np_dtype   - np.dtype, NumPy dtype
    """
    from osgeo import gdal, gdal_array
    if isinstance(gdal_dtype, str):
        gdal_dtype = gdal.GetDataTypeByName(gdal_dtype)
    np_dtype = np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_dtype))
    return np_dtype


#########################################################################
## Slice-based data identification for data reading:
##
## slice   : np.ndarray in 2D,  with/without '-' in their names
##           each slice is unique within a file
## dataset : np.ndarray in 2D or 3D, without '-' in their names
##
## one 2D dataset can be present as one slice
##   e.g.: temporalCoherence
##         velocity
##         mask
##         ...
## one 3D dataset can be present as multiple slices
##   with '-' to connect dataset name and time info
##   e.g.: unwrapPhase-20150115_20150127
##         unwrapPhase-20150115_20150208
##         ...
##         timeseries-20150115
##         timeseries-20150127
##         ...
## one HDF5 file can be present as a combination of multiple 2D/3D datasets
##                           or as a list of slices
##
## ----------------------- Slice Nameing Examples -------------------------
## for version-1.x files:
##     unwrapPhase-20150115_20150127
##     unwrapPhase-20150115_20150208
##     timeseries-20150115
##     timeseries-20150127
##     temporalCoherence
##
## for version-0.x files: (all in 2D dataset)
##     /interferograms/diff_filt_100814-100918_4rlks.unw/diff_filt_100814-100918_4rlks.unw
##     /coherence/diff_filt_100814-100918_4rlks.cor/diff_filt_100814-100918_4rlks.cor
##     /timeseries/20100918
##     /timeseries/20101103
##     /velocity/velocity
##
## for HDF-EOS5 files:
##     /HDFEOS/GRIDS/timeseries/observation/displacement-20150115
##     /HDFEOS/GRIDS/timeseries/observation/displacement-20150127
##     /HDFEOS/GRIDS/timeseries/quality/temporalCoherence
##
## for GIAnT v1.0 files:
##     figram-20150115_20150127
##     recons-20150115
##     cmask
##
#########################################################################



#########################################################################
def read(fname, box=None, datasetName=None, print_msg=True, xstep=1, ystep=1, data_type=None,
         no_data_values=None):
    """Read one dataset and its attributes from input file.

    Parameters: fname          - str, path of file to read
                datasetName    - str or list of str, slice names
                box            - 4-tuple of int area to read, defined in (x0, y0, x1, y1) in pixel coordinate
                x/ystep        - int, number of pixels to pick/multilook for each output pixel
                data_type      - numpy data type, e.g. np.float32, np.bool_, etc. Change the output data type
                no_data_values - list of 2 numbers, change the no-data-value in the output
    Returns:    data           - 2/3/4D matrix in numpy.array format, return None if failed
                atr            - dictionary, attributes of data, return None if failed
    Examples:
        from mintpy.utils import readfile
        data, atr = readfile.read('velocity.h5')
        data, atr = readfile.read('timeseries.h5')
        data, atr = readfile.read('timeseries.h5', datasetName='timeseries-20161020')
        data, atr = readfile.read('ifgramStack.h5', datasetName='unwrapPhase')
        data, atr = readfile.read('ifgramStack.h5', datasetName='unwrapPhase-20161020_20161026')
        data, atr = readfile.read('ifgramStack.h5', datasetName='coherence', box=(100,1100, 500, 2500))
        data, atr = readfile.read('geometryRadar.h5', datasetName='height')
        data, atr = readfile.read('geometryRadar.h5', datasetName='bperp')
        data, atr = readfile.read('100120-110214.unw', box=(100,1100, 500, 2500))
    """
    fname = os.fspath(fname)  # Convert from possible pathlib.Path
    # metadata
    dsname4atr = None   #used to determine UNIT
    if isinstance(datasetName, list):
        dsname4atr = datasetName[0].split('-')[0]
    elif isinstance(datasetName, str):
        dsname4atr = datasetName.split('-')[0]
    atr = read_attribute(fname, datasetName=dsname4atr)

    # box
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])
    if not box:
        box = (0, 0, width, length)

    # read data
    kwargs = dict(
        datasetName=datasetName,
        box=box,
        xstep=xstep,
        ystep=ystep,
    )

    fext = os.path.splitext(os.path.basename(fname))[1].lower()
    if fext in ['.h5', '.he5']:
        data = read_hdf5_file(fname, print_msg=print_msg, **kwargs)

    else:
        data, atr = read_binary_file(fname, **kwargs)

    # customized output data type
    if data_type is not None and data_type != data.dtype:
        if print_msg:
            print(f'convert numpy array from {data.dtype} to {data_type}')
        data = np.array(data, dtype=data_type)

    # convert no-data-value
    if isinstance(no_data_values, list):
        if print_msg:
            print(f'convert no-data-value from {no_data_values[0]} to {no_data_values[1]}')
        data[data == no_data_values[0]] = no_data_values[1]

    return data, atr


#########################################################################
def read_hdf5_file(fname, datasetName=None, box=None, xstep=1, ystep=1, print_msg=True):
    """
    Parameters: fname       : str, name of HDF5 file to read
                datasetName : str or list of str, dataset name in root level with/without date info
                    'timeseries'
                    'timeseries-20150215'
                    'unwrapPhase'
                    'unwrapPhase-20150215_20150227'
                    'HDFEOS/GRIDS/timeseries/observation/displacement'
                    'recons'
                    'recons-20150215'
                    ['recons-20150215', 'recons-20150227', ...]
                    '20150215'
                    'cmask'
                    'igram-20150215_20150227'
                    ...
                box         : 4-tuple of int area to read, defined in (x0, y0, x1, y1) in pixel coordinate
                x/ystep     : int, number of pixels to pick/multilook for each output pixel
    Returns:    data        : 2/3/4D array
                atr         : dict, metadata
    """
    # File Info: list of slice / dataset / dataset2d / dataset3d
    slice_list = get_slice_list(fname)
    ds_list = []
    for i in [i.split('-')[0] for i in slice_list]:
        if i not in ds_list:
            ds_list.append(i)
    ds_2d_list = [i for i in slice_list if '-' not in i]
    ds_3d_list = [i for i in ds_list if i not in ds_2d_list]

    # Input Argument: convert input datasetName into list of slice
    if not datasetName:
        datasetName = [ds_list[0]]
    elif isinstance(datasetName, str):
        datasetName = [datasetName]

    # if datasetName is all date info, add dsFamily as prefix
    # a) if all digit, e.g. YYYYMMDD
    # b) if in isoformat(), YYYY-MM-DDTHH:MM, etc.
    if all(x.isdigit() or x[:4].isdigit() for x in datasetName):
        datasetName = [f'{ds_3d_list[0]}-{x}' for x in datasetName]

    # Input Argument: decompose slice list into dsFamily and inputDateList
    dsFamily = datasetName[0].split('-')[0]
    inputDateList = [x.replace(dsFamily,'') for x in datasetName]
    inputDateList = [x[1:] for x in inputDateList if x.startswith('-')]

    # read hdf5
    with h5py.File(fname, 'r') as f:
        # get dataset object
        dsNames = [i for i in [datasetName[0], dsFamily] if i in f.keys()]
        # support for old mintpy-v0.x files
        dsNamesOld = [i for i in slice_list if f'/{datasetName[0]}' in i]
        if len(dsNames) > 0:
            ds = f[dsNames[0]]
        elif len(dsNamesOld) > 0:
            ds = f[dsNamesOld[0]]
        else:
            raise ValueError(f'input dataset {datasetName} not found in file {fname}')

        # output size for >=2D dataset if x/ystep > 1
        xsize = int((box[2] - box[0]) / xstep)
        ysize = int((box[3] - box[1]) / ystep)

        # 2D dataset
        if ds.ndim == 2:
            # read data
            data = ds[box[1]:box[3],
                      box[0]:box[2]]

            # sampling / nearest interplation in y/xstep
            if xstep * ystep > 1:
                data = data[int(ystep/2)::ystep,
                            int(xstep/2)::xstep]
                data = data[:ysize, :xsize]

        # 3D dataset
        elif ds.ndim == 3:
            # define flag matrix for index in time domain
            slice_flag = np.zeros((ds.shape[0]), dtype=np.bool_)
            if not inputDateList or inputDateList == ['']:
                slice_flag[:] = True
            else:
                date_list = [i.split('-', 1)[1] for i in
                             [j for j in slice_list if j.startswith(dsFamily)]]
                for d in inputDateList:
                    slice_flag[date_list.index(d)] = True

            # read data
            num_slice = np.sum(slice_flag)
            inds = np.where(slice_flag)[0].tolist()

            if xstep * ystep == 1:
                if num_slice / slice_flag.size < 0.05:
                    # single indexing if only a small fraction is read
                    data = np.zeros((num_slice, ysize, xsize), dtype=ds.dtype)
                    for i, ind in enumerate(inds):
                        data[i] = ds[ind,
                                     box[1]:box[3],
                                     box[0]:box[2]]
                else:
                    data = ds[:,
                              box[1]:box[3],
                              box[0]:box[2]][slice_flag]

            else:
                # sampling / nearest interplation in y/xstep
                # use for loop to save memory
                data = np.zeros((num_slice, ysize, xsize), ds.dtype)

                for i, ind in enumerate(inds):
                    # print out msg
                    if print_msg:
                        sys.stdout.write('\r' + f'reading 2D slices {i+1}/{num_slice}...')
                        sys.stdout.flush()

                    # read and index
                    d2 = ds[ind,
                            box[1]:box[3],
                            box[0]:box[2]]
                    d2 = d2[int(ystep/2)::ystep,
                            int(xstep/2)::xstep]
                    data[i, :, :] = d2[:ysize, :xsize]

                if print_msg:
                    print('')

            if any(i == 1 for i in data.shape):
                data = np.squeeze(data)

        # 4D dataset
        elif ds.ndim == 4:
            # custom flag for the time domain is ignore for now
            # a.k.a. read the entire first 2 dimensions

            num1, num2 = ds.shape[0], ds.shape[1]
            shape = (num1, num2, ysize, xsize)
            if print_msg:
                ram_size = num1 * num2 * ysize * xsize * ds.dtype.itemsize / 1024**3
                print(f'initiate a 4D matrix in size of {shape} in {ds.dtype} in the memory ({ram_size:.1f} GB) ...')
            data = np.zeros(shape, ds.dtype) * np.nan

            # loop over the 1st dimension [for more verbose print out msg]
            for i in range(num1):
                if print_msg:
                    sys.stdout.write('\r' + f'reading 3D cubes {i+1}/{num1}...')
                    sys.stdout.flush()

                d3 = ds[i, :,
                        box[1]:box[3],
                        box[0]:box[2]]

                # sampling / nearest interpolation in y/xstep
                if xstep * ystep > 1:
                    d3 = d3[:,
                            int(ystep/2)::ystep,
                            int(xstep/2)::xstep]

                data[i, :, :, :] = d3[:, :ysize, :xsize]

            if print_msg:
                print('')

            if any(i == 1 for i in data.shape):
                data = np.squeeze(data)

    return data


def read_binary_file(fname, datasetName=None, box=None, xstep=1, ystep=1):
    """Read data from binary file, such as .unw, .cor, etc.
    Parameters: fname       : str, path/name of binary file
                datasetName : str, dataset name for file with multiple bands of data
                    e.g.: incidenceAngle, azimuthAngle, rangeCoord, azimuthCoord, ...
                box         : 4-tuple of int area to read, defined in (x0, y0, x1, y1) in pixel coordinate
                x/ystep     : int, number of pixels to pick/multilook for each output pixel
    Returns:    data        : 2D array in size of (length, width) in BYTE / int16 / float32 / complex64 / float64 etc.
                atr         : dict, metadata of binary file
    """
    # Basic Info
    fext = os.path.splitext(os.path.basename(fname))[1].lower()

    # metadata
    atr = read_attribute(fname, datasetName=datasetName)
    processor = atr['PROCESSOR']
    length = int(atr['LENGTH'])
    width = int(atr['WIDTH'])
    box = box if box else (0, 0, width, length)

    # default data structure
    data_type = atr.get('DATA_TYPE', 'float32').lower()
    byte_order = atr.get('BYTE_ORDER', 'little-endian').lower()
    num_band = int(atr.get('BANDS', '1'))
    interleave = atr.get('INTERLEAVE', 'BIL').upper()

    # default data to read
    band = 1
    if datasetName:
        if datasetName.startswith(('mag', 'amp')):
            cpx_band = 'magnitude'
        elif any(x in datasetName.lower() for x in ['phase', 'angle']):
            cpx_band = 'phase'
        elif datasetName.lower() == 'real':
            cpx_band = 'real'
        elif datasetName.lower().startswith('imag'):
            cpx_band = 'imag'
        else:
            cpx_band = 'complex'
    else:
        # use phase as default value, since it's the most common one.
        cpx_band = 'phase'

    # ISCE
    if processor in ['isce']:
        # convert default short name for data type from ISCE
        if data_type in DATA_TYPE_ISCE2NUMPY.keys():
            data_type = DATA_TYPE_ISCE2NUMPY[data_type]

        ftype = atr['FILE_TYPE'].lower().replace('.', '')
        if ftype in ['unw', 'cor', 'ion']:
            band = min(2, num_band)
            if datasetName and datasetName in ['band1','intensity','magnitude']:
                band = 1

        elif ftype in ['slc']:
            if datasetName:
                if datasetName in ['amplitude','magnitude','intensity']:
                    cpx_band = 'magnitude'
                elif datasetName in ['band2','phase']:
                    cpx_band = 'phase'
                else:
                    cpx_band = 'complex'
            else:
                cpx_band = 'complex'

        elif ftype.startswith('los') and datasetName and datasetName.startswith(('band2','az','head')):
            band = min(2, num_band)

        elif ftype in ['incLocal']:
            band = min(2, num_band)
            if datasetName and 'local' not in datasetName.lower():
                band = 1

        elif datasetName:
            if datasetName.lower() == 'band2':
                band = 2
            elif datasetName.lower() == 'band3':
                band = 3
            else:
                # flexible band list
                ds_list = get_slice_list(fname)
                if datasetName in ds_list:
                    band = ds_list.index(datasetName) + 1

        band = min(band, num_band)

        # check file size
        fsize = os.path.getsize(fname)
        dsize = np.dtype(data_type).itemsize * length * width * num_band
        if dsize != fsize:
            warnings.warn(f'file size ({fsize}) does NOT match with metadata ({dsize})!')

    # ROI_PAC
    elif processor in ['roipac']:
        # data structure - file specific based on file extension
        if fext in ['.unw', '.cor', '.hgt', '.msk']:
            num_band = 2
            band = 2

        elif fext in ['.int']:
            data_type = 'complex64'

        elif fext in ['.amp']:
            data_type = 'complex64'
            cpx_band = 'magnitude'

        elif fext in ['.dem', '.wgs84']:
            data_type = 'int16'

        elif fext in ['.flg', '.byt']:
            data_type = 'bool_'

        elif fext in ['.trans']:
            num_band = 2
            if datasetName and datasetName.startswith(('az', 'azimuth')):
                band = 2

    # Gamma
    elif processor == 'gamma':
        # data structure - auto
        byte_order = atr.get('BYTE_ORDER', 'big-endian')

        # convert default short name for data type from GAMMA
        if data_type in DATA_TYPE_GAMMA2NUMPY.keys():
            data_type = DATA_TYPE_GAMMA2NUMPY[data_type]

        if fext in ['.unw', '.cor', '.hgt_sim', '.dem', '.amp', '.ramp']:
            pass

        elif fext in ['.int']:
            data_type = data_type if 'DATA_TYPE' in atr.keys() else 'complex64'

        elif fext.endswith(('to_rdc', '2_rdc', '2rdc')):
            data_type = data_type if 'DATA_TYPE' in atr.keys() else 'float32'
            interleave = 'BIP'
            num_band = 2
            if datasetName and datasetName.startswith(('az', 'azimuth')):
                band = 2

        elif fext in ['.slc', '.rslc']:
            data_type = data_type if 'DATA_TYPE' in atr.keys() else 'complex32'
            cpx_band = 'magnitude'

        elif fext in ['.mli', '.rmli']:
            byte_order = 'little-endian'   # big-endian

    # SNAP
    # BEAM-DIMAP data format
    # https://www.brockmann-consult.de/beam/doc/help/general/BeamDimapFormat.html
    elif processor == 'snap':
        # data structure - auto
        interleave = atr.get('INTERLEAVE', 'BSQ').upper()

        # byte order
        byte_order = atr.get('BYTE_ORDER', 'big-endian')
        if 'byte order' in atr.keys() and atr['byte order'] == '0':
            byte_order = 'little-endian'

    # GDAL / GMTSAR / ASF HyP3
    elif processor in ['gdal', 'gmtsar', 'hyp3', 'cosicorr', 'uavsar']:
        # try to recognize custom dataset names if specified and recognized.
        if datasetName:
            slice_list = get_slice_list(fname)
            if datasetName in slice_list:
                band = slice_list.index(datasetName) + 1

    else:
        print(f'Unknown InSAR processor: {processor}')

    # reading
    kwargs = dict(
        box=box,
        band=band,
        cpx_band=cpx_band,
        xstep=xstep,
        ystep=ystep,
    )
    if processor in ['gdal', 'gmtsar', 'hyp3', 'cosicorr']:
        data = read_gdal(fname, **kwargs)

    else:
        data = read_binary(
            fname,
            shape=(length, width),
            data_type=data_type,
            byte_order=byte_order,
            num_band=num_band,
            interleave=interleave,
            **kwargs
        )

    if 'DATA_TYPE' not in atr:
        atr['DATA_TYPE'] = data_type

    return data, atr


#########################################################################
def get_slice_list(fname, no_complex=False):
    """Get list of 2D slice existed in file (for display).

    Parameters: fname      - str, path to the data file
                no_complex - bool, convert complex into real/imag parts
    Returns:    slice_list - list(str), list of names for 2D matrices
    """

    fbase, fext = _get_file_base_and_ext(fname)
    atr = read_attribute(fname)
    ftype = atr['FILE_TYPE']

    global slice_list
    # HDF5 Files
    if fext in ['.h5', '.he5']:
        with h5py.File(fname, 'r') as f:
            d1_list = [i for i in f.keys() if isinstance(f[i], h5py.Dataset)]

        if ftype == 'timeseries' and any(x.startswith(ftype) for x in d1_list):
            obj = timeseries(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif ftype in ['geometry'] and ftype not in d1_list:
            obj = geometry(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif ftype in ['ifgramStack']:
            obj = ifgramStack(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif ftype in ['HDFEOS']:
            obj = HDFEOS(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif ftype in ['giantTimeseries']:
            obj = giantTimeseries(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif ftype in ['giantIfgramStack']:
            obj = giantIfgramStack(fname)
            obj.open(print_msg=False)
            slice_list = obj.sliceList

        elif ftype == 'timeseries' and 'slc' in d1_list:
            with h5py.File(fname, 'r') as f:
                dates = f['date'][:]
            slice_list = ['slc-{}'.format(i.decode('UTF-8')) for i in dates]

        else:
            ## Find slice by walking through the file structure
            length, width = int(atr['LENGTH']), int(atr['WIDTH'])

            def get_hdf5_2d_dataset(name, obj):
                global slice_list
                if isinstance(obj, h5py.Dataset) and obj.shape[-2:] == (length, width):
                    if obj.ndim == 2:
                        slice_list.append(name)
                    elif obj.ndim == 3:
                        slice_list += [f'{name}-{i+1}' for i in range(obj.shape[0])]
                    else:
                        warnings.warn(f'file has un-defined {obj.ndim}D dataset: {name}')

            # get slice_list
            slice_list = []
            with h5py.File(fname, 'r') as f:
                f.visititems(get_hdf5_2d_dataset)

            # special order for velocity / time func file
            if ftype == 'velocity':
                slice_list = _sort_dataset_list4velocity(slice_list)

    # Binary Files
    else:
        num_band = int(atr.get('BANDS', '1'))
        dtype = atr.get('DATA_TYPE', 'float32')

        if fext in ['.trans'] or fext.endswith(('to_rdc', '2_rdc', '2rdc')):
            # roipac / gamma lookup table
            slice_list = ['rangeCoord', 'azimuthCoord']

        elif fbase.startswith('los') and num_band == 2:
            # isce los file
            slice_list = ['incidenceAngle', 'azimuthAngle']

        elif fext in ['.unw', '.ion']:
            slice_list = ['magnitude', 'phase']

        elif fext in ['.int', '.slc'] or (dtype.startswith('c') and num_band == 1):
            if no_complex:
                slice_list = ['magnitude', 'phase']
            else:
                slice_list = ['complex']

        elif 'offset' in fbase and num_band == 2:
            # ampcor offset file, e.g. offset.bip, dense_offsets.bil
            slice_list = ['azimuthOffset', 'rangeOffset']
            if fname.endswith('_denseoffset.off'):
                # for alos2App dense offset, the file has the opposite band order
                # e.g. 231206-240103_denseoffset.off
                slice_list = ['rangeOffset', 'azimuthOffset']

        elif 'offset' in fbase and 'cov' in fname and num_band == 3:
            # ampcor offset covariance file, e.g. offset_cov.bip, dense_offsets_cov.bil
            slice_list = ['azimuthOffsetVar', 'rangeOffsetVar', 'offsetCovar']
            if fname.endswith('_denseoffset.cov'):
                # for alos2App dense offset, e.g. 231206-240103_denseoffset.cov
                slice_list = ['rangeOffsetVar', 'azimuthOffsetVar', 'offsetCovar']

        elif fext in ['.lkv']:
            slice_list = ['east', 'north', 'up']

        elif fext in ['.llh']:
            slice_list = ['latitude', 'longitude', 'height']

        else:
            slice_list = [f'band{i+1}' for i in range(num_band)]

    return slice_list


def get_dataset_list(fname, datasetName=None):
    """Get list of 2D and 3D dataset to facilitate systematic file reading.

    Parameters: fname       - str, path to the data file
                datasetName - str, dataset of interest
    Returns:    ds_list     - list(str), list of names for 2D/3D datasets
    """
    if datasetName:
        return [datasetName]

    global ds_list
    fext = os.path.splitext(fname)[1].lower()
    if fext in ['.h5', '.he5']:
        # get length/width
        atr = read_attribute(fname)
        length, width = int(atr['LENGTH']), int(atr['WIDTH'])

        def get_hdf5_dataset(name, obj):
            global ds_list
            if isinstance(obj, h5py.Dataset) and obj.shape[-2:] == (length, width):
                ds_list.append(name)

        # get dataset list
        ds_list = []
        with h5py.File(fname, 'r') as f:
            f.visititems(get_hdf5_dataset)

        # special order for velocity / time func file
        if atr['FILE_TYPE'] == 'velocity':
            ds_list = _sort_dataset_list4velocity(ds_list)

    else:
        ds_list = get_slice_list(fname)

    return ds_list


def _sort_dataset_list4velocity(ds_list_in):
    """Sort the dataset list for velocity file type.

    1. time func datasets [required]: velocity
    2. time func datasets [optional]: alphabetic order
    3. time func STD datasets [optional]: velocityStd
    4. time func STD datasets [optional]: alphabetic order
    5. residue

    Parameters: ds_list - list(str), list of names for 2D/3D datasets
    Returns:    ds_list - list(str), list of names for 2D/3D datasets
    """

    ds_list1 = [x for x in ['velocity'] if x in ds_list_in]
    ds_list3 = [x for x in ['velocityStd'] if x in ds_list_in]
    ds_list5 = [x for x in ['intercept', 'interceptStd', 'residue'] if x in ds_list_in]

    ds_list4 = sorted([x for x in ds_list_in if x.endswith('Std') and x not in ds_list1 + ds_list3 + ds_list5])
    ds_list2 = sorted([x for x in ds_list_in if x not in ds_list1 + ds_list3 + ds_list4 + ds_list5])

    ds_list = ds_list1 + ds_list2 + ds_list3 + ds_list4 + ds_list5

    return ds_list


def get_hdf5_dataset_attrs(fname, key='UNIT'):
    """Get the top-level dataset attribute for the given HDF5 file.

    Parameters: fname - str, path to the HDF5 file
    Returns:    attrs - dict, key/value of all top-level datasets
    """

    fext = os.path.splitext(fname)[1]
    if fext not in ['.h5', '.he5']:
        return None

    # default output: empty
    attrs = {}

    # grab attrs from input file for the given key
    with h5py.File(fname) as f:
        # get top-level dataset names
        ds_names = [x for x in f.keys() if isinstance(f[x], h5py.Dataset)]

        # extract given attribute to the output dict
        for ds_name in ds_names:
            if key in f[ds_name].attrs.keys():
                attrs[ds_name] = f[ds_name].attrs[key]

    return attrs


def get_hdf5_compression(fname):
    """Get the compression type of input HDF5 file"""
    ext = os.path.splitext(fname)[1].lower()
    if ext not in ['.h5','.he5']:
        return None

    compression = None
    ds_name = get_dataset_list(fname)[0]
    with h5py.File(fname, 'r') as f:
        compression = f[ds_name].compression
    return compression


def get_no_data_value(fname):
    """Grab the NO_DATA_VALUE of the input file.

    Parameters: fname - str, path to the data file
    Returns:    val   - number, no data value
    """
    val = read_attribute(fname).get('NO_DATA_VALUE', None)
    val = str(val).lower()

    if ut.is_number(val):
        val = float(val)
    elif val in SPECIAL_STR2NUM.keys():
        val = SPECIAL_STR2NUM[val]
    else:
        raise ValueError(f'Un-recognized no-data-value type: {val}')
    return val


def _get_file_base_and_ext(fname):
    """Grab the meaningful file basename and extension.

    Parameters: fname - str, path to the (meta)data file
    Returns:    fbase - str, file basename
                fext  - str, file extension in lower case
    """

    fbase, fext = os.path.splitext(os.path.basename(fname))
    fext = fext.lower()

    # ignore certain meaningless file extensions
    while fext in ['.geo', '.rdr', '.full', '.mli', '.wgs84', '.grd', '.bil', '.bip']:
        fbase, fext = os.path.splitext(fbase)
    # set fext to fbase if nothing left
    fext = fext if fext else fbase

    return fbase, fext


#########################################################################
def read_attribute(fname, datasetName=None, metafile_ext=None):
    """Read attributes of input file into a dictionary
    Parameters: fname : str, path/name of data file
                datasetName : str, name of dataset of interest, for file with multiple datasets
                    e.g. unwrapPhase in ifgramStack.h5
                         coherence   in ifgramStack.h5
                         height      in geometryRadar.h5
                         latitude    in geometryRadar.h5
                         ...
    Returns:    atr : dict, attributes dictionary
    """
    fname = os.fspath(fname)  # Convert from possible pathlib.Path
    fdir = os.path.dirname(fname)
    fbase, fext = os.path.splitext(os.path.basename(fname))
    fext = fext.lower()
    if not os.path.isfile(fname):
        msg = f'input file does not exist: {fname}\n'
        msg += 'current directory: '+os.getcwd()
        raise FileNotFoundError(msg)

    # HDF5 files
    if fext in ['.h5', '.he5']:
        if datasetName:
            # get rid of potential date info
            datasetName = datasetName.split('-')[0]

        with h5py.File(fname, 'r') as f:
            atr = dict(f.attrs)
            g1_list = [i for i in f.keys() if isinstance(f[i], h5py.Group)]
            d1_list = [i for i in f.keys() if isinstance(f[i], h5py.Dataset) and f[i].ndim >= 2]

        # FILE_TYPE
        # pre-defined/known dataset/group names > existing FILE_TYPE > existing dataset/group names
        py2_mintpy_stack_files = ['interferograms', 'coherence', 'wrapped'] #obsolete mintpy format
        if any(i in d1_list for i in ['unwrapPhase', 'rangeOffset', 'azimuthOffset']):
            ftype = 'ifgramStack'
        elif any(i in d1_list for i in ['height', 'latitude', 'azimuthCoord']):
            ftype = 'geometry'
        elif any(i in g1_list+d1_list for i in ['timeseries']):
            ftype = 'timeseries'
        elif any(i in d1_list for i in ['velocity']):
            ftype = 'velocity'
        elif 'HDFEOS' in g1_list:
            ftype = 'HDFEOS'
        elif 'recons' in d1_list:
            ftype = 'giantTimeseries'
        elif any(i in d1_list for i in ['igram', 'figram']):
            ftype = 'giantIfgramStack'
        elif any(i in g1_list for i in py2_mintpy_stack_files):
            ftype = list(set(g1_list) & set(py2_mintpy_stack_files))[0]
        elif 'FILE_TYPE' in atr:
            ftype = atr['FILE_TYPE']
        elif len(d1_list) > 0:
            ftype = d1_list[0]
        elif len(g1_list) > 0:
            ftype = g1_list[0]
        else:
            raise ValueError('unrecognized file type: '+fname)

        # metadata dict
        if ftype == 'giantTimeseries':
            atr = giantTimeseries(fname).get_metadata()
        elif ftype == 'giantIfgramStack':
            atr = giantIfgramStack(fname).get_metadata()

        elif len(atr) > 0 and 'WIDTH' in atr.keys():
            # use the attribute at root level, which is already read from the beginning

            # grab attribute of dataset if specified, e.g. UNIT, no-data value, etc.
            if datasetName and datasetName in d1_list:
                with h5py.File(fname, 'r') as f:
                    atr.update(dict(f[datasetName].attrs))

        else:
            # otherwise, grab the list of attrs in HDF5 file
            # and use the attrs with most items
            global atr_list

            def get_hdf5_attrs(name, obj):
                global atr_list
                if len(obj.attrs) > 0 and 'WIDTH' in obj.attrs.keys():
                    atr_list.append(dict(obj.attrs))

            atr_list = []
            with h5py.File(fname, 'r') as f:
                f.visititems(get_hdf5_attrs)

            # use the attrs with most items
            if atr_list:
                num_list = [len(i) for i in atr_list]
                atr = atr_list[np.argmax(num_list)]
            else:
                raise ValueError('No attribute WIDTH found in file:', fname)

        # decode string format
        for key, value in atr.items():
            try:
                atr[key] = value.decode('utf8')
            except:
                atr[key] = value

        # attribute identified by MintPy
        # 1. FILE_TYPE
        atr['FILE_TYPE'] = str(ftype)

        # 2. DATA_TYPE
        ds = None
        with h5py.File(fname, 'r') as f:
            if datasetName and datasetName in f.keys():
                # get the dataset in the root level
                ds = f[datasetName]
            else:
                # get the 1st dataset in deeper levels
                global ds_list

                def get_hdf5_dataset(name, obj):
                    global ds_list
                    if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
                        ds_list.append(obj)

                ds_list = []
                f.visititems(get_hdf5_dataset)
                if ds_list:
                    ds = ds_list[0]

            if ds is not None:
                atr['DATA_TYPE'] = str(ds.dtype)

        # 3. PROCESSOR
        if 'INSAR_PROCESSOR' in atr.keys():
            atr['PROCESSOR'] = atr['INSAR_PROCESSOR']
        if 'PROCESSOR' not in atr.keys():
            atr['PROCESSOR'] = 'mintpy'

    elif fext == '.dehm':
        # 10 m Digital Ellipsoidal Height Model files from GSI
        atr = {}
        atr['LENGTH'] = 6000           # 40 mins in latitude  per grid
        atr['WIDTH']  = 9000           # 60 mins in longitude per grid
        atr['Y_STEP'] = - 0.4 / 3600.  # degree
        atr['X_STEP'] =   0.4 / 3600.  # degree
        atr['Y_UNIT'] = 'degrees'
        atr['X_UNIT'] = 'degrees'

        # Y/X_FIRST based on the naming convention
        yy, xx = float(fbase[:2]), float(fbase[2:])
        atr['Y_FIRST'] = (yy + 1.) / 1.5
        atr['X_FIRST'] = xx + 100.

        atr['PROCESSOR'] = 'GSI'
        atr['FILE_TYPE'] = fext
        atr['DATA_TYPE'] = 'float32'
        atr['PROJECTION'] = 'LATLON'
        atr['GEODETIC_DATUM'] = 'WGS84'
        atr['UNIT'] = 'm'

        # check file size for potential 5m DEHM data
        if os.path.getsize(fname) != atr['LENGTH'] * atr['WIDTH'] * 4:
            msg = 'input DEHM file size do NOT match with the pre-defined 10m DEHM: '
            msg += '{} * {} in {}!'.format(atr['LENGTH'], atr['WIDTH'], atr['DATA_TYPE'])
            raise ValueError(msg)

    elif fext in ['.lkv', '.llh']:
        # UAVSAR geometry file
        # link: https://uavsar.jpl.nasa.gov/science/documents/stack-format.html
        site, line, version, bcorr = fbase.split('_')[:4]
        ann_files = glob.glob(os.path.join(fdir, f'{site}_{line}_*_{version}_{bcorr}.ann'))
        if len(ann_files) > 0:
            ann = read_uavsar_ann(ann_files[0])
            atr = {}

            # data size
            seg, mli = fbase.split('_')[-2:]
            if seg.startswith('s'):
                # single segment file
                atr['LENGTH'] = ann[f'slc_{seg[1:]}_{mli} Rows']
                atr['WIDTH']  = ann[f'slc_{seg[1:]}_{mli} Columns']

            else:
                # merged/concatenated file
                num_seg = int(ann['Number of Segments'])
                length, width = 0, 0
                for i in range(1, num_seg+1):
                    length += int(ann[f'slc_{i}_{mli} Rows'])
                    width = int(ann[f'slc_{i}_{mli} Columns'])
                atr['LENGTH'] = str(length)
                atr['WIDTH'] = str(width)

            atr['ANTENNA_SIDE'] = '1' if ann['Look Direction'].lower() == 'left' else '-1'
            atr['PROCESSOR'] = 'uavsar'
            atr['PLATFORM'] = 'uavsar'
            atr['FILE_TYPE'] = fext
            atr['DATA_TYPE'] = 'float32'
            atr['RLOOKS'] = mli.split('x')[0]
            atr['ALOOKS'] = mli.split('x')[1]
            atr['BANDS'] = '3'
            atr['INTERLEAVE'] = 'BIP'

        else:
            raise FileNotFoundError('No UAVSAR *.ann file found!')

    else:
        # grab all existed potential metadata file given the data file in preferred order/priority
        # .aux.xml file does not have geo-coordinates info
        # .vrt file (e.g. incLocal.rdr.vrt from isce) does not have band interleavee info
        metafiles = [
            fname + '.rsc',
            fname + '.xml',
            fname + '.par',
            fname + '.hdr',                        # created with SUFFIX=ADD     in gdal envi driver
            os.path.splitext(fname)[0] + '.hdr',   # created with SUFFIX=REPLACE in gdal envi driver
            fname + '.vrt',
            fname + '.aux.xml',
        ]
        metafiles = [i for i in metafiles if os.path.isfile(i)]

        # use metadata files with the specified extension if requested
        if metafile_ext:
            metafiles = [i for i in metafiles if i.endswith(metafile_ext)]

        # for .tif/.grd files, priority:
        # .rsc > file itself > .xml/.aux.xml/.hdr etc.
        if fext in GDAL_FILE_EXTS and not os.path.isfile(fname + '.rsc'):
            metafiles = [fname]
        elif len(metafiles) == 0:
            raise FileNotFoundError(f'No metadata file found for data file: {fname}')

        atr = {}
        # PROCESSOR
        if fname.endswith('.img') and any(i.endswith('.hdr') for i in metafiles):
            atr['PROCESSOR'] = 'snap'

        elif any(i.endswith(('.xml', '.hdr', '.vrt')) for i in metafiles):
            atr['PROCESSOR'] = 'isce'
            xml_files = [i for i in metafiles if i.endswith('.xml')]
            if len(xml_files) > 0:
                atr.update(read_isce_xml(xml_files[0]))

        elif any(i.endswith('.par') for i in metafiles):
            atr['PROCESSOR'] = 'gamma'

        elif any(i.endswith('.rsc') for i in metafiles):
            if 'PROCESSOR' not in atr.keys():
                atr['PROCESSOR'] = 'roipac'

        elif fext in GDAL_FILE_EXTS:
            atr['PROCESSOR'] = 'gdal'

        if 'PROCESSOR' not in atr.keys():
            atr['PROCESSOR'] = 'mintpy'

        # Read metadata file and FILE_TYPE
        metafile = metafiles[0]
        meta_ext = os.path.splitext(metafile)[1].lower()

        # ignore certain meaningless file extensions
        fbase, fext = _get_file_base_and_ext(fname)

        if meta_ext == '.rsc':
            atr.update(read_roipac_rsc(metafile))
            if 'FILE_TYPE' not in atr.keys():
                atr['FILE_TYPE'] = fext

        elif meta_ext == '.xml':
            atr.update(read_isce_xml(metafile))
            if 'FILE_TYPE' not in atr.keys():
                atr['FILE_TYPE'] = fext

        elif meta_ext == '.par':
            atr.update(read_gamma_par(metafile))
            atr['FILE_TYPE'] = fext

        elif meta_ext == '.hdr':
            atr.update(read_envi_hdr(metafile))

            # both snap and isce produce .hdr file
            # grab file type based on their different naming conventions
            if atr['PROCESSOR'] == 'snap':
                fbase = os.path.basename(fname).lower()
                if fbase.startswith('unw'):
                    atr['FILE_TYPE'] = '.unw'
                elif fbase.startswith(('coh','cor')):
                    atr['FILE_TYPE'] = '.cor'
                elif fbase.startswith('phase_ifg'):
                    atr['FILE_TYPE'] = '.int'
                elif 'dem' in fbase:
                    atr['FILE_TYPE'] = 'dem'
                else:
                    atr['FILE_TYPE'] = atr['file type']
            else:
                atr['FILE_TYPE'] = fext

        elif meta_ext in ['.vrt'] + GDAL_FILE_EXTS:
            atr.update(read_gdal_vrt(metafile))
            atr['FILE_TYPE'] = fext

        # DATA_TYPE for ISCE/ROI_PAC products
        data_type_dict = {
            # isce2
            'byte'  : 'int8',
            'float' : 'float32',
            'double': 'float64',
            'cfloat': 'complex64',
            # roipac
            'ci2'   : 'float32',
        }
        data_type = atr.get('DATA_TYPE', 'none').lower()
        if data_type != 'none' and data_type in data_type_dict.keys():
            atr['DATA_TYPE'] = data_type_dict[data_type]

    # UNIT
    if datasetName:
        # ignore Std because it shares the same unit as base parameter
        # e.g. velocityStd and velocity
        datasetName = datasetName.replace('Std','')
        # use the last segment if / is used, e.g. HDF-EOS5
        datasetName = datasetName.split('/')[-1]
    ftype = atr['FILE_TYPE'].replace('.', '')
    if ftype == 'ifgramStack':
        if datasetName and datasetName in DSET_UNIT_DICT.keys():
            atr['UNIT'] = DSET_UNIT_DICT[datasetName]
        else:
            atr['UNIT'] = 'radian'

    elif datasetName and datasetName in DSET_UNIT_DICT.keys():
        atr['UNIT'] = DSET_UNIT_DICT[datasetName]
        # SLC stack
        if datasetName == 'timeseries' and atr.get('DATA_TYPE', 'float32').startswith('complex'):
            atr['UNIT'] = '1'

    elif 'UNIT' not in atr.keys():
        if ftype in DSET_UNIT_DICT.keys():
            atr['UNIT'] = DSET_UNIT_DICT[ftype]
        else:
            atr['UNIT'] = '1'

    # FILE_PATH
    if 'FILE_PATH' in atr.keys() and 'OG_FILE_PATH' not in atr.keys():
        # need to check original source file to successfully subset legacy-sensor products
        atr['OG_FILE_PATH'] = atr['FILE_PATH']
    atr['FILE_PATH'] = os.path.abspath(fname)

    # NO_DATA_VALUE
    atr['NO_DATA_VALUE'] = auto_no_data_value(atr)

    atr = standardize_metadata(atr)

    return atr


def auto_no_data_value(meta):
    """Get default no-data-value for the given file's metadata.

    Parameters: meta          - dict, metadata
    Returns:    no_data_value - str, no data value in lower case string
    """

    if 'NO_DATA_VALUE' in meta.keys():
        no_data_value = meta['NO_DATA_VALUE']

    else:
        processor = meta['PROCESSOR']
        fname = meta['FILE_PATH']
        fbase, fext = _get_file_base_and_ext(fname)
        num_band = int(meta.get('BANDS', 0))

        # known file types
        if processor == 'isce':
            if num_band == 2 and fname.endswith('dense_offsets.bil'):
                # topsApp: dense offset
                no_data_value = -10000.
            elif num_band == 2 and fname.endswith('_denseoffset.off'):
                # alos2App: dense offset
                no_data_value = 0.
            elif num_band == 1 and fname.endswith(('alks_rg.off', 'alks_az.off')):
                # alos2App: geometrical offset
                no_data_value = -999999.0
            else:
                no_data_value = None

        else:
            # default value for unknown file types
            no_data_value = None

    return str(no_data_value).lower()


def standardize_metadata(in_meta, standard_keys=STD_METADATA_KEYS):
    """Convert metadata into ROI_PAC/MintPy format (for metadata with the same values)."""

    # make a copy
    out_meta = dict()
    for key, value in iter(in_meta.items()):
        out_meta[key] = value

    # get potential keys to match
    in_keys = [i for i in out_meta.keys() if i not in standard_keys.keys()]
    std_keys = [i for i in standard_keys.keys() if i not in out_meta.keys()]

    # loop to find match and assign values
    for std_key in std_keys:
        cand_keys = standard_keys[std_key]
        cand_keys = [i for i in cand_keys if i in in_keys]
        if len(cand_keys) > 0:
            out_meta[std_key] = out_meta[cand_keys[0]]

    return out_meta


#########################################################################
def read_template(fname, delimiter='=', skip_chars=None):
    """Read the template file into a dictionary structure.

    Parameters: fname      - str, full path to the template file
                delimiter  - str, string to separate the key and value
                skip_chars - list of str, skip certain characters in values
    Returns:    template   - dict, file content
    Examples:   template = read_template('KyushuAlosAT424.txt')
                template = read_template('smallbaselineApp.cfg')
    """

    if skip_chars and isinstance(skip_chars, str):
        skip_chars = [skip_chars]

    # read input text file / string
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    elif isinstance(fname, str):
        lines = fname.split('\n')
    lines = [x.strip() for x in lines]

    # parse line by line
    template = {}
    for line in lines:
        # split on the 1st occurrence of delimiter
        c = [i.strip() for i in line.split(delimiter, 1)]

        # ignore commented lines or those without variables
        if len(c) >= 2 and not line.startswith(('%', '#', '!')):
            key = c[0]
            value = str.replace(c[1], '\n', '').split("#")[0].strip()
            value = os.path.expanduser(value)  # translate ~ symbol
            value = os.path.expandvars(value)  # translate env variables

            # skip certain characters by replacing them with empty str
            if skip_chars:
                for skip_char in skip_chars:
                    value = value.replace(skip_char, '')

            if value != '':
                template[key] = value

    return template


def read_roipac_rsc(fname, delimiter=' '):
    """Read ROI_PAC .rsc file into a python dict structure.
    Parameters: fname : str.
                    File path of .rsc file.
    Returns:    rscDict : dict
                    Dictionary of keys and values in RSC file.
    Examples:
        from mintpy.utils import readfile
        atr = readfile.read_roipac_rsc('filt_101120_110220_c10.unw.rsc')
    """
    # read .rsc file
    with open(fname) as f:
        lines = f.readlines()

    # convert list of str into dict
    rscDict = {}
    for line in lines:
        c = [i.strip() for i in line.strip().replace('\t',' ').split(delimiter, 1)]
        key = c[0]
        value = c[1].replace('\n', '').strip()
        rscDict[key] = value

    rscDict = standardize_metadata(rscDict)

    return rscDict


def read_gamma_par(fname, delimiter=':', skiprows=3):
    """Read GAMMA .par/.off file into a python dict structure.
    Parameters: fname : str.
                    File path of .par, .off file.
                delimiter : str, optional
                    String used to separate values.
                skiprows : int, optional
                    Skip the first skiprows lines.
    Returns:    parDict : dict
                    Attributes dictionary
    """
    # Read txt file
    with open(fname) as f:
        lines = f.readlines()[skiprows:]

    # convert list of str into dict
    parDict = {}
    for line in lines:
        c = [i.strip() for i in line.strip().split(delimiter, 1)]
        if len(c) < 2 or line.startswith(('%', '#')):
            next
        else:
            key = c[0]
            value = str.replace(c[1], '\n', '').split("#")[0].split()[0].strip()
            parDict[key] = value

    parDict = _attribute_gamma2roipac(parDict)
    parDict = standardize_metadata(parDict)

    return parDict


def _attribute_gamma2roipac(par_dict_in):
    """Convert Gamma metadata into ROI_PAC/MintPy format."""
    par_dict = dict()
    for key, value in iter(par_dict_in.items()):
        par_dict[key] = value

    # LENGTH - number of rows
    for key in par_dict_in.keys():
        if any(key.startswith(i) for i in ['azimuth_lines',
                                           'nlines',
                                           'az_samp',
                                           'interferogram_azimuth_lines']):
            par_dict['LENGTH'] = par_dict[key]

    # WIDTH - number of columns
    for key in par_dict_in.keys():
        if any(key.startswith(i) for i in ['width',
                                           'range_samp',
                                           'interferogram_width']):
            par_dict['WIDTH'] = par_dict[key]

    # radar_frequency -> WAVELENGTH
    key = 'radar_frequency'
    if key in par_dict_in.keys():
        value = float(par_dict[key])
        par_dict['WAVELENGTH'] = str(SPEED_OF_LIGHT / value)

    # sar_to_earth_center/earth_radius_below_sensor -> HEIGHT/EARTH_RADIUS
    key = 'earth_radius_below_sensor'
    if key in par_dict_in.keys():
        Re = float(par_dict[key])
        par_dict['EARTH_RADIUS'] = str(Re)

        key2 = 'sar_to_earth_center'
        if key2 in par_dict_in.keys():
            value = float(par_dict[key2])
            par_dict['HEIGHT'] = str(value - Re)

    # sensor -> PLATFORM
    key = 'sensor'
    if key in par_dict_in.keys():
        par_dict['PLATFORM'] = par_dict[key]

    # heading -> ORBIT_DIRECTION
    key = 'heading'
    if key in par_dict_in.keys():
        value = float(par_dict[key])
        if (270 < value < 360) or (-90 < value < 90):
            par_dict['ORBIT_DIRECTION'] = 'ascending'
        else:
            par_dict['ORBIT_DIRECTION'] = 'descending'

    # azimuth_angle -> ANTENNA_SIDE
    key = 'azimuth_angle'
    if key in par_dict_in.keys():
        value = float(par_dict[key])
        if 0 < value < 180:
            par_dict['ANTENNA_SIDE'] = '-1'
        else:
            par_dict['ANTENNA_SIDE'] = '1'

    return par_dict


def read_isce_xml(fname):
    """Read ISCE .xml file into a python dict structure."""
    root = ET.parse(fname).getroot()
    xmlDict = {}

    # imageFile, e.g. filt_fine.unw.xml
    if root.tag.startswith('image'):
        for child in root.findall('property'):
            key = child.get('name').lower()
            value = child.find('value').text
            xmlDict[key] = value

        # Read lat/lon info for geocoded file
        # in form: root/component coordinate*/property name/value
        for coord_name, prefix in zip(['coordinate1', 'coordinate2'], ['X', 'Y']):
            e_step  = root.find(f"./component[@name='{coord_name}']/property[@name='delta']")
            e_first = root.find(f"./component[@name='{coord_name}']/property[@name='startingvalue']")
            v_step  = float(e_step.find('value').text)  if e_step  is not None else None
            v_first = float(e_first.find('value').text) if e_first is not None else None
            # check against None to better handle the valid v_first value of 0
            if v_first is not None and v_step is not None and 1e-7 < abs(v_step) < 1.:
                xmlDict[f'{prefix}_STEP'] = v_step
                xmlDict[f'{prefix}_FIRST'] = v_first - v_step / 2.
                xmlDict[f'{prefix}_UNIT'] = 'degrees'

        # data_type
        xmlDict['data_type'] = DATA_TYPE_ISCE2NUMPY[xmlDict['data_type'].lower()]

    # PAMDataset, e.g. hgt.rdr.aux.xml
    elif root.tag == 'PAMDataset':
        meta = root.find("./Metadata[@domain='ENVI']")
        for child in meta.findall("MDI"):
            key = child.get('key')
            value = child.text
            xmlDict[key] = value

        # data_type
        xmlDict['data_type'] = DATA_TYPE_ENVI2NUMPY[xmlDict['data_type']]

        # NoDataValue
        meta = root.find("./PAMRasterBand/NoDataValue")
        if meta is not None:
            xmlDict['NoDataValue'] = meta.text

    # configeration file, e.g. topsApp.xml
    elif root.tag.endswith('App'):
        for child in root[0].findall('property'):
            key = child.get('name').lower()
            value = child.find('value')
            xmlDict[key] = value.text if value is not None else child.text

    # standardize metadata keys
    xmlDict = standardize_metadata(xmlDict)

    return xmlDict


def read_envi_hdr(fname):
    """Read ENVI .hdr file into a python dict structure"""
    atr = read_template(fname, delimiter='=')
    atr['DATA_TYPE'] = DATA_TYPE_ENVI2NUMPY[atr.get('data type', '4')]
    atr['BYTE_ORDER'] = ENVI_BYTE_ORDER[atr.get('byte order', '1')]

    # ENVI seems to use the center of the upper-left pixel as the first coordinates
    # link: https://www.l3harrisgeospatial.com/docs/OverviewMapInformationInENVI.html
    if 'map info' in atr.keys():
        map_info = [i.replace('{','').replace('}','').strip() for i in atr['map info'].split(',')]
        x_step = abs(float(map_info[5]))
        y_step = abs(float(map_info[6])) * -1.
        #unit = map_info[-1].replace('}','').split('=')[1].lower()

        if abs(x_step) < 1. and abs(x_step) > 1e-7:
            atr['X_UNIT'] = 'degrees'
            atr['Y_UNIT'] = 'degrees'
            atr['X_STEP'] = str(x_step)
            atr['Y_STEP'] = str(y_step)
            atr['X_FIRST'] = str(float(map_info[3]) - x_step / 2.)
            atr['Y_FIRST'] = str(float(map_info[4]) - y_step / 2.)

    atr = standardize_metadata(atr)

    return atr


def read_gdal_vrt(fname):
    """Read GDAL .vrt file into a python dict structure using gdal

    Modified from $ISCE_HOME/applications/gdal2isce_xml.gdal2isce_xml() written by David Bekaert.
    """
    try:
        from osgeo import gdal, osr
    except ImportError:
        raise ImportError('Cannot import gdal and osr!')

    # read dataset using gdal
    # Using os.fspath to convert Path objects to str, recommended by
    # https://github.com/OSGeo/gdal/issues/1613#issuecomment-824703596
    ds = gdal.Open(os.fspath(fname), gdal.GA_ReadOnly)

    atr = {}
    atr['WIDTH']  = ds.RasterXSize
    atr['LENGTH'] = ds.RasterYSize
    atr['BANDS'] = ds.RasterCount

    # data type
    atr['DATA_TYPE'] = DATA_TYPE_GDAL2NUMPY[ds.GetRasterBand(1).DataType]

    # interleave
    interleave = ds.GetMetadata('IMAGE_STRUCTURE').get('INTERLEAVE', 'PIXEL')
    atr['INTERLEAVE'] = ENVI_BAND_INTERLEAVE[interleave]

    # transformation contains gridcorners
    #   lines/pixels with a spacing of 1/-1 OR lonlat with a spacing of deltalon/deltalat
    # GDAL uses the upper-left corner of the upper-left pixel as the first coordinate,
    #   which is the same as ROI_PAC and MintPy
    #   link: https://gdal.org/tutorials/geotransforms_tut.html
    transform = ds.GetGeoTransform()
    x0 = transform[0]
    y0 = transform[3]
    x_step = abs(transform[1])
    y_step = abs(transform[5]) * -1.

    atr['X_STEP'] = x_step
    atr['Y_STEP'] = y_step
    atr['X_FIRST'] = x0
    atr['Y_FIRST'] = y0

    # projection / coordinate unit
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    atr['EPSG'] = srs.GetAttrValue('AUTHORITY', 1)
    srs_name = srs.GetName()
    if srs_name and 'UTM' in srs_name:
        atr['UTM_ZONE'] = srs_name.split('UTM zone')[-1].strip()
        atr['X_UNIT'] = 'meters'
        atr['Y_UNIT'] = 'meters'

    elif abs(x_step) < 1. and abs(x_step) > 1e-7:
        atr['X_UNIT'] = 'degrees'
        atr['Y_UNIT'] = 'degrees'
        # constrain longitude within (-180, 180]
        if atr['X_FIRST'] > 180.:
            atr['X_FIRST'] -= 360.

    # no data value
    atr['NoDataValue'] = ds.GetRasterBand(1).GetNoDataValue()

    atr = standardize_metadata(atr)

    return atr


def read_uavsar_ann(fname, comment=';', delimiter='='):
    """Read the UAVSAR annotation file into dictionary.
    """
    # read the entirer text file into list of strings
    lines = None
    with open(fname) as f:
        lines = f.readlines()

    # convert the list of strings into a dict object
    meta = {}
    for line in lines:
        line = line.strip()
        c = [x.strip() for x in line.split(delimiter, 1)]
        if len(c) >= 2 and not line.startswith(comment):
            key = c[0].split('(')[0].strip()
            value = str.replace(c[1], '\n', '').split(comment)[0].strip()
            meta[key] = value

    return meta


def read_gmtsar_prm(fname, delimiter='='):
    """Read GMTSAR .prm file into a python dict structure.
    Parameters: fname : str.
                    File path of .rsc file.
    Returns:    prmDict : dict
                    Dictionary of keys and values in the PRM file.
    """
    # read .prm file
    with open(fname) as f:
        lines = f.readlines()

    # convert list of str into dict
    prmDict = {}
    for line in lines:
        c = [i.strip() for i in line.strip().replace('\t',' ').split(delimiter, 1)]
        key = c[0]
        value = c[1].replace('\n', '').strip()
        prmDict[key] = value

    prmDict = _attribute_gmtsar2roipac(prmDict)
    prmDict = standardize_metadata(prmDict)

    return prmDict


def _attribute_gmtsar2roipac(prm_dict_in):
    """Convert GMTSAR metadata into ROI_PAC/MintPy format (for metadata with different values)."""
    prm_dict = dict()
    for key, value in iter(prm_dict_in.items()):
        prm_dict[key] = value

    # lookdir -> ANTENNA_SIDE
    key = 'lookdir'
    if key in prm_dict_in.keys():
        value = prm_dict[key]
        if value.upper() == 'R':
            prm_dict['ANTENNA_SIDE'] = '-1'
        else:
            prm_dict['ANTENNA_SIDE'] = '1'

    # orbdir -> ORBIT_DIRECTION
    key = 'orbdir'
    if key in prm_dict_in.keys():
        prm_dict['ORBIT_DIRECTION'] = {
            'A' : 'ASCENDING',
            'D' : 'DESCENDING',
        }[prm_dict[key]]

    # SC_vel -> AZIMUTH_PIXEL_SIZE (in single look)
    key = 'SC_vel'
    if key in prm_dict_in.keys():
        vel = float(prm_dict[key])
        Re = float(prm_dict['earth_radius'])
        PRF = float(prm_dict['PRF'])
        height = float(prm_dict['SC_height'])
        az_pixel_size = vel / PRF * Re / (Re + height)
        prm_dict['AZIMUTH_PIXEL_SIZE'] = az_pixel_size

    # rng_samp_rate -> RANGE_PIXEL_SIZE (in single look)
    key = 'rng_samp_rate'
    if key in prm_dict_in.keys():
        value = float(prm_dict[key])
        prm_dict['RANGE_PIXEL_SIZE'] = SPEED_OF_LIGHT / value / 2.0

    # SC_clock_start/stop -> CENTER_LINE_TUC
    dt_center = (float(prm_dict['SC_clock_start']) + float(prm_dict['SC_clock_stop'])) / 2.0
    t_center = dt_center - int(dt_center)
    prm_dict['CENTER_LINE_UTC'] = str(t_center * 24. * 60. * 60.)

    # SC_identity -> PLATFORM
    prm_dict['PLATFORM'] = GMTSAR_SENSOR_ID2NAME[int(prm_dict['SC_identity'])]

    return prm_dict


def read_snap_dim(fname):
    """Read SNAP *.dim file into a python dict structure.
    Parameters: fname - str, path of the *.dim file
    Returns:    dim_dict - dict, ditionary of keys and values
    Examples:   from mintpy.utils import readfile
                atr = readfile.read_snap_dim('20190303_20190408_unw_tc.dim')
    """
    root = ET.parse(fname).getroot()
    ds = root.find("Dataset_Sources/MDElem[@name='metadata']/MDElem[@name='Abstracted_Metadata']")

    # initial dict elements
    dim_dict = {}
    dim_dict['PROCESSOR'] = 'snap'
    dim_dict['FILE_TYPE'] = os.path.basename(fname).split('_')[-2].lower()

    # read abstracted_metadata into dict
    for child in ds.findall('MDATTR'):
        key = child.get('name').lower()
        value = child.text
        dim_dict[key] = value

    ## 1. standardize
    dim_dict = standardize_metadata(dim_dict)

    ## 2. extra calculations
    # antenna_side
    dim_dict['ANTENNA_SIDE'] = '-1' if dim_dict['antenna_pointing'] == 'right' else '1'

    # center_line_utc
    start_utc = dt.datetime.strptime(dim_dict['first_line_time'], '%d-%b-%Y %H:%M:%S.%f')
    end_utc   = dt.datetime.strptime(dim_dict['last_line_time'],  '%d-%b-%Y %H:%M:%S.%f')
    center_utc = start_utc + ((end_utc - start_utc) / 2)
    center_secs = (center_utc.hour * 3600.0 +
                   center_utc.minute * 60. +
                   center_utc.second)
    dim_dict['CENTER_LINE_UTC'] = center_secs

    # data_type / length / width
    band = root.find("Image_Interpretation/Spectral_Band_Info")
    dim_dict['DATA_TYPE'] = band.find("DATA_TYPE").text
    dim_dict['LENGTH']    = band.find("BAND_RASTER_HEIGHT").text
    dim_dict['WIDTH']     = band.find("BAND_RASTER_WIDTH").text

    # earth_radius + height
    orbit = ds.find("MDElem[@name='Orbit_State_Vectors']/MDElem[@name='orbit_vector1']")
    x_pos = float(orbit.find("MDATTR[@name='x_pos']").text)
    y_pos = float(orbit.find("MDATTR[@name='y_pos']").text)
    z_pos = float(orbit.find("MDATTR[@name='z_pos']").text)
    height, radius = ut.xyz_to_local_radius((x_pos, y_pos, z_pos))
    dim_dict['HEIGHT'] = height
    dim_dict['EARTH_RADIUS'] = radius

    # platform
    dim_dict['PLATFORM'] = sensor.standardize_sensor_name(dim_dict['PLATFORM'])

    # wavelength
    # radar_frequency is in the unit of MHz
    dim_dict['WAVELENGTH'] = SPEED_OF_LIGHT / (float(dim_dict['radar_frequency']) * 1e6)

    # x/y_first/step_unit
    transform = root.find("Geoposition/IMAGE_TO_MODEL_TRANSFORM").text.split(',')
    transform = [str(float(i)) for i in transform]     # Convert 3.333e-4 to 0.0003333
    dim_dict["X_STEP"]  = transform[0]
    dim_dict["Y_STEP"]  = transform[3]
    dim_dict["X_FIRST"] = transform[4]
    dim_dict["Y_FIRST"] = transform[5]
    dim_dict["X_UNIT"]  = "degrees"
    dim_dict["Y_UNIT"]  = "degrees"

    ## 3. geometry datasets
    # incidence_angle
    dim_dict['INCIDENCE_ANGLE'] = (float(dim_dict['incidence_near']) + float(dim_dict['incidence_near'])) / 2.0

    # slant_range_distance
    dim_dict['SLANT_RANGE_DISTANCE'] = ut.incidence_angle2slant_range_distance(dim_dict, dim_dict['INCIDENCE_ANGLE'])

    ## 4. specific to each interferogram
    bases = ds.find("MDElem[@name='Baselines']").findall("MDElem")[0].findall("MDElem")

    # date12
    # support both delimiters of : and _
    dates = [re.split(':|_', x.get('name'))[1].strip() for x in bases][-2:]
    [date1, date2] = sorted(dt.datetime.strptime(x, '%d%b%Y').strftime('%Y%m%d') for x in dates)
    dim_dict['DATE12'] = f'{date1[2:]}-{date2[2:]}'

    # p_baseline
    pbases = [x.find("MDATTR[@name='Perp Baseline']").text for x in bases]
    pbase = [x for x in pbases if float(x) != 0][0]
    dim_dict['P_BASELINE_TOP_HDR'] = pbase
    dim_dict['P_BASELINE_BOTTOM_HDR'] = pbase

    # convert all key & value in string format
    for key, value in dim_dict.items():
        dim_dict[key] = str(value)

    # ensure int type metadata value
    for key in ['ALOOKS', 'RLOOKS', 'LENGTH', 'WIDTH']:
        if key in dim_dict.keys():
            dim_dict[key] = str(int(float(dim_dict[key])))

    return dim_dict



#########################################################################
def read_binary(fname, shape, box=None, data_type='float32', byte_order='l',
                num_band=1, interleave='BIL', band=1, cpx_band='phase',
                xstep=1, ystep=1):
    """Read binary file using np.fromfile.

    Parameters: fname : str, path/name of data file to read
                shape : tuple of 2 int in (length, width)
                box   : tuple of 4 int in (x0, y0, x1, y1)
                data_type : str, data type of stored array, e.g.:
                    bool_
                    int8, int16, int32
                    float16, float32, float64
                    complex64, complex128
                byte_order      : str, little/big-endian
                num_band        : int, number of bands
                interleave : str, band interleav type, e.g.: BIP, BIL, BSQ
                band     : int, band of interest, between 1 and num_band.
                cpx_band : str, e.g.:
                    real,
                    imag, imaginary
                    phase,
                    mag, magnitude
                    cpx
                x/ystep  : int, number of pixels to pick/multilook for each output pixel
    Returns:    data     : 2D np.array
    Examples:   # ISCE files
                atr = read_attribute(fname)
                shape = (int(atr['LENGTH']), int(atr['WIDTH']))
                data = read_binary('filt_fine.unw', shape, num_band=2, band=2)
                data = read_binary('filt_fine.cor', shape)
                data = read_binary('filt_fine.int', shape, data_type='complex64', cpx_band='phase')
                data = read_binary('burst_01.slc', shape, data_type='complex64', cpx_band='mag')
                data = read_binary('los.rdr', shape, num_band=2, band=1)
    """
    length, width = shape
    if not box:
        box = (0, 0, width, length)

    # read data
    if data_type == 'complex32':
        # for numpy un-supported data type: complex32
        data = read_complex_int16(fname, box=box, byte_order=byte_order, cpx_band='complex')[0]

    else:
        # for numpy supported data type
        if byte_order in ['b', 'big', 'big-endian', 'ieee-be']:
            letter, digit = re.findall(r'(\d+|\D+)', data_type)
            # convert into short style: float32 --> c4
            if len(letter) > 1:
                letter = letter[0]
                digit = int(int(digit) / 8)
            data_type = f'>{letter}{digit}'

        interleave = interleave.upper()
        if interleave == 'BIL':
            count = box[3] * width * num_band
            data = np.fromfile(fname, dtype=data_type, count=count).reshape(-1, width*num_band)
            c0 = width * (band - 1) + box[0]
            c1 = width * (band - 1) + box[2]
            data = data[box[1]:box[3], c0:c1]

        elif interleave == 'BIP':
            count = box[3] * width * num_band
            data = np.fromfile(fname, dtype=data_type, count=count).reshape(-1, width*num_band)
            inds = np.arange(box[0], box[2]) * num_band + band - 1
            data = data[box[1]:box[3], inds]

        elif interleave == 'BSQ':
            count = (box[3] + length * (band - 1)) * width
            data = np.fromfile(fname, dtype=data_type, count=count).reshape(-1, width)
            r0 = length * (band - 1) + box[1]
            r1 = length * (band - 1) + box[3]
            data = data[r0:r1, box[0]:box[2]]

        else:
            raise ValueError('unrecognized band interleaving:', interleave)

    # adjust output band for complex data
    if data_type.replace('>', '').startswith('c'):
        if cpx_band.startswith('real'):
            data = data.real
        elif cpx_band.startswith('imag'):
            data = data.imag

        elif cpx_band.startswith('pha'):
            data = np.angle(data)

            # set ~pi value to 0, as sometimes shown in gamma products
            boundary_values = [-3.1415927, 3.1415927]
            for boundary_value in boundary_values:
                if np.sum(data == boundary_value) > 100:
                    print(f'~pi boundary value ({boundary_value}) detected, convert to zero')
                    data[data == boundary_value] = 0

        elif cpx_band.startswith(('mag', 'amp')):
            data = np.absolute(data)
        elif cpx_band.startswith(('cpx', 'complex')):
            pass
        else:
            raise ValueError('unrecognized complex band:', cpx_band)

    # skipping/multilooking
    if xstep * ystep > 1:
        # output size if x/ystep > 1
        xsize = int((box[2] - box[0]) / xstep)
        ysize = int((box[3] - box[1]) / ystep)

        # sampling
        data = data[int(ystep/2)::ystep,
                    int(xstep/2)::xstep]
        data = data[:ysize, :xsize]

    return data


def read_gdal(fname, box=None, band=1, cpx_band='phase', xstep=1, ystep=1):
    """Read binary data file using gdal.

    Parameters: fname    : str, path/name of data file to read
                box      : tuple of 4 int in (x0, y0, x1, y1)
                band     : int, band of interest, between 1 and num_band.
                cpx_band : str, e.g.: real, imag, phase, mag, cpx
                x/ystep  : int, number of pixels to pick/multilook for each output pixel
    Returns:    data     : 2D np.array
    """
    try:
        from osgeo import gdal
    except ImportError:
        raise ImportError('Cannot import gdal!')

    # open data file
    # Using os.fspath to convert Path objects to str, recommended by
    # https://github.com/OSGeo/gdal/issues/1613#issuecomment-824703596
    ds = gdal.Open(os.fspath(fname), gdal.GA_ReadOnly)
    bnd = ds.GetRasterBand(band)

    # box
    if not box:
        box = (0, 0, ds.RasterXSize, ds.RasterYSize)

    # read
    # Link: https://gdal.org/api/python/osgeo.gdal.html#osgeo.gdal.Band.ReadAsArray
    kwargs = dict(xoff=int(box[0]), win_xsize=int(box[2]-box[0]),
                  yoff=int(box[1]), win_ysize=int(box[3]-box[1]))
    data = bnd.ReadAsArray(**kwargs)

    # adjust output band for complex data
    data_type = DATA_TYPE_GDAL2NUMPY[bnd.DataType]
    if data_type.replace('>', '').startswith('c'):
        if cpx_band.startswith('real'):
            data = data.real
        elif cpx_band.startswith('imag'):
            data = data.imag
        elif cpx_band.startswith('pha'):
            data = np.angle(data)
        elif cpx_band.startswith(('mag', 'amp')):
            data = np.abs(data)
        elif cpx_band.startswith(('cpx', 'complex')):
            pass
        else:
            raise ValueError('unrecognized complex band:', cpx_band)

    # skipping/multilooking
    if xstep * ystep > 1:
        # output size if x/ystep > 1
        xsize = int((box[2] - box[0]) / xstep)
        ysize = int((box[3] - box[1]) / ystep)

        # sampling
        data = data[int(ystep/2)::ystep,
                    int(xstep/2)::xstep]
        data = data[:ysize, :xsize]

    return data



############################  Read GMT Faults  ################################

def read_gmt_lonlat_file(ll_file, SNWE=None, min_dist=10, print_msg=True):
    """Read GMT lonlat file into list of 2D np.ndarray.

    Parameters: ll_file  - str, path to the GMT lonlat file
                SNWE     - tuple of 4 float, area of interest in lat/lon
                min_dist - float, minimum distance in km of fault segments
    Returns:    faults   - list of 2D np.ndarray in size of [num_point, 2] in float32
                           with each row for one point in [lon, lat] in degrees
    Examples:
        # prepare GMT lonlat file
        cd ~/data/aux/faults
        gmt kml2gmt UCERF3_Fault.kml > UCERF3_Fault.lonlat

        # read faults data
        ll_file = os.path.expanduser('~/data/aux/faults/UCERF3_Fault.lonlat')
        faults = read_gmt_lonlat_file(ll_file, SNWE=(31, 36, -118, -113), min_dist=0.1)

        # add faults to the existing plot
        fig, ax = plt.subplots(figsize=[7, 7], subplot_kw=dict(projection=ccrs.PlateCarree()))
        data, atr, inps = view.prep_slice(cmd)
        ax, inps, im, cbar = view.plot_slice(ax, data, atr, inps)

        prog_bar = ptime.progressBar(maxValue=len(faults))
        for i, fault in enumerate(faults):
            ax.plot(fault[:,0], fault[:,1], 'k-', lw=0.2)
            prog_bar.update(i+1, every=10)
        prog_bar.close()
        ax.set_xlim(inps.geo_box[0], inps.geo_box[2])
        ax.set_ylim(inps.geo_box[3], inps.geo_box[1])

    """
    # read text file
    lines = None
    with open(ll_file) as f:
        lines = f.readlines()
    lines = [x for x in lines if not x.startswith('#')]

    debug_mode = False
    if debug_mode:
        lines = lines[:1000]

    # loop to extract/organize the data into list of arrays
    faults, fault = [], []
    num_line = len(lines)
    print_msg = False if num_line < 5000 else print_msg
    prog_bar = ptime.progressBar(maxValue=num_line, print_msg=print_msg)
    for i, line in enumerate(lines):
        prog_bar.update(i+1, every=1000, suffix=f'line {i+1} / {num_line}')

        line = line.strip().replace('\n','').replace('\t', ' ')
        if line.startswith('>'):
            fault = []
        else:
            fault.append([float(x) for x in line.split()[:2]])

        # save if 1) this is the last line OR 2) the next line starts a new fault
        if i == num_line - 1 or lines[i+1].startswith('>'):
            fault = np.array(fault, dtype=np.float32)
            s = np.nanmin(fault[:,1]); n = np.nanmax(fault[:,1])
            w = np.nanmin(fault[:,0]); e = np.nanmax(fault[:,0])

            if fault is not None and SNWE:
                S, N, W, E = SNWE
                if e < W or w > E or s > N or n < S:
                    # check overlap of two rectangles
                    # link: https://stackoverflow.com/questions/40795709
                    fault = None

            if fault is not None and min_dist > 0:
                dist = abs(n - s) * 108 * abs(e - w) * 108 * np.cos((n+s)/2 * np.pi/180)
                if dist < min_dist:
                    fault = None

            if fault is not None:
                faults.append(fault)
    prog_bar.close()

    return faults



############################ Obsolete Functions ###############################
def read_float32(fname, box=None, byte_order='l'):
    """Reads roi_pac data (RMG format, interleaved line by line).

    ROI_PAC file: .unw, .cor, .hgt, .trans, .msk

    RMG format (named after JPL radar pionner Richard M. Goldstein): made
    up of real*4 numbers in two arrays side-by-side. The two arrays often
    show the magnitude of the radar image and the phase, although not always
    (sometimes the phase is the correlation). The length and width of each
    array are given as lines in the metadata (.rsc) file. Thus the total
    width width of the binary file is (2*width) and length is (length), data
    are stored as:
    magnitude, magnitude, magnitude, ...,phase, phase, phase, ...
    magnitude, magnitude, magnitude, ...,phase, phase, phase, ...
    ......

       box  : 4-tuple defining the left, upper, right, and lower pixel coordinate.
    Example:
       a,p,r = read_float32('100102-100403.unw')
       a,p,r = read_float32('100102-100403.unw',(100,1200,500,1500))
    """

    atr = read_attribute(fname)
    if 'DATA_TYPE' not in atr.keys():
        atr['DATA_TYPE'] = 'rmg'
    width = int(float(atr['WIDTH']))
    length = int(float(atr['LENGTH']))
    if not box:
        box = [0, 0, width, length]

    data_type = 'f4'
    if byte_order in ['b', 'big', 'big-endian', 'ieee-be']:
        data_type = '>f4'

    data = np.fromfile(fname,
                       dtype=data_type,
                       count=box[3]*2*width).reshape(box[3], 2*width)
    amplitude = data[box[1]:box[3],
                     box[0]:box[2]]
    phase = data[box[1]:box[3],
                 width+box[0]:width+box[2]]

    return amplitude, phase, atr


def read_real_float64(fname, box=None, byte_order='l'):
    """Read real float64/double data matrix, i.e. isce lat/lon.rdr
    """
    atr = read_attribute(fname)
    if 'DATA_TYPE' not in atr.keys():
        atr['DATA_TYPE'] = 'float64'
    width = int(float(atr['WIDTH']))
    length = int(float(atr['LENGTH']))
    if not box:
        box = [0, 0, width, length]

    data_type = 'f8'
    if byte_order in ['b', 'big', 'big-endian', 'ieee-be']:
        data_type = '>f8'

    data = np.fromfile(fname,
                       dtype=data_type,
                       count=box[3]*width).reshape(box[3], width)
    data = data[box[1]:box[3],
                box[0]:box[2]]
    return data, atr


def read_complex_float32(fname, box=None, byte_order='l', band='phase'):
    """Read complex float 32 data matrix, i.e. roi_pac int or slc data.
    old name: read_complex64()

    ROI_PAC file: .slc, .int, .amp

    Data is stored as:
    real, imaginary, real, imaginary, ...
    real, imaginary, real, imaginary, ...
    ...

    Parameters: fname : str,
                    input file name
                box : 4-tuple
                    defining (left, upper, right, lower) pixel coordinate.
                byte_order : str, optional
                    order of reading byte in the file
                band : str
                    output format, default = phase
                    phase, amplitude, real, imag, complex
    Returns: data : 2D np.array in complex float32
    Example:
        amp, phase, atr = read_complex_float32('geo_070603-070721_0048_00018.int')
        data, atr       = read_complex_float32('150707.slc', 1)
    """

    atr = read_attribute(fname)
    if 'DATA_TYPE' not in atr.keys():
        atr['DATA_TYPE'] = 'complex64'
    width = int(float(atr['WIDTH']))
    length = int(float(atr['LENGTH']))
    if not box:
        box = [0, 0, width, length]

    data_type = 'c8'
    if byte_order in ['b', 'big', 'big-endian', 'ieee-be']:
        data_type = '>c8'

    data = np.fromfile(fname,
                       dtype=data_type,
                       count=box[3]*width).reshape(box[3], width)
    data = data[box[1]:box[3],
                box[0]:box[2]]

    if band == 'phase':
        dataOut = np.angle(data)
    elif band == 'amplitude':
        dataOut = np.absolute(data)
    elif band == 'real':
        dataOut = data.real
    elif band == 'imag':
        dataOut = data.imag
    else:
        dataOut = data

    return dataOut, atr


def read_real_float32(fname, box=None, byte_order='l'):
    """Read real float 32 data matrix, i.e. GAMMA .mli file
    Parameters: fname     : str, path, filename to be read
                byte_order : str, optional, order of reading byte in the file
    Returns: data : 2D np.array, data matrix
             atr  : dict, attribute dictionary
    Usage: data, atr = read_real_float32('20070603.mli')
           data, atr = read_real_float32('diff_filt_130118-130129_4rlks.unw')
    """
    atr = read_attribute(fname)
    if 'DATA_TYPE' not in atr.keys():
        atr['DATA_TYPE'] = 'float32'
    width = int(float(atr['WIDTH']))
    length = int(float(atr['LENGTH']))
    if not box:
        box = [0, 0, width, length]

    data_type = 'f4'
    if byte_order in ['b', 'big', 'big-endian', 'ieee-be']:
        data_type = '>f4'

    data = np.fromfile(fname,
                       dtype=data_type,
                       count=box[3]*width).reshape(box[3], width)
    data = data[box[1]:box[3],
                box[0]:box[2]]
    return data, atr


def read_complex_int16(fname, box=None, byte_order='l', cpx_band='complex'):
    """Read complex int 16 data matrix, i.e. GAMMA SCOMPLEX file (.slc)

    Gamma file: .slc

    Parameters: fname      - str, path to the binary file
                box        - tuple(int), bounding box in (x0, y0, x1, y1)
                byte_order - str, byte order in big/little-endian
                cpx_band   - str, cpx / real / imag / mag / pha
    Returns:    data       - np.ndarray, 2D matrix
                atr        - dict, metadata
    Example: data, atr = read_complex_int16('100102.slc')
             data, atr = read_complex_int16('100102.slc', box=(100,1200,500,1500))
    """

    # read attributes
    atr = read_attribute(fname)
    if 'DATA_TYPE' not in atr.keys():
        atr['DATA_TYPE'] = 'complex32'
    width = int(float(atr['WIDTH']))
    length = int(float(atr['LENGTH']))

    # box
    if not box:
        box = [0, 0, width, length]

    # data_type
    data_type = 'i2'
    if byte_order in ['b', 'big', 'big-endian', 'ieee-be']:
        data_type = '>i2'

    # read from file
    data = np.fromfile(fname, dtype=data_type, count=box[3]*2*width).reshape(box[3], 2*width)
    data = data[box[1]:box[3], 2*box[0]:2*box[2]].flatten()

    odd_idx = np.arange(1, len(data), 2)
    real = data[odd_idx-1].reshape(box[3]-box[1], box[2]-box[0])
    imag = data[odd_idx].reshape(box[3]-box[1], box[2]-box[0])

    # adjust output band for complex data
    if cpx_band.startswith('real'):
        data = real
    elif cpx_band.startswith('imag'):
        data = imag
    elif cpx_band.startswith(('mag', 'amp')):
        data = np.hypot(imag, real)
    elif cpx_band.startswith('pha'):
        data = np.arctan2(imag, real)
    elif cpx_band.startswith(('cpx', 'complex')):
        data = real + 1j * imag
    else:
        raise ValueError('unrecognized complex band:', cpx_band)

    return data, atr


def read_real_int16(fname, box=None, byte_order='l'):
    atr = read_attribute(fname)
    if 'DATA_TYPE' not in atr.keys():
        atr['DATA_TYPE'] = 'int16'
    width = int(float(atr['WIDTH']))
    length = int(float(atr['LENGTH']))
    if not box:
        box = [0, 0, width, length]

    data_type = 'i2'
    if byte_order in ['b', 'big', 'big-endian', 'ieee-be']:
        data_type = '>i2'

    data = np.fromfile(fname,
                       dtype=data_type,
                       count=box[3]*width).reshape(box[3], width)
    data = data[box[1]:box[3],
                box[0]:box[2]]
    return data, atr


def read_bool(fname, box=None):
    """Read binary file with flags, 1-byte values with flags set in bits
    For ROI_PAC .flg, *_snap_connect.byt file.
    """
    # Read attribute
    if fname.endswith('_snap_connect.byt'):
        rscFile = fname.split('_snap_connect.byt')[0]+'.unw.rsc'
    else:
        rscFile = fname+'.rsc'
    atr = read_attribute(rscFile.split('.rsc')[0])
    if 'DATA_TYPE' not in atr.keys():
        atr['DATA_TYPE'] = 'bool'
    width = int(float(atr['WIDTH']))
    length = int(float(atr['LENGTH']))
    if not box:
        box = [0, 0, width, length]

    data = np.fromfile(fname,
                       dtype=np.bool_,
                       count=box[3]*width).reshape(box[3], width)
    data = data[box[1]:box[3],
                box[0]:box[2]]
    return data, atr


def read_GPS_USGS(fname):
    yyyymmdd = np.loadtxt(fname,
                          dtype=bytes,
                          usecols=(0, 1)).astype(str)[:, 0]
    YYYYMMDD = []
    for y in yyyymmdd:
        YYYYMMDD.append(y)
    data = np.loadtxt(fname, usecols=(1, 2, 3, 4))
    dates = data[:, 0]
    north = np.array(data[:, 1])
    east = np.array(data[:, 2])
    up = np.array(data[:, 3])

    return east, north, up, dates, YYYYMMDD

#########################################################################
