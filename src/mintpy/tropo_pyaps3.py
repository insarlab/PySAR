############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, 2015               #
############################################################


import glob
import os
import re
from configparser import ConfigParser

import h5py
import numpy as np
import pyaps3 as pa

import mintpy.cli.diff
from mintpy.objects import geometry, timeseries
from mintpy.utils import ptime, readfile, utils as ut, writefile

WEATHER_MODEL_HOURS = {
    'ERA5'   : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    'ERAINT' : [0, 6, 12, 18],
    'MERRA'  : [0, 6, 12, 18],
}

INT_DATA_TYPES = (int, np.int16, np.int32, np.int64)


###############################################################
def read_inps2date_time(inps):
    """Read dates and time info from input arguments.
    Related options: --file OR --date-list, --hour

    Parameters: inps      - Namespace for the input arguments
    Returns:    date_list - list of str, dates in YYYYMMDD format
                hour      - str, hour in 2-digit with zero padding
    """

    # if --file is specified
    if inps.dis_file:
        # 1) ignore --date-list and --hour
        for key in ['date_list', 'hour']:
            if vars(inps)[key] is not None:
                vars(inps)[key] = None
                msg = f'input "{key:<10}" is ignored'
                msg += f', use info from file {inps.dis_file} instead'
                print(msg)

        # 2) read dates/time from time-series file
        print(f'read dates/time info from file: {inps.dis_file}')
        atr = readfile.read_attribute(inps.dis_file)
        if atr['FILE_TYPE'] == 'timeseries':
            ts_obj = timeseries(inps.dis_file)
            ts_obj.open(print_msg=False)
            inps.date_list = ts_obj.dateList
        else:
            inps.date_list = ptime.yyyymmdd(atr['DATE12'].split('-'))
        inps.hour = closest_weather_model_hour(atr['CENTER_LINE_UTC'], grib_source=inps.tropo_model)

    # read dates if --date-list is text file
    if len(inps.date_list) == 1 and os.path.isfile(inps.date_list[0]):
        date_file = inps.date_list[0]
        if date_file.startswith('SAFE_'):
            print(f'read date list and hour info from Sentinel-1 SAFE filenames: {date_file}')
            inps.date_list, inps.hour = safe2date_time(date_file, inps.tropo_model)
        else:
            print(f'read date list from text file: {date_file}')
            inps.date_list = np.loadtxt(date_file, dtype=bytes, usecols=(0,)).astype(str).tolist()
            inps.date_list = ptime.yyyymmdd(inps.date_list)

    # at east 2 dates are required (for meaningful calculation)
    if len(inps.date_list) < 2:
        raise AttributeError('input number of dates < 2!')

    # print time info
    if inps.hour is None:
        raise AttributeError('time info (--hour) not found!')
    print(f'time of cloest available product: {inps.hour}:00 UTC')

    return inps.date_list, inps.hour


def get_grib_info(inps):
    """Read the following info from inps
        inps.grib_dir
        inps.atr
        inps.snwe
        inps.grib_files
    """
    # utils sub-functions
    def snwe_str2num(snwe_str):
        snwe = snwe_str.split('_')
        snwe = [x.replace('S','-').replace('N','').replace('W','-').replace('E','') for x in snwe]
        snwe = [int(x) for x in snwe]
        return snwe

    def snwe1_contains_snwe2(snwe1, snwe2):
        s1, n1, w1, e1 = snwe1
        s2, n2, w2, e2 = snwe2
        if s1 <= s2 and n1 >= n2 and w1 <= w2 and e1 >= e2:
            return True
        else:
            return False

    def get_larger_snwe_in_local_files(gfile):
        # default output
        larger_snwe = None

        # 0. input grib file info
        gdir = os.path.dirname(gfile)
        gparts = os.path.basename(gfile).split('_')
        # ignore grib files in global scale
        if len(gparts) < 7:
            return larger_snwe
        snwe_str = '_'.join(gparts[1:5])
        snwe = snwe_str2num(snwe_str)
        suffix = os.path.basename(gfile).split(snwe_str)[1]

        # 1. find candidate snwe in local files
        cand_gfiles = glob.glob(os.path.join(gdir, f'*{suffix}'))
        # ignore grib files in global scale
        cand_gfiles = [x for x in cand_gfiles if len(os.path.basename(x).split('_')) >= 7]
        if len(cand_gfiles) > 0:
            cand_snwe_str_list = ['_'.join(os.path.basename(x).split('_')[1:5]) for x in cand_gfiles]
            cand_snwe_list = [snwe_str2num(x) for x in cand_snwe_str_list]

            # 2. find candidate snwe that contains the input snwe, i.e. larger_snwe
            larger_snwe_list = []
            for cand_snwe in cand_snwe_list:
                if snwe1_contains_snwe2(cand_snwe, snwe):
                    larger_snwe_list.append(cand_snwe)

            # 3. locate the larger_snwe with the most existing files
            if len(larger_snwe_list) > 0:
                num_file_list = [len(glob.glob(os.path.join(gdir, f'*{x}*{suffix}'))) for x in larger_snwe_list]
                larger_snwe = larger_snwe_list[np.argmax(num_file_list)]

        return larger_snwe

    # grib data directory, under weather_dir
    inps.grib_dir = os.path.join(inps.weather_dir, inps.tropo_model)
    if not os.path.isdir(inps.grib_dir):
        os.makedirs(inps.grib_dir)
        print(f'make directory: {inps.grib_dir}')

    # read metadata
    if inps.dis_file:
        inps.atr = readfile.read_attribute(inps.dis_file)
    elif inps.geom_file:
        inps.atr = readfile.read_attribute(inps.geom_file)
    else:
        inps.atr = dict()

    # area extent for ERA5 grib data download
    inps.snwe = get_snwe(inps.atr, geom_file=inps.geom_file) if inps.atr else None

    # grib file list
    inps.grib_files = get_grib_filenames(
        date_list=inps.date_list,
        hour=inps.hour,
        model=inps.tropo_model,
        grib_dir=inps.grib_dir,
        snwe=inps.snwe)

    # check existing local files (with the same grib source and time) with larger area extent
    larger_snwe = get_larger_snwe_in_local_files(inps.grib_files[0])
    if larger_snwe is not None:
        larger_grib_files = get_grib_filenames(
            date_list=inps.date_list,
            hour=inps.hour,
            model=inps.tropo_model,
            grib_dir=inps.grib_dir,
            snwe=larger_snwe)

        # use the larger grib file if more local files exist than the current snwe
        #   to avoid re-downloading as much as possible
        grib_files_exist = check_exist_grib_file(inps.grib_files, print_msg=False)
        larger_grib_files_exist = check_exist_grib_file(larger_grib_files, print_msg=False)
        if len(larger_grib_files_exist) > len(grib_files_exist):
            print('Sweet! Find more local grib files with larger SNWE extent, use the larger SNWE instead.')
            inps.snwe = larger_snwe
            inps.grib_files = larger_grib_files

    return inps


def get_grib_filenames(date_list, hour, model, grib_dir, snwe=None):
    """Get default grib file names based on input info.
    Parameters: date_list  - list of str, date in YYYYMMDD format
                hour       - str, hour in 2-digit with zero padding
                model      - str, global atmospheric model name
                grib_dir   - str, local directory to save grib files
                snwe       - tuple of 4 int, for ERA5 only.
    Returns:    grib_files - list of str, local grib file path
    """
    # area extent
    def snwe2str(snwe):
        """Get area extent in string"""
        if not snwe:
            return None
        s, n, w, e = snwe
        area = ''
        area += f'_S{abs(s)}' if s < 0 else f'_N{abs(s)}'
        area += f'_S{abs(n)}' if n < 0 else f'_N{abs(n)}'
        area += f'_W{abs(w)}' if w < 0 else f'_E{abs(w)}'
        area += f'_W{abs(e)}' if e < 0 else f'_E{abs(e)}'
        return area

    area = snwe2str(snwe)

    grib_files = []
    for d in date_list:
        if model == 'ERA5':
            if area:
                grib_file = f'ERA5{area}_{d}_{hour}.grb'
            else:
                grib_file = f'ERA5_{d}_{hour}.grb'

        elif model == 'ERAINT': grib_file = f'ERA-Int_{d}_{hour}.grb'
        elif model == 'MERRA' : grib_file = f'merra-{d}-{hour}.nc4'
        elif model == 'NARR'  : grib_file = f'narr-a_221_{d}_{hour}00_000.grb'
        elif model == 'ERA'   : grib_file = f'ERA_{d}_{hour}.grb'
        elif model == 'MERRA1': grib_file = f'merra-{d}-{hour}.hdf'
        grib_files.append(os.path.join(grib_dir, grib_file))
    return grib_files


###############################################################
def closest_weather_model_hour(sar_acquisition_time, grib_source='ERA5'):
    """Find closest available time of weather product from SAR acquisition time
    Parameters: sar_acquisition_time - str, SAR data acquisition time in seconds
                grib_source          - str, Grib Source of weather reanalysis product
    Returns:    grib_hr              - str, time of closest available weather product
    Example:
        '06' = closest_weather_model_hour(atr['CENTER_LINE_UTC'])
        '12' = closest_weather_model_hour(atr['CENTER_LINE_UTC'], 'NARR')
    """
    # get hour/min of SAR acquisition time
    sar_time = float(sar_acquisition_time)

    # find closest time in available weather products
    grib_hr_list = WEATHER_MODEL_HOURS[grib_source]
    grib_hr = int(min(grib_hr_list, key=lambda x: abs(x-sar_time/3600.)))

    # add zero padding
    grib_hr = f"{grib_hr:02d}"
    return grib_hr


def safe2date_time(safe_file, tropo_model):
    """generate date_list and hour from safe_list"""

    def define_date(string):
        """extract date from *.SAFE"""
        filename = string.split(str.encode('.'))[0].split(str.encode('/'))[-1]
        date = filename.split(str.encode('_'))[5][0:8]
        return date

    def define_second(string):
        """extract CENTER_LINE_UTC from *.SAFE"""
        filename = string.split(str.encode('.'))[0].split(str.encode('/'))[-1]
        time1 = filename.split(str.encode('_'))[5][9:15]
        time2 = filename.split(str.encode('_'))[6][9:15]
        time1_second = int(time1[0:2]) * 3600 + int(time1[2:4]) * 60 + int(time1[4:6])
        time2_second = int(time2[0:2]) * 3600 + int(time2[2:4]) * 60 + int(time2[4:6])
        utc_sec = (time1_second + time2_second) / 2
        return utc_sec

    def seconds_UTC(seconds):
        """generate second list"""
        if isinstance(seconds, list):
            secondsOut = []
            for second in seconds:
                secondsOut.append(second)
        else:
            print('\nUn-recognized CENTER_LINE_UTC input!')
            return None
        return secondsOut

    date_list = ptime.yyyymmdd(np.loadtxt(safe_file, dtype=bytes, converters={0:define_date}).astype(str).tolist())
    second_list = seconds_UTC(np.loadtxt(safe_file, dtype=bytes, converters={0:define_second}).astype(str).tolist())

    # change second into hour
    hour_list = [closest_weather_model_hour(float(second), tropo_model) for second in second_list]
    hour = ut.most_common(hour_list)

    return date_list, hour


def get_snwe(meta, geom_file=None, min_buffer=2, step=10):
    # utils sub-functions
    def ceil2multiple(x, step=10):
        """Given a number x, find the smallest number in multiple of step >= x."""
        assert isinstance(x, INT_DATA_TYPES), f'input number is not int: {type(x)}'
        if x % step == 0:
            return x
        else:
            return x + (step - x % step)

    def floor2multiple(x, step=10):
        """Given a number x, find the largest number in multiple of step <= x."""
        assert isinstance(x, INT_DATA_TYPES), f'input number is not int: {type(x)}'
        return x - x % step

    # get bounding box
    lat0, lat1, lon0, lon1 = get_bounding_box(meta, geom_file=geom_file)

    # lat/lon0/1 --> SNWE
    S = np.floor(min(lat0, lat1) - min_buffer).astype(int)
    N = np.ceil( max(lat0, lat1) + min_buffer).astype(int)
    W = np.floor(min(lon0, lon1) - min_buffer).astype(int)
    E = np.ceil( max(lon0, lon1) + min_buffer).astype(int)

    # SNWE in multiple of 10
    if step > 1:
        S = floor2multiple(S, step=step)
        W = floor2multiple(W, step=step)
        N = ceil2multiple(N, step=step)
        E = ceil2multiple(E, step=step)

    return (S, N, W, E)


def get_bounding_box(meta, geom_file=None):
    """Get lat/lon range (roughly), in the same order of data file
    lat0/lon0 - starting latitude/longitude (first row/column)
    lat1/lon1 - ending latitude/longitude (last row/column)
    """
    length, width = int(meta['LENGTH']), int(meta['WIDTH'])
    if 'Y_FIRST' in meta.keys():
        # geo coordinates
        lat0 = float(meta['Y_FIRST'])
        lon0 = float(meta['X_FIRST'])
        lat_step = float(meta['Y_STEP'])
        lon_step = float(meta['X_STEP'])
        lat1 = lat0 + lat_step * (length - 1)
        lon1 = lon0 + lon_step * (width - 1)

        # for UTM projection, e.g. ASF HyP3
        if not meta.get('Y_UNIT', 'degrees').lower().startswith('deg'):
            lat0, lon0 = ut.utm2latlon(meta, easting=lon0, northing=lat0)
            lat1, lon1 = ut.utm2latlon(meta, easting=lon1, northing=lat1)

    else:
        # radar coordinates
        if geom_file and os.path.isfile(geom_file):
            geom_dset_list = readfile.get_dataset_list(geom_file)
        else:
            geom_dset_list = []

        if 'latitude' in geom_dset_list:
            lats = readfile.read(geom_file, datasetName='latitude')[0]
            lons = readfile.read(geom_file, datasetName='longitude')[0]
            lats[lats == 0] = np.nan
            lons[lons == 0] = np.nan
            lat0 = np.nanmin(lats)
            lat1 = np.nanmax(lats)
            lon0 = np.nanmin(lons)
            lon1 = np.nanmax(lons)

        else:
            # use the rough (not accurate) lat/lon info of the four corners
            lats = [float(meta[f'LAT_REF{i}']) for i in [1,2,3,4]]
            lons = [float(meta[f'LON_REF{i}']) for i in [1,2,3,4]]
            # for UTM projection, e.g. ASF HyP3
            if not meta.get('Y_UNIT', 'degrees').lower().startswith('deg'):
                lats, lons = ut.utm2latlon(meta, easting=lons, northing=lats)
            lat0 = np.mean(lats[0:2])
            lat1 = np.mean(lats[2:4])
            lon0 = np.mean(lons[0:3:2])
            lon1 = np.mean(lons[1:4:2])

    return lat0, lat1, lon0, lon1


###############################################################
def check_exist_grib_file(gfile_list, print_msg=True):
    """Check input list of grib files, and return the existing ones with right size."""
    gfile_exist = ut.get_file_list(gfile_list)
    if gfile_exist:
        file_sizes = [os.path.getsize(i) for i in gfile_exist]
        if file_sizes:
            comm_size = ut.most_common([i for i in file_sizes])
            if print_msg:
                print(f'common file size: {comm_size} bytes')
                print(f'number of grib files existed    : {len(gfile_exist)}')

            gfile_corrupt = []
            for gfile in gfile_exist:
                if os.path.getsize(gfile) < comm_size * 0.9:
                    gfile_corrupt.append(gfile)
        else:
            gfile_corrupt = gfile_exist

        if gfile_corrupt:
            if print_msg:
                print('------------------------------------------------------------------------------')
                print('corrupted grib files detected! Delete them and re-download...')
                print(f'number of grib files corrupted  : {len(gfile_corrupt)}')

            for gfile in gfile_corrupt:
                print(f'remove {gfile}')
                os.remove(gfile)
                gfile_exist.remove(gfile)

            if print_msg:
                print('------------------------------------------------------------------------------')
    return gfile_exist


def check_pyaps_account_config(tropo_model):
    """Check for input in PyAPS config file. If they are default values or are empty, then raise error.
    Parameters: tropo_model - str, tropo model being used to calculate tropospheric delay
    Returns:    None
    """
    # Convert MintPy tropo model name to data archive center name
    # NARR model included for completeness but no key required
    MODEL2ARCHIVE_NAME = {
        'ERA5' : 'CDS',
        'ERAI' : 'ECMWF',
        'MERRA': 'MERRA',
        'NARR' : 'NARR',
    }
    SECTION_OPTS = {
        'CDS'  : ['key'],
        'ECMWF': ['email', 'key'],
        'MERRA': ['user', 'password'],
    }

    # Default values in cfg file
    default_values = [
        'the-email-address-used-as-login@ecmwf-website.org',
        'the-user-name-used-as-login@earthdata.nasa.gov',
        'the-password-used-as-login@earthdata.nasa.gov',
        'the-email-adress-used-as-login@ucar-website.org',
        'your-uid:your-api-key',
    ]

    # account file for pyaps3 < and >= 0.3.0
    cfg_file = os.path.join(os.path.dirname(pa.__file__), 'model.cfg')
    rc_file = os.path.expanduser('~/.cdsapirc')

    # for ERA5: ~/.cdsapirc
    if tropo_model == 'ERA5' and os.path.isfile(rc_file):
        pass

    # check account info for the following models
    elif tropo_model in ['ERA5', 'ERAI', 'MERRA']:
        section = MODEL2ARCHIVE_NAME[tropo_model]

        # Read model.cfg file
        cfg_file = os.path.join(os.path.dirname(pa.__file__), 'model.cfg')
        cfg = ConfigParser()
        cfg.read(cfg_file)

        # check all required option values
        for opt in SECTION_OPTS[section]:
            val = cfg.get(section, opt)
            if not val or val in default_values:
                msg = 'PYAPS: No account info found '
                msg += f'for {tropo_model} in {section} section in file: {cfg_file}'
                raise ValueError(msg)

    return


###############################################################
from concurrent.futures import ThreadPoolExecutor


def dload_grib_files(grib_files, tropo_model='ERA5', snwe=None):
    # import pdb; pdb.set_trace()
    with ThreadPoolExecutor(max_workers=64) as worker:
        futures = []
        for grib_file in grib_files:
            future = worker.submit(dload_grib_files_worker, [grib_file], tropo_model, snwe)
            futures.append(future)
        for future in futures:
            future.result()

    return dload_grib_files_worker(grib_files, tropo_model, snwe)

def dload_grib_files_worker(grib_files, tropo_model='ERA5', snwe=None):
    """Download weather re-analysis grib files using PyAPS
    Parameters: grib_files : list of string of grib files
    Returns:    grib_files : list of string
    """
    print('-'*50)
    print('downloading weather model data using PyAPS ...')

    # Get date list to download (skip already downloaded files)
    grib_files_exist = check_exist_grib_file(grib_files, print_msg=True)
    grib_files2dload = sorted(list(set(grib_files) - set(grib_files_exist)))
    date_list2dload = [str(re.findall(r'\d{8}', os.path.basename(i))[0]) for i in grib_files2dload]
    print('number of grib files to download: %d' % len(date_list2dload))
    print('-'*50)

    # Download grib file using PyAPS
    if len(date_list2dload) > 0:
        # This was the default for a file pattern of "ERA5_N30_N50_W130_W110_20200403_14.grb"
        pattern = re.findall(r'\d{8}[-_]\d{2}', os.path.basename(grib_files2dload[0]))
        if len(pattern) == 0:
            # This is an exception for when the file has a naming format like: "ERA5_N30_N40_W130_W110_20080125T000000_06.grb"
            pattern = re.findall(r'\d{8}T\d{6}[-_]\d{2}', os.path.basename(grib_files2dload[0]))
        hour = pattern[0].replace('-', '_').split('_')[1]
        #hour = re.findall(r'\d{8}[-_]\d{2}', os.path.basename(grib_files2dload[0]))[0].replace('-', '_').split('_')[1]
        grib_dir = os.path.dirname(grib_files2dload[0])

        # Check for non-empty account info in PyAPS config file
        check_pyaps_account_config(tropo_model)

        # try 3 times to download, then use whatever downloaded to calculate delay
        i = 0
        while i < 3:
            i += 1
            try:
                if tropo_model in ['ERA5', 'ERAINT']:
                    pa.ECMWFdload(
                        date_list2dload,
                        hour,
                        grib_dir,
                        model=tropo_model,
                        snwe=snwe,
                        flist=grib_files2dload)

                elif tropo_model == 'MERRA':
                    pa.MERRAdload(date_list2dload, hour, grib_dir)

                elif tropo_model == 'NARR':
                    pa.NARRdload(date_list2dload, hour, grib_dir)
            except:
                if i < 3:
                    print(f'WARNING: the {i} attempt to download failed, retry it.\n')
                else:
                    print('\n\n'+'*'*50)
                    print('WARNING: downloading failed for 3 times, stop trying and continue.')
                    print('*'*50+'\n\n')
                pass

    # check potentially corrupted files
    grib_files = check_exist_grib_file(grib_files, print_msg=False)
    return grib_files


def get_delay(grib_file, tropo_model, delay_type, dem, inc, lat, lon, mask=None, verbose=False):
    """Get delay matrix using PyAPS for one acquisition
    Parameters: grib_file       - str, grib file path
                tropo_model     - str, GAM model
                delay_type      - str, dry/wet/comb
                dem/inc/lat/lon - 2D np.ndarray in float32 for DEM, incidence angle, latitude/longitude
                verbose         - bool, verbose message
    Returns:    pha             - 2D np.ndarray in float32, single path tropospheric delay
                                  temporally absolute, spatially referenced to ref_y/x
    """
    if verbose:
        print(f'GRIB FILE: {grib_file}')

    # initiate pyaps object
    aps_obj = pa.PyAPS(
        grib_file,
        grib=tropo_model,
        Del=delay_type,
        dem=dem,
        inc=inc,
        lat=lat,
        lon=lon,
        mask=mask,
        verb=verbose)

    # estimate delay
    pha = np.zeros((aps_obj.ny, aps_obj.nx), dtype=np.float32)
    aps_obj.getdelay(pha)

    # reverse the sign for consistency between different phase correction steps/methods
    pha *= -1
    return pha


###############################################################
def run_or_skip(grib_files, tropo_file, geom_file):
    """Run or skip the calculation of tropo delay time-series file."""

    def get_dataset_size(fname):
        atr = readfile.read_attribute(fname)
        shape = (int(atr['LENGTH']), int(atr['WIDTH']))
        return shape

    print('update mode: ON')
    print(f'output file: {tropo_file}')
    flag = 'skip'

    # check existence and modification time
    if ut.run_or_skip(out_file=tropo_file, in_file=grib_files, print_msg=False) == 'run':
        flag = 'run'
        print('1) output file either do NOT exist or is NOT newer than all GRIB files.')

    else:
        print('1) output file exists and is newer than all GRIB files.')

        # check dataset size in space / time
        date_list = [str(re.findall(r'\d{8}', os.path.basename(i))[0]) for i in grib_files]
        if (get_dataset_size(tropo_file) != get_dataset_size(geom_file)
                or any(i not in timeseries(tropo_file).get_date_list() for i in date_list)):
            flag = 'run'
            print(f'2) output file does NOT have the same len/wid as the geometry file {geom_file} or does NOT contain all dates')
        else:
            print('2) output file has the same len/wid as the geometry file and contains all dates')

            # check if output file is fully written
            with h5py.File(tropo_file, 'r') as f:
                if np.all(f['timeseries'][-1,:,:] == 0):
                    flag = 'run'
                    print('3) output file is NOT fully written.')
                else:
                    print('3) output file is fully written.')

    # result
    print(f'run or skip: {flag}')
    return flag


def calc_delay_timeseries(inps):
    """Calculate delay time-series and write it to HDF5 file.
    Parameters: inps : namespace, all input parameters
    Returns:    tropo_file : str, file name of ECMWF.h5
    """

    ## 1. prepare geometry data
    geom_obj = geometry(inps.geom_file)
    geom_obj.open()
    inps.inc = geom_obj.read(datasetName='incidenceAngle')
    inps.dem = geom_obj.read(datasetName='height')

    # for testing
    if inps.custom_height:
        print(f'use input custom height of {inps.custom_height} m for vertical integration')
        inps.dem[:] = inps.custom_height

    if 'latitude' in geom_obj.datasetNames:
        # for lookup table in radar-coord (isce, doris)
        inps.lat = geom_obj.read(datasetName='latitude')
        inps.lon = geom_obj.read(datasetName='longitude')

    elif 'Y_FIRST' in geom_obj.metadata:
        # for lookup table in geo-coded (gamma, roipac) and obs. in geo-coord
        inps.lat, inps.lon = ut.get_lat_lon(geom_obj.metadata)

    else:
        # for lookup table in geo-coded (gamma, roipac) and obs. in radar-coord
        inps.lat, inps.lon = ut.get_lat_lon_rdc(inps.atr)

    # mask of valid pixels
    mask = np.multiply(inps.inc != 0, ~np.isnan(inps.inc))


    ## 2. prepare output file
    # metadata
    atr = inps.atr.copy()
    atr['FILE_TYPE'] = 'timeseries'
    atr['UNIT'] = 'm'

    # remove metadata related with double reference
    # because absolute delay is calculated and saved
    for key in ['REF_DATE','REF_X','REF_Y','REF_LAT','REF_LON']:
        if key in atr.keys():
            atr.pop(key)

    # instantiate time-series
    length, width = int(atr['LENGTH']), int(atr['WIDTH'])
    num_date = len(inps.grib_files)
    date_list = [str(re.findall(r'\d{8}', os.path.basename(i))[0]) for i in inps.grib_files]
    dates = np.array(date_list, dtype=np.bytes_)
    ds_name_dict = {
        "date"       : [dates.dtype, (num_date,), dates],
        "timeseries" : [np.float32,  (num_date, length, width), None],
    }
    writefile.layout_hdf5(inps.tropo_file, ds_name_dict, metadata=atr)


    ## 3. calculate phase delay
    print('-'*50)
    print('calculating absolute delay for each date using PyAPS (Jolivet et al., 2011; 2014) ...')
    print(f'number of grib files used: {num_date}')

    prog_bar = ptime.progressBar(maxValue=num_date, print_msg=~inps.verbose)
    for i in range(num_date):
        grib_file = inps.grib_files[i]

        # calc tropo delay
        tropo_data = get_delay(
            grib_file,
            tropo_model=inps.tropo_model,
            delay_type=inps.delay_type,
            dem=inps.dem,
            inc=inps.inc,
            lat=inps.lat,
            lon=inps.lon,
            mask=mask,
            verbose=inps.verbose)

        # write tropo delay to file
        block = [i, i+1, 0, length, 0, width]
        writefile.write_hdf5_block(
            inps.tropo_file,
            data=tropo_data,
            datasetName='timeseries',
            block=block,
            print_msg=False)

        prog_bar.update(i+1, suffix=os.path.basename(grib_file))
    prog_bar.close()

    return inps.tropo_file


###############################################################
def run_tropo_pyaps3(inps):

    ## print key input info
    delay_type2name = {
        'dry'  : 'dry (hydrostatic)',
        'wet'  : 'wet',
        'comb' : 'dry (hydrostatic) and wet',
    }
    print(f'weather model: {inps.tropo_model} - {delay_type2name[inps.delay_type]} delay')
    print(f'weather directory: {inps.weather_dir}')
    print(f'output tropospheric delay     time-series file: {inps.tropo_file}')
    print(f'output corrected displacement time-series file: {inps.cor_dis_file}')

    # read dates / time info
    read_inps2date_time(inps)

    # get corresponding grib files info
    get_grib_info(inps)

    ## 1. download
    print('\n'+'-'*80)
    print('Download global atmospheric model files...')
    if inps.geom_file and run_or_skip(inps.grib_files, inps.tropo_file, inps.geom_file) == 'skip':
        print(f'Skip downloading and use existed troposhperic delay HDF5 file: {inps.tropo_file}.')
    else:
        inps.grib_files = dload_grib_files(
            inps.grib_files,
            tropo_model=inps.tropo_model,
            snwe=inps.snwe)

    ## 2. calculate tropo delay and save to h5 file
    if not inps.geom_file:
        print('No input geometry file, skip calculating and correcting tropospheric delays.')
        return

    print('\n'+'-'*80)
    print('Calculate tropospheric delay and write to HDF5 file...')
    if run_or_skip(inps.grib_files, inps.tropo_file, inps.geom_file) == 'run':
        calc_delay_timeseries(inps)
    else:
        print(f'Skip re-calculating and use existed troposhperic delay HDF5 file: {inps.tropo_file}.')

    ## 3. correct tropo delay from displacement time-series (using diff.py)
    if inps.dis_file:
        print('\n'+'-'*80)
        print('Applying tropospheric correction to displacement file...')
        if ut.run_or_skip(inps.cor_dis_file, [inps.dis_file, inps.tropo_file]) == 'run':
            # diff.py can handle different reference in space and time
            # e.g. the absolute delay and the double referenced time-series
            print('correcting delay via diff.py')
            iargs = [inps.dis_file, inps.tropo_file, '-o', inps.cor_dis_file, '--force']
            print('diff.py', ' '.join(iargs))
            mintpy.cli.diff.main(iargs)

        else:
            print(f'Skip re-applying and use existed corrected displacement file: {inps.cor_dis_file}.')
    else:
        print('No input displacement file, skip correcting tropospheric delays.')

    return
