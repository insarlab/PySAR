#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Feb 2021                           #
############################################################


import os
import sys
import time
import argparse
import h5py
import numpy as np

import mintpy
from mintpy.objects import timeseries
from mintpy.utils import ptime, readfile, writefile, utils as ut
from mintpy.simulation import iono


SPEED_OF_LIGHT = 299792458 # m/s



#####################################################################################
REFERENCE = """references:
  Yunjun, Z., Fattahi, H., Pi, X., Rosen, P., Simons, M., Agram, P., & Aoki, Y. (2022). Range
    Geolocation Accuracy of C/L-band SAR and its Implications for Operational Stack Coregistration.
    IEEE Trans. Geosci. Remote Sens., doi:10.1109/TGRS.2022.3168509. 
  Schaer, S., Gurtner, W., & Feltens, J. (1998). IONEX: The ionosphere map exchange format version 1.1. 
    Paper presented at the Proceedings of the IGS AC workshop, Darmstadt, Germany, Darmstadt, Germany.
"""

EXAMPLE = """example:
  iono_tec.py timeseriesRg.h5 -g inputs/geometryRadar.h5
  iono_tec.py timeseriesRg.h5 -g inputs/geometryRadar.h5 -s cod
"""

def create_parser():
    parser = argparse.ArgumentParser(description='Calculate ionospheric ramps using Global Iono Maps (GIM) from GNSS-based TEC products.',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog=REFERENCE+'\n'+EXAMPLE)
    parser.add_argument('dis_file', help='displacement time-series HDF5 file, i.e. timeseries.h5')
    parser.add_argument('-g','--geomtry', dest='geom_file', type=str, required=True,
                        help='geometry file including incidence/azimuthAngle.')
    parser.add_argument('-s','--sol','--tec-sol', dest='tec_sol', default='jpl',
                        help='TEC solution center (default: %(default)s). \n'
                             '    jpl - JPL (Final)\n'
                             '    igs - IGS (Final)\n'
                             '    cod - CODE (Final)\n'
                             'Check more at:\n'
                             '    https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/atmospheric_products.html')
    parser.add_argument('--tec-dir', dest='tec_dir', default='${WEATHER_DIR}/GIM_IGS',
                        help='directory of downloaded GNSS TEC data (default: %(default)s).')

    # output
    parser.add_argument('--update', dest='update_mode', action='store_true', help='Enable update mode.')
    parser.add_argument('--iono-file', dest='iono_file', help='calculated LOS iono ramp time-series file.')
    #parser.add_argument('-o', dest='cor_dis_file', help='Output file name for the corrected time-series.')

    # GIM extraction
    tec_cfg = parser.add_argument_group('GIM extraction',
                                        'Parameters to extract TEC at point of interest from GIM (mainly for impact demonstration).')
    tec_cfg.add_argument('-i','--interp', dest='interp_method', default='linear3d', choices={'nearest', 'linear2d', 'linear3d'},
                         help='Interpolation method to grab the GIM value at the point of interest (default: %(default)s).')
    tec_cfg.add_argument('--norotate', dest='rotate_tec_map', action='store_false',
                         help="Rotate TEC maps along the longitude direction to compensate the correlation between\n"
                              "the ionosphere and the Sun's position, as suggested by Schaer et al. (1998).\n"
                              "For 'interp_method == linear3d' ONLY. (default: %(default)s).")
    tec_cfg.add_argument('--ratio', dest='sub_tec_ratio', type=str,
                         help='Ratio to calculate the sub-orbital TEC from the total TEC.\n'
                              'Set to "adaptive" for seasonally adaptive scaling.\n'
                              '     Based on equation (14) from Yunjun et al. (2022).\n'
                              'Set to "a value" within (0,1] for a fixed scaling\n'
                              'E.g. 0.75 for TerraSAR-X (Gisinger et al., 2021)\n'
                              '     0.90 for Sentinel-1 (Gisinger et al., 2021)\n'
                              '     0.69 for Sentinel-1 (Yunjun et al., 2022)\n')

    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # --tec-dir
    inps.tec_dir = os.path.expanduser(inps.tec_dir)
    inps.tec_dir = os.path.expandvars(inps.tec_dir)
    if not os.path.isdir(inps.tec_dir):
        print(f'WARNING: Input TEC dir "{inps.tec_dir}" does not exist!')
        inps.tec_dir = os.path.join(os.path.dirname(inps.dis_file), 'TEC')
        print(f'Use "{inps.tec_dir}" instead.')

    # --ratio
    if inps.sub_tec_ratio is None:
        suffix = ''
    elif ut.is_number(inps.sub_tec_ratio):
        suffix = 'R{:.0f}'.format(float(inps.sub_tec_ratio)*100)
    elif inps.sub_tec_ratio.startswith('adap'):
        suffix = 'RA'
    else:
        raise ValueError('Input is neither a number nor startswith adap!')

    # input/output filenames
    inps.dis_file = os.path.abspath(inps.dis_file)
    inps.geom_file = os.path.abspath(inps.geom_file)

    if not inps.iono_file:
        geom_dir = os.path.dirname(inps.geom_file)
        inps.iono_file = os.path.join(geom_dir, 'TEC{}lr{}.h5'.format(inps.tec_sol[0], suffix))

    #if not inps.cor_dis_file:
    #    dis_dir = os.path.dirname(inps.dis_file)
    #    fbase, fext = os.path.splitext(os.path.basename(inps.dis_file))
    #    suffix = os.path.splitext(os.path.basename(inps.iono_file))[0]
    #    inps.cor_dis_file = os.path.join(dis_dir, f'{fbase}_{suffix}{fext}')

    return inps


#####################################################################################
def get_dataset_size(fname):
    atr = readfile.read_attribute(fname)
    shape = (int(atr['LENGTH']), int(atr['WIDTH']))
    return shape


def run_or_skip(iono_file, grib_files, dis_file, geom_file):
    print('update mode: ON')
    print('output file: {}'.format(iono_file))
    flag = 'skip'

    # check existance and modification time
    if ut.run_or_skip(out_file=iono_file, in_file=grib_files, print_msg=False) == 'run':
        flag = 'run'
        print('1) output file either do NOT exist or is NOT newer than all IONEX files.')

    else:
        print('1) output file exists and is newer than all IONEX files.')

        # check dataset size in space / time
        ds_size_dis = get_dataset_size(dis_file)
        ds_size_ion = get_dataset_size(geom_file)
        date_list_dis = timeseries(dis_file).get_date_list()
        date_list_ion = timeseries(iono_file).get_date_list()
        if ds_size_ion != ds_size_dis or any (x not in date_list_ion for x in date_list_dis):
            flag = 'run'
            print(f'2) output file does NOT have the same len/wid as the geometry file {geom_file} or does NOT contain all dates')
        else:
            print('2) output file has the same len/wid as the geometry file and contains all dates')

            # check if output file is fully written
            with h5py.File(iono_file, 'r') as f:
                if np.all(f['timeseries'][-1,:,:] == 0):
                    flag = 'run'
                    print('3) output file is NOT fully written.')
                else:
                    print('3) output file is fully written.')

    # result
    print('run or skip: {}'.format(flag))
    return flag


#####################################################################################
def download_igs_tec(date_list, tec_dir, tec_sol='jpl'):
    """Download IGS TEC products for the input list of dates.

    Parameters: date_list - list of str, in YYYYMMDD
                tec_dir   - str, path to IGS_TEC directory, e.g. ~/data/aux/IGS_TEC
                tec_sol   - str, TEC solution center, e.g. jpl, cod, igs
    Returns:    fnames    - list of str, path of the downloaded TEC files
    """
    print("\n------------------------------------------------------------------------------")
    print("downloading GNSS-based TEC products from NASA's Archive of Space Geodesy Data (CDDIS) ...")
    print('Link: https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/atmospheric_products.html')
    num_date = len(date_list)
    n = len(str(num_date))
    print(f'number of TEC files to download: {num_date}')
    print(f'local TEC file directory: {tec_dir}')

    # output file names/sizes
    fnames = []
    for date_str in date_list:
        fnames.append(iono.get_igs_tec_filename(tec_dir, date_str, sol=tec_sol))

    # remove all existing files
    debug_mode = False
    if debug_mode:
        for fname in fnames:
            for x in [fname, fname+'.Z']:
                if os.path.isfile(x):
                    os.remove(x)

    fsizes = [os.path.getsize(i) / 1024 if os.path.isfile(i) else 0 for i in fnames]

    # download: skip existing ones
    fsizec = ut.most_common(fsizes)
    if fsizec < 400:
        # too small, does not seem right --> download them all
        date_list2dload = list(date_list)

    else:
        # download missing ones
        date_list2dload = [d for d, s in zip(date_list, fsizes) if s < fsizec * 0.9]

    num_date2dload = len(date_list2dload)
    if num_date2dload == 0:
        print(f'ALL files exists with consistent file size (~{fsizec:.0f} KB) --> skip re-downloading.\n')

    else:
        for i, date_str in enumerate(date_list2dload):
            print('-'*20)
            print('DATE {}/{}: {}'.format(i+1, num_date2dload, date_str))
            iono.dload_igs_tec(date_str, tec_dir, sol=tec_sol, print_msg=True)

        # print file size info, after downloading
        fsizes = [os.path.getsize(i) / 1024 if os.path.isfile(i) else 0 for i in fnames]
        for i in range(num_date):
            print('[{i:0{n}d}/{N}] {f}: {s:.2f} KB'.format(n=n, i=i+1, N=num_date, f=fnames[i], s=fsizes[i]))

    return fnames


def calc_iono_ramp_timeseries_igs(tec_dir, tec_sol, interp_method, ts_file, geom_file, iono_file,
                                  rotate_tec_map=True, sub_tec_ratio=None, update_mode=True):
    """Calculate the time-series of 2D ionospheric delay from IGS TEC data.
    Considering the variation of the incidence angle along range direction.

    Parameters: tec_dir   - str, path of the local TEC directory
                ts_file   - str, path of the time-series file
                geom_file - str, path of the geometry file including incidenceAngle data
                iono_file - str, path of output iono ramp time-series file
    Returns:    iono_file - str, path of output iono ramp time-series file
    """
    print("\n------------------------------------------------------------------------------")
    # prepare geometry
    iono_lat, iono_lon = iono.prep_geometry_iono(geom_file, print_msg=True)[1:3]

    # prepare date/time
    date_list = timeseries(ts_file).get_date_list()
    meta = readfile.read_attribute(ts_file)
    utc_sec = float(meta['CENTER_LINE_UTC'])
    h, s = divmod(utc_sec, 3600)
    m, s = divmod(s, 60)
    print('UTC time: {:02.0f}:{:02.0f}:{:02.1f}'.format(h, m, s))

    # read IGS TEC
    vtec_list = []
    print('read IGS TEC file ...')
    print('interpolation method: {}'.format(interp_method))
    prog_bar = ptime.progressBar(maxValue=len(date_list))
    for i, date_str in enumerate(date_list):
        # read zenith TEC
        tec_file = iono.get_igs_tec_filename(tec_dir, date_str, sol=tec_sol)
        vtec = iono.get_igs_tec_value(
            tec_file,
            utc_sec,
            lat=iono_lat,
            lon=iono_lon,
            interp_method=interp_method,
            rotate_tec_map=rotate_tec_map,
        )
        vtec_list.append(vtec)
        prog_bar.update(i+1, suffix=date_str)
    prog_bar.close()

    # TEC --> iono ramp
    vtec2iono_ramp_timeseries(
        date_list=date_list,
        vtec_list=vtec_list,
        geom_file=geom_file,
        iono_file=iono_file,
        sub_tec_ratio=sub_tec_ratio,
        update_mode=update_mode,
    )

    return iono_file


def vtec2iono_ramp_timeseries(date_list, vtec_list, geom_file, iono_file, sub_tec_ratio=None,
                              ds_dict_ext=None, update_mode=True):
    """Convert zenith TEC to 2D slant range delay (ramp due to the incidence angle variation)
    and write to HDF5 time-series file.

    Parameters: date_list   - list of str, dates in YYYYMMDD format
                vtec_list   - list of float32, zenith TEC in TECU
                geom_file   - str, path of the geometry file including incidenceAngle data
                iono_file   - str, path of output iono ramp time-series file
                update_mode - bool,
                ds_dict_ext - dict, extra dictionary of dataset to be saved into the HDF5 file.
    Returns:    iono_file   - str, path of output iono ramp time-series file
    """
    top_perc_file = os.path.join(os.path.dirname(mintpy.__file__), 'data', 'top_tec_perc_s1.txt')

    # prepare geometry
    (iono_inc_angle,
     iono_lat,
     iono_lon,
     iono_height) = iono.prep_geometry_iono(geom_file, print_msg=True)

    # prepare date/time
    num_date = len(date_list)
    if len(vtec_list) != num_date:
        msg = 'Input tec_list and date_list have different size!'
        msg += '\nFor acquisitions without TEC data, set it to NaN.'
        raise ValueError(msg)

    meta = readfile.read_attribute(geom_file)
    length = int(meta['LENGTH'])
    width = int(meta['WIDTH'])
    freq = SPEED_OF_LIGHT / float(meta['WAVELENGTH'])

    # Note: Scaling gives slightly better RMSE for SenD but much worse RMSE for SenA and Alos2
    # thus this is not used by default.
    if sub_tec_ratio is not None:
        if ut.is_number(sub_tec_ratio):
            print('multiply VTEC by {}'.format(sub_tec_ratio))
            vtec_list = (np.array(vtec_list).flatten() * float(sub_tec_ratio)).tolist()

        elif sub_tec_ratio.startswith('adap'):
            dates = ptime.date_list2vector(date_list)[0]
            ydays = np.array([x.timetuple().tm_yday for x in dates])
            fc = np.loadtxt(top_perc_file, dtype=bytes).astype(np.float32)
            print('multiply VTEC adaptively based on the day of the year from: {}'.format(top_perc_file))
            sub_perc = fc[:,2][np.array(ydays)]
            vtec_list = (np.array(vtec_list).flatten() * sub_perc).tolist()

    # loop to calculate the range delay (ramp)
    print('calculating ionospheric phase ramp time-series from TEC ...')
    ts_ramp = np.zeros((num_date, length, width), dtype=np.float32)
    prog_bar = ptime.progressBar(maxValue=num_date)
    for i, date_str in enumerate(date_list):
        ts_ramp[i,:,:] = iono.vtec2range_delay(
            vtec_list[i],
            inc_angle=iono_inc_angle,
            freq=freq,
        )
        prog_bar.update(i+1, suffix=date_str)
    prog_bar.close()

    ## output
    # prepare metadata
    meta['FILE_TYPE'] = 'timeseries'
    meta['UNIT'] = 'm'
    meta['IONO_LAT'] = iono_lat
    meta['IONO_LON'] = iono_lon
    meta['IONO_HEIGHT'] = iono_height
    meta['IONO_INCIDENCE_ANGLE'] = np.nanmean(iono_inc_angle)
    # absolute delay without double reference
    for key in ['REF_X','REF_Y','REF_LAT','REF_LON','REF_DATE']:
        if key in meta.keys():
            meta.pop(key)

    # prepare data matrix
    ds_dict = {}
    ds_dict['date'] = np.array(date_list, dtype=np.string_)
    ds_dict['vtec'] = np.array(vtec_list, dtype=np.float32)
    ds_dict['timeseries'] = ts_ramp

    # add the extra dataset if specified, e.g. vtec_gim, vtec_top, vtec_sub
    if ds_dict_ext is not None:
        ds_names = ds_dict.keys()
        for ds_name, ds_val in ds_dict_ext.items():
            if ds_name not in ds_names:
                ds_dict[ds_name] = ds_val

    # write to disk
    writefile.write(ds_dict, iono_file, metadata=meta)

    return iono_file


def correct_timeseries(dis_file, iono_file, cor_dis_file):
    """Correct time-series for the solid Earth tides."""
    # diff.py can handle different reference in space and time
    # between the absolute iono ramp and the double referenced time-series
    print('\n------------------------------------------------------------------------------')
    print('correcting relative delay for input time-series using diff.py')
    from mintpy import diff

    iargs = [dis_file, iono_file, '-o', cor_dis_file]
    print('diff.py', ' '.join(iargs))
    diff.main(iargs)
    return cor_dis_file


#####################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    start_time = time.time()

    # download
    date_list = timeseries(inps.dis_file).get_date_list()
    tec_files = download_igs_tec(date_list, tec_dir=inps.tec_dir, tec_sol=inps.tec_sol)

    # calculate
    if run_or_skip(inps.iono_file, tec_files, inps.dis_file, inps.geom_file) == 'run':
        calc_iono_ramp_timeseries_igs(
            tec_dir=inps.tec_dir,
            tec_sol=inps.tec_sol,
            interp_method=inps.interp_method,
            ts_file=inps.dis_file,
            geom_file=inps.geom_file,
            iono_file=inps.iono_file,
            rotate_tec_map=inps.rotate_tec_map,
            sub_tec_ratio=inps.sub_tec_ratio,
            update_mode=inps.update_mode,
        )

    ## correct
    #correct_timeseries(dis_file=inps.dis_file,
    #                   iono_file=inps.iono_file,
    #                   cor_dis_file=inps.cor_dis_file)

    m, s = divmod(time.time() - start_time, 60)
    print('time used: {:02.0f} mins {:02.1f} secs.\n'.format(m, s))
    return


#####################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
