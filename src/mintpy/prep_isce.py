############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, 2018               #
############################################################


import glob
import os

import numpy as np

from mintpy.utils import (
    attribute as attr,
    isce_utils,
    ptime,
    readfile,
    writefile,
)

GEOMETRY_PREFIXS = ['hgt', 'lat', 'lon', 'los', 'shadowMask', 'waterMask', 'incLocal']


#########################################################################
def add_ifgram_metadata(metadata_in, dates=[], baseline_dict={}):
    """Add metadata unique for each interferogram
    Parameters: metadata_in   : dict, input common metadata for the entire dataset
                dates         : list of str in YYYYMMDD or YYMMDD format
                baseline_dict : dict, output of baseline_timeseries()
    Returns:    metadata      : dict, updated metadata
    """
    # make a copy of input metadata
    metadata = {}
    for k in metadata_in.keys():
        metadata[k] = metadata_in[k]

    # DATE12
    metadata['DATE12'] = f'{dates[0][2:]}-{dates[1][2:]}'

    # P_BASELINE*
    if baseline_dict:
        bperp_top = baseline_dict[dates[1]][0] - baseline_dict[dates[0]][0]
        bperp_bottom = baseline_dict[dates[1]][1] - baseline_dict[dates[0]][1]
        metadata['P_BASELINE_TOP_HDR'] = str(bperp_top)
        metadata['P_BASELINE_BOTTOM_HDR'] = str(bperp_bottom)
    return metadata


def prepare_geometry(geom_dir, geom_files=[], metadata=dict(), processor='tops', update_mode=True):
    """Prepare and extract metadata from geometry files.

    Parameters: geom_dir   - str, path to the directorry for the geometry data files
                geom_files - list(str), basenames of geometry data files
                metadata   - dict, common metadata for the stack
                processor  - str, isce-2 stack processor
    """

    print('preparing RSC file for geometry files')
    geom_dir = os.path.abspath(geom_dir)

    # default file basenames
    if not geom_files:
        if processor in ['tops', 'stripmap']:
            geom_files = [f'{i}.rdr' for i in GEOMETRY_PREFIXS]

        elif processor in ['alosStack']:
            alooks = metadata['ALOOKS']
            rlooks = metadata['RLOOKS']
            fexts = ['.hgt', '.lat', '.lon', '.los', '.wbd']
            geom_files = [f'*_{rlooks}rlks_{alooks}alks{fext}' for fext in fexts]

        else:
            raise Exception(f'unknown processor: {processor}')

    # get absolute file paths
    geom_files = [os.path.join(geom_dir, i) for i in geom_files]

    # check the full resolution version if no multilooked version exists
    if all(not os.path.isfile(i) for i in geom_files):
        geom_files = [i+'.full' for i in geom_files]

    # get existed files
    geom_files = [i for i in geom_files if os.path.isfile(i)]
    # remove duplicates while preserving order
    seen = set()
    seen_add = seen.add
    geom_files = [i for i in geom_files if not (i in seen or seen_add(i))]

    # write rsc file for each file
    for geom_file in geom_files:
        # prepare metadata for current file
        geom_meta = {**metadata}
        if os.path.isfile(geom_file+'.xml'):
            geom_meta.update(readfile.read_attribute(geom_file, metafile_ext='.xml'))
        else:
            geom_meta.update(readfile.read_attribute(geom_file))

        # write .rsc file
        rsc_file = geom_file+'.rsc'
        writefile.write_roipac_rsc(geom_meta, rsc_file,
                                   update_mode=update_mode,
                                   print_msg=True)
    return


def gen_random_baseline_timeseries(obs_file, max_bperp=10):
    """Generate a baseline time series with random values
    with date12 values grabbed from the directory names of the given path pattern from obs_file.
    """
    # list of dates
    date12s = sorted(os.path.basename(os.path.dirname(x)) for x in glob.glob(obs_file))
    date1s = [x.split('_')[0] for x in date12s]
    date2s = [x.split('_')[1] for x in date12s]
    date_list = sorted(list(set(date1s + date2s)))

    # list of bperp
    bperp_list = [0] + np.random.randint(-max_bperp, max_bperp, len(date_list)-1).tolist()

    # prepare output
    bDict = {}
    for date_str, bperp in zip(date_list, bperp_list):
        bDict[date_str] = [bperp, bperp]

    return bDict


def prepare_stack(obs_file, metadata=dict(), baseline_dict=dict(), update_mode=True):
    """Prepare metadata for a stack of observation data files.

    Parameters: obs_file     : path pattern with wildcards for the primary observation files, e.g. *.unw file.
                metadata     : dict, common metadata for the stack
                baseline_dir : dict, baseline time series
    """
    print(f'preparing RSC file for: {obs_file}')
    isce_files = sorted(glob.glob(obs_file))
    if len(isce_files) == 0:
        raise FileNotFoundError(f'NO file found with path pattern: {obs_file}')

    # make a copy
    meta = {**metadata}

    # update A/RLOOKS, RANGE/AZIMUTH_PIXEL_SIZE, NCORRLOOKS
    # for low resolution ionosphere from isce2/topsStack
    keys = ['LENGTH', 'WIDTH']
    if all(x in meta.keys() for x in keys):
        atr = readfile.read_attribute(isce_files[0], metafile_ext='.xml')
        if any(int(meta[x]) != int(atr[x]) for x in keys):
            resize2shape = (int(atr['LENGTH']), int(atr['WIDTH']))
            meta = attr.update_attribute4resize(meta, resize2shape)

    # write .rsc file for each interferogram file
    num_file = len(isce_files)
    print_msg = True if num_file > 5 else False   # do not print progress bar for <=5 files
    prog_bar = ptime.progressBar(maxValue=num_file, print_msg=print_msg)
    for i, isce_file in enumerate(isce_files):
        if obs_file.endswith('ion_dates/*.ion') or obs_file.endswith('ion_burst_ramp_merged_dates/*.float'):
            # get date for timeseries
            date = ptime.get_date_str_format(isce_file)
            prog_bar.update(i+1, suffix=f'{date} {i+1}/{num_file}')
            ts_meta = {**meta}
            ts_meta.update(readfile.read_attribute(isce_file, metafile_ext='.xml'))
            out_meta = ts_meta
        else:
            # get date1/2
            date12 = ptime.get_date12_from_path(isce_file)
            dates = ptime.yyyymmdd(date12.replace('-','_').split('_'))
            prog_bar.update(i+1, suffix=f'{dates[0]}_{dates[1]} {i+1}/{num_file}')

            # merge metadata from: data.rsc, *.unw.xml and DATE12/P_BASELINE_TOP/BOTTOM_HDR
            ifg_meta = {**meta}
            ifg_meta.update(readfile.read_attribute(isce_file, metafile_ext='.xml'))
            ifg_meta = add_ifgram_metadata(ifg_meta, dates, baseline_dict)
            out_meta = ifg_meta

        # write .rsc file
        rsc_file = isce_file+'.rsc'
        writefile.write_roipac_rsc(out_meta, rsc_file,
                                   update_mode=update_mode,
                                   print_msg=False)

    prog_bar.close()
    return


#########################################################################
def prep_isce(inps):
    """Prepare ISCE-2 metadata files."""

    inps.processor = isce_utils.get_processor(inps.meta_file)

    # read common metadata
    metadata = {}
    if inps.meta_file:
        rsc_file = os.path.join(os.path.dirname(inps.meta_file), 'data.rsc')
        metadata = isce_utils.extract_isce_metadata(
            inps.meta_file,
            geom_dir=inps.geom_dir,
            rsc_file=rsc_file,
            update_mode=inps.update_mode)[0]

    # prepare metadata for geometry file
    if inps.geom_dir:
        prepare_geometry(
            inps.geom_dir,
            geom_files=inps.geom_files,
            metadata=metadata,
            processor=inps.processor,
            update_mode=inps.update_mode)

    # read baseline info
    baseline_dict = {}
    if inps.baseline_dir:
        if inps.baseline_dir.startswith('rand') and inps.obs_files:
            baseline_dict = gen_random_baseline_timeseries(inps.obs_files[0])
        else:
            baseline_dict = isce_utils.read_baseline_timeseries(
                inps.baseline_dir,
                processor=inps.processor)

    # prepare metadata for ifgram file
    if inps.obs_files:
        for obs_file in inps.obs_files:
            prepare_stack(
                obs_file,
                metadata=metadata,
                baseline_dict=baseline_dict,
                update_mode=inps.update_mode)

    print('Done.')
    return
