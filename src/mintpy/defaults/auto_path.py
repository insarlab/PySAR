"""Utilities for automatic configuration for input file paths"""
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Mar 2018                           #
############################################################
# Recommended usage:
#   from mintpy.defaults import auto_path


import glob
import os
import re

# Default path of data files from different InSAR processors to be loaded into MintPy
AUTO_PATH_ISCE_TOPS = '''##----------Default file path of ISCE/topsStack products
mintpy.load.processor       = isce
mintpy.load.metaFile        = ../reference/IW*.xml
mintpy.load.baselineDir     = ../baselines

mintpy.load.unwFile         = ../merged/interferograms/*/filt*.unw
mintpy.load.corFile         = ../merged/interferograms/*/filt*.cor
mintpy.load.connCompFile    = ../merged/interferograms/*/filt*.unw.conncomp
mintpy.load.intFile         = None

mintpy.load.ionFile          = ../ion_dates/*.ion
mintpy.load.ionBurstRampFile = ../ion_burst_ramp_merged_dates/*.float

mintpy.load.demFile         = ../merged/geom_reference/hgt.rdr
mintpy.load.lookupYFile     = ../merged/geom_reference/lat.rdr
mintpy.load.lookupXFile     = ../merged/geom_reference/lon.rdr
mintpy.load.incAngleFile    = ../merged/geom_reference/los.rdr
mintpy.load.azAngleFile     = ../merged/geom_reference/los.rdr
mintpy.load.shadowMaskFile  = ../merged/geom_reference/shadowMask.rdr
mintpy.load.waterMaskFile   = ../merged/geom_reference/waterMask.rdr
mintpy.load.bperpFile       = None
'''

AUTO_PATH_ISCE_STRIPMAP = '''##----------Default file path of ISCE/stripmapStack products
mintpy.load.processor       = isce
mintpy.load.metaFile        = ${ref_shelve}/data.dat
mintpy.load.baselineDir     = ../baselines

mintpy.load.unwFile         = ../Igrams/*/filt*.unw
mintpy.load.corFile         = ../Igrams/*/filt*.cor
mintpy.load.connCompFile    = ../Igrams/*/filt*.unw.conncomp
mintpy.load.intFile         = None

mintpy.load.demFile         = ../geom_reference/hgt.rdr
mintpy.load.lookupYFile     = ../geom_reference/lat.rdr
mintpy.load.lookupXFile     = ../geom_reference/lon.rdr
mintpy.load.incAngleFile    = ../geom_reference/los.rdr
mintpy.load.azAngleFile     = ../geom_reference/los.rdr
mintpy.load.shadowMaskFile  = ../geom_reference/shadowMask.rdr
mintpy.load.waterMaskFile   = ../geom_reference/waterMask.rdr
mintpy.load.bperpFile       = None
'''

AUTO_PATH_ISCE_ALOS = '''##----------Default file path of ISCE/alosStack products
mintpy.load.metaFile         = ../dates_res*/${ref_date}/${ref_date}.track.xml
mintpy.load.baselineDir      = ../baseline

mintpy.load.unwFile          = ../pairs/*-*/insar/filt_*-*_${multilook}.unw
mintpy.load.corFile          = ../pairs/*-*/insar/*-*_${multilook}.cor
mintpy.load.connCompFile     = ../pairs/*-*/insar/filt_*-*_${multilook}.unw.conncomp

mintpy.load.ionUnwFile       = ../pairs_ion/*-*/ion/ion_cal/filt_ion_*.ion
mintpy.load.ionCorFile       = ../pairs_ion/*-*/ion/ion_cal/diff_*.cor
mintpy.load.ionConnCompFile  = None

mintpy.load.demFile          = ../dates_res*/${ref_date}/insar/*_${multilook}.hgt
mintpy.load.lookupYFile      = ../dates_res*/${ref_date}/insar/*_${multilook}.lat
mintpy.load.lookupXFile      = ../dates_res*/${ref_date}/insar/*_${multilook}.lon
mintpy.load.incAngleFile     = ../dates_res*/${ref_date}/insar/*_${multilook}.los
mintpy.load.azAngleFile      = ../dates_res*/${ref_date}/insar/*_${multilook}.los
mintpy.load.waterMaskFile    = ../dates_res*/${ref_date}/insar/*_${multilook}.wbd
'''

AUTO_PATH_ROIPAC = '''##----------Default file path of ROI_PAC products
mintpy.load.processor       = roipac
mintpy.load.unwFile         = ../PROCESS/DONE/IFG*/filt*.unw
mintpy.load.corFile         = ../PROCESS/DONE/IFG*/filt*.cor
mintpy.load.connCompFile    = ../PROCESS/DONE/IFG*/filt*snap_connect.byt

mintpy.load.demFile         = ../PROCESS/DONE/*${ref_date12}*/radar_*rlks.hgt
mintpy.load.lookupYFile     = ../PROCESS/GEO/geo_${ref_date12}/geomap_*rlks.trans
mintpy.load.lookupXFile     = ../PROCESS/GEO/geo_${ref_date12}/geomap_*rlks.trans
mintpy.load.incAngleFile    = None
mintpy.load.azAngleFile     = None
mintpy.load.shadowMaskFile  = None
mintpy.load.bperpFile       = None
'''

AUTO_PATH_GAMMA = '''##----------Default file path of GAMMA products
mintpy.load.processor       = gamma
mintpy.load.unwFile         = ../PROCESS/DONE/IFG*/diff*rlks.unw
mintpy.load.corFile         = ../PROCESS/DONE/IFG*/*filt*rlks.cor
mintpy.load.connCompFile    = None

mintpy.load.demFile         = ../PROCESS/SIM/sim_${ref_date12}/sim*.hgt_sim
mintpy.load.lookupYFile     = ../PROCESS/SIM/sim_${ref_date12}/sim*.UTM_TO_RDC
mintpy.load.lookupXFile     = ../PROCESS/SIM/sim_${ref_date12}/sim*.UTM_TO_RDC
mintpy.load.incAngleFile    = None
mintpy.load.azAngleFile     = None
mintpy.load.shadowMaskFile  = None
mintpy.load.bperpFile       = ../merged/baselines/*/*.base_perp
'''

AUTO_PATH_ARIA = '''##----------Default file path of ARIA products
mintpy.load.processor       = aria
mintpy.load.unwFile         = ../stack/unwrapStack.vrt
mintpy.load.corFile         = ../stack/cohStack.vrt
mintpy.load.connCompFile    = ../stack/connCompStack.vrt

mintpy.load.demFile         = ../DEM/*.dem
mintpy.load.lookupYFile     = None
mintpy.load.lookupXFile     = None
mintpy.load.incAngleFile    = ../incidenceAngle/*.vrt
mintpy.load.azAngleFile     = ../azimuthAngle/*.vrt
mintpy.load.shadowMaskFile  = None
mintpy.load.waterMaskFile   = ../mask/watermask.msk
'''

AUTO_PATH_NISAR = '''##----------Default file path of NISAR products
mintpy.load.unwFile         = ../interferograms/*.h5
mintpy.load.demFile         = ../dem.tiff
'''

AUTO_PATH_DICT = {
    'isce_tops'     : AUTO_PATH_ISCE_TOPS,
    'isce_stripmap' : AUTO_PATH_ISCE_STRIPMAP,
    'isce_alos'     : AUTO_PATH_ISCE_ALOS,
    'roipac'        : AUTO_PATH_ROIPAC,
    'gamma'         : AUTO_PATH_GAMMA,
    'aria'          : AUTO_PATH_ARIA,
    'nisar'         : AUTO_PATH_NISAR,
}

prefix = 'mintpy.load.'


##----------------- Functions from mintpy.utils.readfile to be independent module ---------##
def read_str2dict(inString, delimiter='=', print_msg=False):
    '''Read multiple lines of string into dict
    Based on mintpy.utils.readfile.read_template()
    '''
    strDict = {}
    lines = inString.split('\n')
    for line in lines:
        c = [i.strip() for i in line.strip().split(delimiter, 1)]
        if len(c) < 2 or line.startswith(('%', '#')):
            next
        else:
            key = c[0]
            value = str.replace(c[1], '\n', '').split("#")[0].strip()
            if value != '':
                strDict[key] = value

    # set 'None' to None
    for key, value in strDict.items():
        if value.lower() == 'none':
            strDict[key] = None
    return strDict


##----------------------------------------------------------------------------------------##
def get_auto_path(processor, work_dir, template):
    """Update template options with auto path defined in AUTO_PATH_DICT
    Parameters: processor - str, isce / roipac / gamma
                work_dir  - str, mintpy work directory, e.g. ./GalapagosSenDT128/mintpy
                template  - dict,
    Returns:    template  - dict,
    """
    ## 1. *AutoPath --> auto_path_dict
    proj_dir = os.path.dirname(work_dir)

    # specific stack processor within ISCE
    if processor == 'isce':
        if os.path.exists(proj_dir + '/reference'):
            processor = 'isce_tops'
        elif os.path.exists(proj_dir + '/Igrams'):
            processor = 'isce_stripmap'
        elif os.path.exists(proj_dir + '/dates_resampled'):
            processor = 'isce_alos'
        else:
            raise ValueError('un-recognized ISCE file directory, thus, cannot use auto path setting!')

    # read auto_path_dict
    auto_path_dict = read_str2dict(AUTO_PATH_DICT[processor], print_msg=False)

    ## 2. translate variables in *AutoPath
    ## e.g.: ref_shelve, ref_date12
    var_dict = {}

    if processor in ['roipac', 'gamma']:
        ref_date12 = get_reference_date12(proj_dir, processor)
        if ref_date12:
            var_dict['${ref_date12}'] = ref_date12

        dem_file = get_dem_file(proj_dir, ref_date12, processor)
        if dem_file:
            auto_path_dict[prefix+'demFile'] = dem_file

    elif processor == 'isce_stripmap':
        date_str = os.listdir(os.path.join(proj_dir, 'merged/SLC'))[0]
        var_dict['${ref_shelve}'] = os.path.join(proj_dir, 'merged/SLC', date_str, 'referenceShelve')

    elif processor == 'isce_alos':
        var_dict['${ref_date}'] = get_reference_date(proj_dir)
        var_dict['${multilook}'] = get_multilook_suffix(proj_dir)

    # update auto_path_dict
    for key, value in auto_path_dict.items():
        if value:
            for var1, var2 in var_dict.items():
                value = value.replace(var1, var2)
            auto_path_dict[key] = value

    ## 3. update input template option with auto value
    ## so that one could overwrite part of the auto path with custom template option inputs
    max_digit = max(len(key) for key in auto_path_dict.keys())
    for key, value in auto_path_dict.items():
        if value and template[key] == 'auto':
            template[key] = value
            print('    {k:<{d}} : auto --> {v}'.format(d=max_digit, k=key, v=value))

    return template


def get_multilook_suffix(proj_dir):
    """Get multilook suffix in the interferogram file name for isce2/alosStack from its pairs dir.
    There could be two multilooked versions for most files, the larger one should be used.
    """
    insar_dir = glob.glob(os.path.join(proj_dir, 'pairs/*-*/insar'))[0]
    int_file = sorted(glob.glob(os.path.join(insar_dir, 'filt_*-*.int')))[-1]
    print(f'Grab multilook suffix from file: {int_file}')
    rlks, alks = os.path.splitext(os.path.basename(int_file))[0].split('_')[-2:]
    lks_suffix = f'{rlks}_{alks}'
    print(f'multilook suffix of the interferogram stack: {lks_suffix}')
    return lks_suffix


def get_reference_date(proj_dir):
    """Get reference date of the stack for isce2/alosStack from its XML config file."""
    import xml.etree.ElementTree as ET

    ref_date = None
    ref_key = 'reference date of the stack'

    xml_file = os.path.join(proj_dir, 'alosStack.xml')
    print(f'Grab reference date from the config file: {xml_file}.')
    if not os.path.exists(xml_file):
        raise FileNotFoundError(f'Config file {xml_file} NOT found!')

    # read the XML file
    root = ET.parse(xml_file).getroot()
    for child in root[0].findall('property'):
        key = child.get('name').lower()
        if key.replace(' ', '') == ref_key.replace(' ', ''):
            ref_date = child.text
            print('reference date of the stack:', ref_date)
            break

    if ref_date is None:
        raise ValueError('NO element named: {ref_key} found in the config file!')

    return ref_date


def get_reference_date12(proj_dir, processor='roipac'):
    """date12 of reference interferogram in YYMMDD-YYMMDD format (for UMiami)."""
    import numpy as np

    ref_date12 = None

    # opt 1 - reference_ifgram.txt
    ref_ifg_file = os.path.join(proj_dir, 'PROCESS', 'reference_ifgram.txt')
    if os.path.isfile(ref_ifg_file):
        ref_date12 = str(np.loadtxt(ref_ifg_file, dtype=bytes).astype(str))
        return ref_date12

    # opt 2 - folders under GEO/SIM
    if processor == 'roipac':
        try:
            lookup_file = glob.glob(os.path.join(proj_dir, 'PROCESS/GEO/geo_*/geomap*.trans'))[0]
            ref_date12 = re.findall(r'\d{6}-\d{6}', lookup_file)[0]
        except:
            print("No reference interferogram found! Check the PROCESS/GEO/geo_* folder")

    elif processor == 'gamma':
        geom_dir = os.path.join(proj_dir, 'PROCESS/SIM')
        try:
            ref_date12 = os.walk(geom_dir).next()[1][0].split('sim_')[1]
        except:
            print("No reference interferogram found! Check the PROCESS/SIM/sim_* folder")

    return ref_date12


def get_dem_file(proj_dir, ref_date12, processor):
    """get DEM file in case both radar_2rlks.hgt and radar_8rlks.hgt exist (for UMiami)"""
    dem_file = None

    if ref_date12 and processor == 'roipac':
        # get the number of looks used in lookup table file
        lookup_file = os.path.join(proj_dir, f'PROCESS/GEO/geo_{ref_date12}/geomap*.trans')
        lks = re.findall(r'_\d+rlks', glob.glob(lookup_file)[0])[0]

        # use the one with same multilook info as the lookup table file.
        dem_file = os.path.join(proj_dir, 'PROCESS/DONE/*${ref_date12}*', f'radar{lks}.hgt')

    return dem_file
