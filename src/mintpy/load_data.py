############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, 2013               #
############################################################


import glob
import importlib
import os
import time
import warnings

from mintpy import subset
from mintpy.defaults import auto_path
from mintpy.objects import (
    GEOMETRY_DSET_NAMES,
    IFGRAM_DSET_NAMES,
    TIMESERIES_DSET_NAMES,
    geometry,
    ifgramStack,
    sensor,
)
from mintpy.objects.stackDict import (
    geometryDict,
    ifgramDict,
    ifgramStackDict,
    timeseriesAcqDict,
    timeseriesDict,
)
from mintpy.utils import ptime, readfile, utils as ut

#################################################################
PROCESSOR_LIST = ['isce', 'aria', 'hyp3', 'gmtsar', 'snap', 'gamma', 'roipac', 'cosicorr', 'nisar']

# primary observation dataset names
OBS_DSET_NAMES = ['unwrapPhase', 'rangeOffset', 'azimuthOffset']

IFG_DSET_NAME2TEMPLATE_KEY = {
    'unwrapPhase'     : 'mintpy.load.unwFile',
    'coherence'       : 'mintpy.load.corFile',
    'connectComponent': 'mintpy.load.connCompFile',
    'wrapPhase'       : 'mintpy.load.intFile',
    'magnitude'       : 'mintpy.load.magFile',
}

ION_DSET_NAME2TEMPLATE_KEY = {
    'ionosphericDelay'     : 'mintpy.load.ionFile',
    'ionosphericBurstRamp' : 'mintpy.load.ionBurstRampFile',
}

OFF_DSET_NAME2TEMPLATE_KEY = {
    'azimuthOffset'   : 'mintpy.load.azOffFile',
    'azimuthOffsetStd': 'mintpy.load.azOffStdFile',
    'rangeOffset'     : 'mintpy.load.rgOffFile',
    'rangeOffsetStd'  : 'mintpy.load.rgOffStdFile',
    'offsetSNR'       : 'mintpy.load.offSnrFile',
}

GEOM_DSET_NAME2TEMPLATE_KEY = {
    'height'          : 'mintpy.load.demFile',
    'latitude'        : 'mintpy.load.lookupYFile',
    'longitude'       : 'mintpy.load.lookupXFile',
    'azimuthCoord'    : 'mintpy.load.lookupYFile',
    'rangeCoord'      : 'mintpy.load.lookupXFile',
    'incidenceAngle'  : 'mintpy.load.incAngleFile',
    'azimuthAngle'    : 'mintpy.load.azAngleFile',
    'shadowMask'      : 'mintpy.load.shadowMaskFile',
    'waterMask'       : 'mintpy.load.waterMaskFile',
    'bperp'           : 'mintpy.load.bperpFile',
}


#################################################################
def read_inps2dict(inps):
    """Read input Namespace object info into iDict.

    It grab the following contents into iDict
    1. inps & all template files
    2. configurations: processor, autoPath, updateMode, compression, x/ystep
    3. extra metadata: PLATFORM, PROJECT_NAME,
    4. translate autoPath

    Parameters: inps  - namespace, input arguments from command line & template file
    Returns:    iDict - dict,      input arguments from command line & template file
    """
    # Read input info into iDict
    iDict = vars(inps)
    iDict['PLATFORM'] = None
    iDict['processor'] = 'isce'

    # Read template file
    template = {}
    for fname in inps.template_file:
        temp = readfile.read_template(fname)
        temp = ut.check_template_auto_value(temp)
        template.update(temp)
    for key, value in template.items():
        iDict[key] = value

    # group - load
    prefix = 'mintpy.load.'
    key_list = [i.split(prefix)[1] for i in template.keys() if i.startswith(prefix)]
    for key in key_list:
        value = template[prefix+key]
        if key in ['processor', 'autoPath', 'updateMode', 'compression']:
            iDict[key] = template[prefix+key]
        elif value:
            iDict[prefix+key] = template[prefix+key]
    print('processor : {}'.format(iDict['processor']))

    if iDict['compression'] is False:
        iDict['compression'] = None

    # group - multilook
    prefix = 'mintpy.multilook.'
    key_list = [i.split(prefix)[1] for i in template.keys() if i.startswith(prefix)]
    for key in key_list:
        value = template[prefix+key]
        if key in ['xstep', 'ystep', 'method']:
            iDict[key] = template[prefix+key]

    iDict['xstep']  = int(iDict.get('xstep', 1))
    iDict['ystep']  = int(iDict.get('ystep', 1))
    iDict['method'] = str(iDict.get('method', 'nearest'))

    # PROJECT_NAME --> PLATFORM
    if not iDict['PROJECT_NAME']:
        cfile = [i for i in list(inps.template_file) if os.path.basename(i) != 'smallbaselineApp.cfg']
        iDict['PROJECT_NAME'] = sensor.project_name2sensor_name(cfile)[1]

    msg = 'SAR platform/sensor : '
    sensor_name = sensor.project_name2sensor_name(str(iDict['PROJECT_NAME']))[0]
    if sensor_name:
        msg += str(sensor_name)
        iDict['PLATFORM'] = str(sensor_name)
    else:
        msg += 'unknown from project name "{}"'.format(iDict['PROJECT_NAME'])
    print(msg)

    # update file path with auto
    if iDict.get('autoPath', False):
        print('use auto path defined in mintpy.defaults.auto_path for options in auto')
        iDict = auto_path.get_auto_path(processor=iDict['processor'],
                                        work_dir=os.getcwd(),
                                        template=iDict)

    return iDict


def read_subset_box(iDict):
    """read the following items:
    geocoded - bool, if the stack of observations geocoded or not
    box      - tuple of 4 int, pixel box for stackObj and geomRadarObj, for obs in geo & radar coordinates
    box4geo  - tuple of 4 int, pixel box for geomGeoObj, box4geo is the same as box, except for:
               obs in radar coordinate with lookup table [for gamma and roipac], where box4geo is
               the geo bounding box of the box above.
    """
    # Read subset info from template
    iDict['box'] = None
    iDict['box4geo'] = None
    pix_box, geo_box = subset.read_subset_template2box(iDict['template_file'][0])

    # Grab required info to read input geo_box into pix_box
    lookup_y_files = glob.glob(str(iDict['mintpy.load.lookupYFile']))
    lookup_x_files = glob.glob(str(iDict['mintpy.load.lookupXFile']))
    if len(lookup_y_files) > 0 and len(lookup_x_files) > 0:
        lookup_file = [lookup_y_files[0], lookup_x_files[0]]
    else:
        lookup_file = None

    # use DEM file attribute as reference, because
    # 1) it is required AND
    # 2) it is in the same coordinate type as observation files
    dem_files = glob.glob(iDict['mintpy.load.demFile'])
    if len(dem_files) > 0:
        atr = readfile.read_attribute(dem_files[0])
    else:
        atr = dict()

    geocoded = True if 'Y_FIRST' in atr.keys() else False
    iDict['geocoded'] = geocoded

    # Check conflict
    if geo_box and not geocoded and lookup_file is None:
        geo_box = None
        print('WARNING: mintpy.subset.lalo is not supported'
              ' if 1) no lookup file AND'
              '    2) radar/unknown coded dataset')
        print('\tignore it and continue.')

    if not geo_box and not pix_box:
        # adjust for the size inconsistency problem in SNAP geocoded products
        # ONLY IF there is no input subset
        # Use the min bbox if files size are different
        if iDict['processor'] == 'snap':
            fnames = ut.get_file_list(iDict['mintpy.load.unwFile'])
            pix_box = update_box4files_with_inconsistent_size(fnames)

        if not pix_box:
            return iDict

    # geo_box --> pix_box
    coord = ut.coordinate(atr, lookup_file=lookup_file)
    if geo_box is not None:
        pix_box = coord.bbox_geo2radar(geo_box)
        pix_box = coord.check_box_within_data_coverage(pix_box)
        print(f'input bounding box of interest in lalo: {geo_box}')
    print(f'box to read for datasets in y/x: {pix_box}')

    # Get box for geocoded lookup table (for gamma/roipac)
    box4geo_lut = None
    if lookup_file is not None:
        atrLut = readfile.read_attribute(lookup_file[0])
        if not geocoded and 'Y_FIRST' in atrLut.keys():
            geo_box = coord.bbox_radar2geo(pix_box)
            box4geo_lut = ut.coordinate(atrLut).bbox_geo2radar(geo_box)
            print(f'box to read for geocoded lookup file in y/x: {box4geo_lut}')

    iDict['box'] = pix_box
    iDict['box4geo'] = box4geo_lut if box4geo_lut else pix_box
    return iDict


#################################################################
def update_box4files_with_inconsistent_size(fnames):
    """Check the size (row / column number) of a list of files
    For SNAP geocoded products has one line missing in some interferograms, Andre, 2019-07-16
    Parameters: fnames  - list of path for interferogram files
    Returns:    pix_box - None if all files are in same size
                          (0, 0, min_width, min_length) if not.
    """
    atr_list = [readfile.read_attribute(fname) for fname in fnames]
    length_list = [int(atr['LENGTH']) for atr in atr_list]
    width_list = [int(atr['WIDTH']) for atr in atr_list]
    if any(len(set(i)) for i in [length_list, width_list]):
        min_length = min(length_list)
        min_width = min(width_list)
        pix_box = (0, 0, min_width, min_length)

        # print out warning message
        msg = '\n'+'*'*80
        msg += '\nWARNING: NOT all input unwrapped interferograms have the same row/column number!'
        msg += f'\nMinimum size is: ({min_length}, {min_width})'
        msg += '\n'+'-'*30
        msg += '\nThe following dates have different size:'

        for i in range(len(fnames)):
            if length_list[i] != min_length or width_list[i] != min_width:
                msg += '\n\t{}\t({}, {})'.format(atr_list[i]['DATE12'], length_list[i], width_list[i])

        msg += '\n'+'-'*30
        msg += '\nAssuming the interferograms above have:'
        msg += '\n\textra line(s) at the bottom OR'
        msg += '\n\textra column(s) at the right'
        msg += '\nContinue to load data using subset of the minimum size.'
        msg += '\n'+'*'*80+'\n'
        print(msg)
    else:
        pix_box = None
    return pix_box


def skip_files_with_inconsistent_size(dsPathDict, pix_box=None, dsName='unwrapPhase'):
    """Skip files by removing the file path from the input dsPathDict."""
    atr_list = [readfile.read_attribute(fname) for fname in dsPathDict[dsName]]
    length_list = [int(atr['LENGTH']) for atr in atr_list]
    width_list = [int(atr['WIDTH']) for atr in atr_list]

    # Check size requirements
    drop_inconsistent_files = False
    if any(len(set(size_list)) > 1 for size_list in [length_list, width_list]):
        if pix_box is None:
            drop_inconsistent_files = True
        else:
            # if input subset is within the min file sizes: do NOT drop
            max_box_width, max_box_length = pix_box[2:4]
            if max_box_length > min(length_list) or max_box_width > min(width_list):
                drop_inconsistent_files = True

    # update dsPathDict
    if drop_inconsistent_files:
        common_length = ut.most_common(length_list)
        common_width = ut.most_common(width_list)

        # print out warning message
        msg = '\n'+'*'*80
        msg += '\nWARNING: NOT all input unwrapped interferograms have the same row/column number!'
        msg += f'\nThe most common size is: ({common_length}, {common_width})'
        msg += '\n'+'-'*30
        msg += '\nThe following dates have different size:'

        dsNames = list(dsPathDict.keys())
        date12_list = [atr['DATE12'] for atr in atr_list]
        num_drop = 0
        for date12, length, width in zip(date12_list, length_list, width_list):
            if length != common_length or width != common_width:
                dates = ptime.yyyymmdd(date12.split('-'))
                # update file list for all datasets
                for dsName in dsNames:
                    fnames = [i for i in dsPathDict[dsName]
                              if all(d[2:8] in i for d in dates)]
                    if len(fnames) > 0:
                        dsPathDict[dsName].remove(fnames[0])
                msg += f'\n\t{date12}\t({length}, {width})'
                num_drop += 1

        msg += '\n'+'-'*30
        msg += f'\nSkip loading the above interferograms ({num_drop}).'
        msg += f'\nContinue to load the rest interferograms ({len(date12_list) - num_drop}).'
        msg += '\n'+'*'*80+'\n'
        print(msg)
    return dsPathDict


def read_inps_dict2ifgram_stack_dict_object(iDict, ds_name2template_key):
    """Read input arguments into ifgramStackDict object.

    Parameters: iDict                - dict, input arguments from command line & template file
                ds_name2template_key - dict, to relate the HDF5 dataset name to the template key
    Returns:    stackObj             - ifgramStackDict object or None
    """
    if not iDict['load_ifg']:
        return None

    if 'mintpy.load.unwFile' in ds_name2template_key.values():
        obs_type = 'interferogram'
    elif 'mintpy.load.azOffFile' in ds_name2template_key.values():
        obs_type = 'offset'

    # iDict --> dsPathDict
    print('-'*50)
    print(f'searching {obs_type} pairs info')
    print('input data files:')
    max_digit = max(len(i) for i in list(ds_name2template_key.keys()))
    dsPathDict = {}
    for dsName in [i for i in IFGRAM_DSET_NAMES if i in ds_name2template_key.keys()]:
        key = ds_name2template_key[dsName]
        if key in iDict.keys():
            files = sorted(glob.glob(str(iDict[key])))
            if len(files) > 0:
                dsPathDict[dsName] = files
                print(f'{dsName:<{max_digit}}: {iDict[key]}')

    # Check 1: required dataset
    dsName0s = [x for x in OBS_DSET_NAMES if x in ds_name2template_key.keys()]
    dsName0 = [i for i in dsName0s if i in dsPathDict.keys()]
    if len(dsName0) == 0:
        print(f'WARNING: No data files found for the required dataset: {dsName0s}! Skip loading for {obs_type} stack.')
        return None
    else:
        dsName0 = dsName0[0]

    # Check 2: data dimension for unwrapPhase files
    dsPathDict = skip_files_with_inconsistent_size(
        dsPathDict=dsPathDict,
        pix_box=iDict['box'],
        dsName=dsName0)

    # Check 3: number of files for all dataset types
    # dsPathDict --> dsNumDict
    dsNumDict = {}
    for key in dsPathDict.keys():
        num_file = len(dsPathDict[key])
        dsNumDict[key] = num_file
        print(f'number of {key:<{max_digit}}: {num_file}')

    dsNumList = list(dsNumDict.values())
    if any(i != dsNumList[0] for i in dsNumList):
        msg = 'WARNING: NOT all types of dataset have the same number of files.'
        msg += ' -> skip interferograms with missing files and continue.'
        print(msg)
        #raise Exception(msg)

    # dsPathDict --> pairsDict --> stackObj
    dsNameList = list(dsPathDict.keys())

    #####################################
    # A dictionary of data file paths for a list of pairs, e.g.:
    # pairsDict = {
    #     ('date1', 'date2') : ifgramPathDict1,
    #     ('date1', 'date3') : ifgramPathDict2,
    #     ...,
    # }

    pairsDict = {}
    for i, dsPath0 in enumerate(dsPathDict[dsName0]):
        # date string used in the file/dir path
        # YYYYDDD       for gmtsar [day of the year - 1]
        # YYYYMMDDTHHMM for uavsar
        # YYYYMMDD      for all the others
        date6s = readfile.read_attribute(dsPath0)['DATE12'].replace('_','-').split('-')
        if iDict['processor'] == 'gmtsar':
            date12MJD = os.path.basename(os.path.dirname(dsPath0))
        else:
            date12MJD = None

        #####################################
        # A dictionary of data file paths for a given pair.
        # One pair may have several types of dataset, e.g.:
        # ifgramPathDict1 = {
        #     'unwrapPhase': /dirPathToFile/filt_fine.unw,
        #     'coherence'  : /dirPathToFile/filt_fine.cor,
        #     ...
        # }
        # All path of data file must contain the reference and secondary date, in file/dir name.

        ifgramPathDict = {}
        for dsName in dsNameList:
            # search the matching data file for the given date12
            # 1st guess: file in the same order as the one for dsName0
            dsPath1 = dsPathDict[dsName][i]
            if (all(d6 in dsPath1 for d6 in date6s)
                    or (date12MJD and date12MJD in dsPath1)):
                ifgramPathDict[dsName] = dsPath1

            else:
                # 2nd guess: any file in the list
                dsPath2 = [p for p in dsPathDict[dsName]
                           if (all(d6 in p for d6 in date6s)
                                   or (date12MJD and date12MJD in dsPath1))]

                if len(dsPath2) > 0:
                    ifgramPathDict[dsName] = dsPath2[0]
                else:
                    print(f'WARNING: {dsName:>18} file missing for pair {date6s}')

        # initiate ifgramDict object
        ifgramObj = ifgramDict(datasetDict=ifgramPathDict)

        # update pairsDict object
        date8s = ptime.yyyymmdd(date6s)
        pairsDict[tuple(date8s)] = ifgramObj

    if len(pairsDict) > 0:
        stackObj = ifgramStackDict(pairsDict=pairsDict, dsName0=dsName0)
    else:
        stackObj = None
    return stackObj


def read_inps_dict2timeseries_dict_object(iDict, ds_name2tmpl_opt):
    """Read input arguments into timeseriesDict object(s).

    Parameters: iDict                - dict, input arguments from command line & template file
                ds_name2template_key - dict, to relate the HDF5 dataset name to the template key
    Returns:    timeseriesObj        - timeseriesDict object or None
    """
    if not iDict['load_ion']:
        return None

    if 'mintpy.load.unwFile' in ds_name2tmpl_opt.values():
        obs_type = 'interferogram'
    elif 'mintpy.load.ionFile' in ds_name2tmpl_opt.values():
        obs_type = 'ionosphere'
    elif 'mintpy.load.ionBurstRampFile' in ds_name2tmpl_opt.values():
        obs_type = 'ionosphere burst ramp'
    elif 'mintpy.load.azOffFile' in ds_name2tmpl_opt.values():
        obs_type = 'offset'

    # iDict --> dsPathDict
    print('-'*50)
    print(f'searching {obs_type} pairs info')
    print('input data files:')
    max_digit = max(len(i) for i in list(ds_name2tmpl_opt.keys()))
    dsPathDict = {}
    for dsName in [i for i in TIMESERIES_DSET_NAMES if i in ds_name2tmpl_opt.keys()]:
        key = ds_name2tmpl_opt[dsName]
        if key in iDict.keys():
            files = sorted(glob.glob(str(iDict[key])))
            if len(files) > 0:
                dsPathDict[dsName] = files
                print(f'{dsName:<{max_digit}}: {iDict[key]}')

    # Check 1: required dataset
    dsName0s = [x for x in TIMESERIES_DSET_NAMES if x in ds_name2tmpl_opt.keys()]
    dsName0 = [i for i in dsName0s if i in dsPathDict.keys()]
    if len(dsName0) == 0:
        print(f'WARNING: No data files found for the required dataset: {dsName0s}! Skip loading for {obs_type} stack.')
        return None
    else:
        dsName0 = dsName0[0]

    # Check 2: data dimension for unwrapPhase files
    dsPathDict = skip_files_with_inconsistent_size(
        dsPathDict=dsPathDict,
        pix_box=iDict['box'],
        dsName=dsName0)

    # Check 3: number of files for all dataset types
    # dsPathDict --> dsNumDict
    dsNumDict = {}
    for key in dsPathDict.keys():
        num_file = len(dsPathDict[key])
        dsNumDict[key] = num_file
        print(f'number of {key:<{max_digit}}: {num_file}')

    dsNumList = list(dsNumDict.values())
    if any(i != dsNumList[0] for i in dsNumList):
        msg = 'WARNING: NOT all types of dataset have the same number of files.'
        msg += ' -> skip interferograms with missing files and continue.'
        print(msg)
        raise Exception(msg)

    # dsPathDict --> pairsDict --> tsObj
    dsNameList = list(dsPathDict.keys())



    #####################################

    datesDict = {}
    for i, dsPath0 in enumerate(dsPathDict[dsName0]):
        date8s = os.path.basename(dsPath0).split('.')[0]

        tsPathDict = {}
        for dsName in dsNameList:
            # search the matching data file for the given date12
            dsPath1 = dsPathDict[dsName][i]
            if date8s in dsPath1:
                tsPathDict[dsName] = dsPath1


        # initiate timeseriesAcqDict object
        acqObj = timeseriesAcqDict(datasetDict=tsPathDict, metadata={'data_unit': 'radian'})

        # update datesDict object
        datesDict[date8s] = acqObj

    if len(datesDict) > 0:
        tsObj = timeseriesDict(datesDict=datesDict, dsName0=dsName0)
    else:
        tsObj = None
    return tsObj


def read_inps_dict2geometry_dict_object(iDict, dset_name2template_key):
    """Read input arguments into geometryDict object(s).

    Parameters: iDict        - dict, input arguments from command line & template file
    Returns:    geomGeoObj   - geometryDict object in geo   coordinates or None
                geomRadarObj - geometryDict object in radar coordinates or None
    """
    if not iDict['load_geom']:
        return None

    # eliminate lookup table dsName for input files in radar-coordinates
    if iDict['processor'] in ['isce', 'doris']:
        # for processors with lookup table in radar-coordinates, remove azimuth/rangeCoord
        dset_name2template_key.pop('azimuthCoord')
        dset_name2template_key.pop('rangeCoord')
    elif iDict['processor'] in ['roipac', 'gamma']:
        # for processors with lookup table in geo-coordinates, remove latitude/longitude
        dset_name2template_key.pop('latitude')
        dset_name2template_key.pop('longitude')
    elif iDict['processor'] in ['aria', 'gmtsar', 'hyp3', 'snap', 'cosicorr']:
        # for processors with geocoded products support only, do nothing for now.
        # check again when adding products support in radar-coordiantes
        pass
    else:
        print('Un-recognized InSAR processor: {}'.format(iDict['processor']))

    # iDict --> dsPathDict
    print('-'*50)
    print('searching geometry files info')
    print('input data files:')
    max_digit = max(len(i) for i in list(dset_name2template_key.keys()))
    dsPathDict = {}
    for dsName in [i for i in GEOMETRY_DSET_NAMES if i in dset_name2template_key.keys()]:
        key = dset_name2template_key[dsName]
        if key in iDict.keys():
            files = sorted(glob.glob(str(iDict[key])))
            if len(files) > 0:
                if dsName == 'bperp':
                    bperpDict = {}
                    for file in files:
                        date = ptime.yyyymmdd(os.path.basename(os.path.dirname(file)))
                        bperpDict[date] = file
                    dsPathDict[dsName] = bperpDict
                    print(f'{dsName:<{max_digit}}: {iDict[key]}')
                    print(f'number of bperp files: {len(list(bperpDict.keys()))}')
                else:
                    dsPathDict[dsName] = files[0]
                    print(f'{dsName:<{max_digit}}: {files[0]}')

    # Check required dataset
    dsName0 = GEOMETRY_DSET_NAMES[0]
    if dsName0 not in dsPathDict.keys():
        print(f'WARNING: No reqired {dsName0} data files found!')

    # extra metadata from observations
    # e.g. EARTH_RADIUS, HEIGHT, etc.
    obsMetaGeo = None
    obsMetaRadar = None
    for obsName in OBS_DSET_NAMES:
        obsFiles = sorted(glob.glob(iDict[dset_name2template_key[obsName]]))
        if len(obsFiles) > 0:
            atr = readfile.read_attribute(obsFiles[0])
            if 'Y_FIRST' in atr.keys():
                obsMetaGeo = atr.copy()
            else:
                obsMetaRadar = atr.copy()
            break

    # dsPathDict --> dsGeoPathDict + dsRadarPathDict
    dsNameList = list(dsPathDict.keys())
    dsGeoPathDict = {}
    dsRadarPathDict = {}
    for dsName in dsNameList:
        if dsName == 'bperp':
            atr = readfile.read_attribute(next(iter(dsPathDict[dsName].values())))
        else:
            atr = readfile.read_attribute(dsPathDict[dsName])
        if 'Y_FIRST' in atr.keys():
            dsGeoPathDict[dsName] = dsPathDict[dsName]
        else:
            dsRadarPathDict[dsName] = dsPathDict[dsName]

    geomGeoObj = None
    geomRadarObj = None
    if len(dsGeoPathDict) > 0:
        geomGeoObj = geometryDict(
            processor=iDict['processor'],
            datasetDict=dsGeoPathDict,
            extraMetadata=obsMetaGeo)
    if len(dsRadarPathDict) > 0:
        geomRadarObj = geometryDict(
            processor=iDict['processor'],
            datasetDict=dsRadarPathDict,
            extraMetadata=obsMetaRadar)

    return geomGeoObj, geomRadarObj


#################################################################
def run_or_skip(outFile, inObj, box, updateMode=True, xstep=1, ystep=1, geom_obj=None):
    """Check if re-writing is necessary.

    Do not write HDF5 file if ALL the following meet:
        1. HDF5 file exists and is readable,
        2. HDF5 file contains all the datasets and in the same size
        3. For ifgramStackDict, HDF5 file contains all date12.

    Parameters: outFile    - str, path to the output HDF5 file
                inObj      - ifgramStackDict or geometryDict, object to write
                box        - tuple of int, bounding box in (x0, y0, x1, y1)
                updateMode - bool
                x/ystep    - int
                geom_obj   - geometryDict object or None, for ionosphere only
    Returns:    flag       - str, run or skip
    """

    flag = 'run'

    # skip if there is no dict object to write
    if not inObj:
        flag = 'skip'
        return flag

    # run if not in update mode
    if not updateMode:
        return flag

    if ut.run_or_skip(outFile, readable=True) == 'skip':
        kwargs = dict(box=box, xstep=xstep, ystep=ystep)

        if inObj.name == 'ifgramStack':
            in_size = inObj.get_size(geom_obj=geom_obj, **kwargs)[1:]
            in_dset_list = inObj.get_dataset_list()
            in_date12_list = inObj.get_date12_list()

            outObj = ifgramStack(outFile)
            outObj.open(print_msg=False)
            out_size = (outObj.length, outObj.width)
            out_dset_list = outObj.datasetNames
            out_date12_list = outObj.date12List

            if (out_size[1:] == in_size[1:]
                    and set(in_dset_list).issubset(set(out_dset_list))
                    and set(in_date12_list).issubset(set(out_date12_list))):
                print('All date12   exists in file {} with same size as required,'
                      ' no need to re-load.'.format(os.path.basename(outFile)))
                flag = 'skip'

        elif inObj.name == 'geometry':
            in_size = inObj.get_size(**kwargs)
            in_dset_list = inObj.get_dataset_list()

            outObj = geometry(outFile)
            outObj.open(print_msg=False)
            out_size = (outObj.length, outObj.width)
            out_dset_list = outObj.datasetNames

            if (out_size == in_size
                    and set(in_dset_list).issubset(set(out_dset_list))):
                print('All datasets exists in file {} with same size as required,'
                      ' no need to re-load.'.format(os.path.basename(outFile)))
                flag = 'skip'

    return flag


def prepare_metadata(iDict):
    """Prepare metadata via prep_{processor}.py scripts."""
    processor = iDict['processor']
    script_name = f'prep_{processor}.py'
    print('-'*50)
    print(f'prepare metadata files for {processor} products')

    if processor not in PROCESSOR_LIST:
        msg = f'un-recognized InSAR processor: {processor}'
        msg += f'\nsupported processors: {PROCESSOR_LIST}'
        raise ValueError(msg)

    # import prep_{processor}
    prep_module = importlib.import_module(f'mintpy.cli.prep_{processor}')

    if processor in ['gamma', 'hyp3', 'roipac', 'snap', 'cosicorr']:
        # run prep_module
        for key in [i for i in iDict.keys()
                    if (i.startswith('mintpy.load.')
                        and i.endswith('File')
                        and i != 'mintpy.load.metaFile')]:
            if len(glob.glob(str(iDict[key]))) > 0:
                # print command line
                iargs = [iDict[key]]
                if processor == 'gamma':
                    if iDict['PLATFORM']:
                        iargs += ['--sensor', iDict['PLATFORM'].lower()]
                    # add DEM file to faciliate the checking and metadata extraction for geocoded datasets
                    iargs += ['--dem', iDict['mintpy.load.demFile']]

                elif processor == 'cosicorr':
                    iargs += ['--metadata', iDict['mintpy.load.metaFile']]
                ut.print_command_line(script_name, iargs)
                # run
                prep_module.main(iargs)

    elif processor == 'nisar':
        dem_file = iDict['mintpy.load.demFile']
        gunw_files = iDict['mintpy.load.unwFile']
        water_mask = iDict['mintpy.load.waterMaskFile']

        # run prep_*.py
        iargs = ['-i', gunw_files, '-d', dem_file]

        if os.path.exists(water_mask):
            iargs = iargs + ['--mask', water_mask]

        if iDict['mintpy.subset.yx']:
            warnings.warn('Subset in Y/X is not implemented for NISAR. \n'
                          'There might be shift in the coordinates of different products. \n'
                          'Use lat/lon instead.')
        if iDict['mintpy.subset.lalo']:
            lalo = iDict['mintpy.subset.lalo'].split(',')
            lats = lalo[0].split(':')
            lons = lalo[1].split(':')
            iargs = iargs + ['--sub-lat', lats[0], lats[1], '--sub-lon', lons[0], lons[1]]

        ut.print_command_line(script_name, iargs)
        try:
            prep_module.main(iargs)
        except:
            warnings.warn('prep_nisar.py failed. Assuming its result exists and continue...')

    elif processor == 'isce':
        from mintpy.utils import isce_utils, s1_utils

        # --meta-file
        meta_files = sorted(glob.glob(iDict['mintpy.load.metaFile']))
        if len(meta_files) > 0:
            meta_file = meta_files[0]
        else:
            warnings.warn('No input metadata file found: {}'.format(iDict['mintpy.load.metaFile']))
            meta_file = 'auto'

        # --baseline-dir / --geometry-dir
        baseline_dir = iDict['mintpy.load.baselineDir']
        geom_dir = os.path.dirname(iDict['mintpy.load.demFile'])
        geom_dir = os.path.abspath(geom_dir)

        # --dset-dir / --file-pattern
        obs_keys = [
            'mintpy.load.unwFile',
            'mintpy.load.ionFile',
            'mintpy.load.ionBurstRampFile',
            'mintpy.load.rgOffFile',
            'mintpy.load.azOffFile',
        ]
        obs_paths = [iDict[key] for key in obs_keys if iDict[key].lower() != 'auto']
        obs_paths = [x for x in obs_paths if len(glob.glob(x)) > 0]

        # --geom-files for the basenames only
        geom_names = ['dem', 'lookupY', 'lookupX', 'incAngle', 'azAngle', 'shadowMask', 'waterMask']
        geom_keys = [f'mintpy.load.{i}File' for i in geom_names]
        geom_files = [os.path.basename(iDict[key]) for key in geom_keys
                      if iDict.get(key, 'auto') not in ['auto', 'None', 'no',  None, False]]

        # compose list of input arguments
        iargs = ['-m', meta_file, '-g', geom_dir]
        if baseline_dir:
            iargs += ['-b', baseline_dir]
        if len(obs_paths) > 0:
            iargs += ['-f'] + obs_paths
        if geom_files:
            iargs += ['--geom-files'] + geom_files

        # run module
        ut.print_command_line(script_name, iargs)
        try:
            prep_module.main(iargs)
        except:
            warnings.warn('prep_isce.py failed. Assuming its result exists and continue...')

        # [optional] for topsStack: SAFE_files.txt --> S1A/B_date.txt
        if os.path.isfile(meta_file) and isce_utils.get_processor(meta_file) == 'topsStack':
            safe_list_file = os.path.join(os.path.dirname(os.path.dirname(meta_file)), 'SAFE_files.txt')
            if os.path.isfile(safe_list_file):
                s1_utils.get_s1ab_date_list_file(
                    mintpy_dir=os.getcwd(),
                    safe_list_file=safe_list_file,
                    print_msg=True)

    elif processor == 'aria':
        ## compose input arguments
        # use the default template file if exists & input
        default_temp_files = [fname for fname in iDict['template_file']
                              if fname.endswith('smallbaselineApp.cfg')]
        if len(default_temp_files) > 0:
            temp_file = default_temp_files[0]
        else:
            temp_file = iDict['template_file'][0]
        iargs = ['--template', temp_file]

        # file name/dir/path
        ARG2OPT_DICT = {
            '--stack-dir'           : 'mintpy.load.unwFile',
            '--unwrap-stack-name'   : 'mintpy.load.unwFile',
            '--coherence-stack-name': 'mintpy.load.corFile',
            '--conn-comp-stack-name': 'mintpy.load.connCompFile',
            '--dem'                 : 'mintpy.load.demFile',
            '--incidence-angle'     : 'mintpy.load.incAngleFile',
            '--azimuth-angle'       : 'mintpy.load.azAngleFile',
            '--water-mask'          : 'mintpy.load.waterMaskFile',
        }

        for arg_name, opt_name in ARG2OPT_DICT.items():
            arg_value = iDict.get(opt_name, 'auto')
            if arg_value.lower() not in ['auto', 'no', 'none']:
                if arg_name.endswith('dir'):
                    iargs += [arg_name, os.path.dirname(arg_value)]
                elif arg_name.endswith('name'):
                    iargs += [arg_name, os.path.basename(arg_value)]
                else:
                    iargs += [arg_name, arg_value]

        # configurations
        if iDict['compression']:
            iargs += ['--compression', iDict['compression']]
        if iDict['updateMode']:
            iargs += ['--update']

        ## run
        ut.print_command_line(script_name, iargs)
        prep_module.main(iargs)

    elif processor == 'gmtsar':
        # use the custom template file if exists & input
        custom_temp_files = [fname for fname in iDict['template_file']
                             if not fname.endswith('smallbaselineApp.cfg')]
        if len(custom_temp_files) == 0:
            raise FileExistsError('Custom template file NOT found and is required for GMTSAR!')

        # run prep_*.py
        iargs = [custom_temp_files[0]]
        ut.print_command_line(script_name, iargs)
        try:
            prep_module.main(iargs)
        except:
            warnings.warn('prep_gmtsar.py failed. Assuming its result exists and continue...')

    return


def get_extra_metadata(iDict):
    """Extra metadata with key names in MACRO_CASE to be written into stack file.

    Parameters: iDict     - dict, input arguments from command lines & template file
                extraDict - dict, extra metadata from template file:
                            E.g. PROJECT_NAME, PLATFORM, ORBIT_DIRECTION, SUBSET_X/YMIN, etc.
    """
    extraDict = {}
    # all keys in MACRO_CASE
    upper_keys = [i for i in iDict.keys() if i.isupper()]
    for key in upper_keys:
        value  = iDict[key]
        if key in ['PROJECT_NAME', 'PLATFORM']:
            if value:
                extraDict[key] = value
        else:
            extraDict[key] = value
    return extraDict


#################################################################
def load_data(inps):
    """load data into HDF5 files."""

    ## 0. read input
    start_time = time.time()
    iDict = read_inps2dict(inps)

    ## 1. prepare metadata
    prepare_metadata(iDict)
    extraDict = get_extra_metadata(iDict)

    # skip data writing as it is included in prep_aria/nisar
    if iDict['processor'] in ['aria', 'nisar']:
        return

    ## 2. search & write data files
    print('-'*50)
    print('updateMode : {}'.format(iDict['updateMode']))
    print('compression: {}'.format(iDict['compression']))
    print('multilook x/ystep: {}/{}'.format(iDict['xstep'], iDict['ystep']))
    print('multilook method : {}'.format(iDict['method']))
    kwargs = dict(updateMode=iDict['updateMode'], xstep=iDict['xstep'], ystep=iDict['ystep'])

    # read subset info [need the metadata from above]
    iDict = read_subset_box(iDict)

    # read specific datasets
    iDict['load_ifg']=True   if 'ifg'  in iDict['listDset']   else False
    iDict['load_geom']=True  if 'geom' in iDict['listDset']   else False
    iDict['load_ion']=True   if 'ion'  in iDict['listDset']   else False

    # 3. geometry in geo / radar coordinates
    geom_dset_name2template_key = {
        **GEOM_DSET_NAME2TEMPLATE_KEY,
        **IFG_DSET_NAME2TEMPLATE_KEY,
        **OFF_DSET_NAME2TEMPLATE_KEY,
    }
    geom_geo_obj, geom_radar_obj = read_inps_dict2geometry_dict_object(iDict, geom_dset_name2template_key)
    geom_geo_file = os.path.abspath('./inputs/geometryGeo.h5')
    geom_radar_file = os.path.abspath('./inputs/geometryRadar.h5')

    if run_or_skip(geom_geo_file, geom_geo_obj, iDict['box4geo'], **kwargs) == 'run':
        geom_geo_obj.write2hdf5(
            outputFile=geom_geo_file,
            access_mode='w',
            box=iDict['box4geo'],
            xstep=iDict['xstep'],
            ystep=iDict['ystep'],
            compression='lzf')

    if run_or_skip(geom_radar_file, geom_radar_obj, iDict['box'], **kwargs) == 'run':
        geom_radar_obj.write2hdf5(
            outputFile=geom_radar_file,
            access_mode='w',
            box=iDict['box'],
            xstep=iDict['xstep'],
            ystep=iDict['ystep'],
            compression='lzf',
            extra_metadata=extraDict)

    # 4. observations: ifgram or offset
    # loop over obs stacks
    stack_ds_name2tmpl_key_list = [
        IFG_DSET_NAME2TEMPLATE_KEY,
        OFF_DSET_NAME2TEMPLATE_KEY,
    ]
    stack_files = ['ifgramStack.h5', 'offsetStack.h5']
    stack_files = [os.path.abspath(os.path.join('./inputs', x)) for x in stack_files]
    for ds_name2tmpl_opt, stack_file in zip(stack_ds_name2tmpl_key_list, stack_files):

        # initiate dict objects
        stack_obj = read_inps_dict2ifgram_stack_dict_object(iDict, ds_name2tmpl_opt)

        geom_obj = None

        # write dict objects to HDF5 files
        if run_or_skip(stack_file, stack_obj, iDict['box'], geom_obj=geom_obj, **kwargs) == 'run':
            stack_obj.write2hdf5(
                outputFile=stack_file,
                access_mode='w',
                box=iDict['box'],
                xstep=iDict['xstep'],
                ystep=iDict['ystep'],
                mli_method=iDict['method'],
                compression=iDict['compression'],
                extra_metadata=extraDict,
                geom_obj=geom_obj)


    # 5. observations: ionosphericDelay, ionosphericBurstRamp
    # loop over obs of timeseries
    stack_ds_name2tmpl_key_list = [
        {k: v} for k, v in ION_DSET_NAME2TEMPLATE_KEY.items()
    ]
    ts_files = ['ion.h5', 'ionBurstRamp.h5']
    ts_files = [os.path.abspath(os.path.join('./inputs', x)) for x in ts_files]
    for (ds_name2tmpl_opt, ts_file) in zip(stack_ds_name2tmpl_key_list, ts_files):
        # rename the object key to timeseries
        for k in ION_DSET_NAME2TEMPLATE_KEY.keys():
            if k in ds_name2tmpl_opt:
                ds_name2tmpl_opt[TIMESERIES_DSET_NAMES[0]] = ds_name2tmpl_opt[k]
                del ds_name2tmpl_opt[k]

        # initiate dict objects (a new func getting timeseries_obj)
        ts_obj = read_inps_dict2timeseries_dict_object(iDict, ds_name2tmpl_opt)

        # use geom_obj as size reference while loading ionosphere
        geom_obj = None
        if os.path.basename(ts_file).startswith('ion'):
            geom_obj = geom_geo_obj if iDict['geocoded'] else geom_radar_obj

        # write dict objects to HDF5 files
        if run_or_skip(ts_file, ts_obj, iDict['box'], geom_obj=geom_obj, **kwargs) == 'run':
            ts_obj.write2hdf5(
                outputFile=ts_file,
                access_mode='w',
                box=iDict['box'],
                xstep=iDict['xstep'],
                ystep=iDict['ystep'],
                mli_method=iDict['method'],
                compression=iDict['compression'],
                extra_metadata=extraDict,
                geom_obj=geom_obj)

    # used time
    m, s = divmod(time.time()-start_time, 60)
    print(f'time used: {m:02.0f} mins {s:02.1f} secs.\n')

    return
