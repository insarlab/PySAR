#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Antonio Valentino, Heresh Fattahi, Aug 2022      #
############################################################


import glob
import os
import sys

from mintpy.utils.arg_utils import create_argument_parser

####################################################################################
TEMPLATE = """template options:
  ########## 1. load_data
  ## no   - save   0% disk usage, fast [default]
  ## lzf  - save ~57% disk usage, relative slow
  ## gzip - save ~62% disk usage, very slow [not recommend]
  mintpy.load.processor      = aria  #[isce, aria, snap, gamma, roipac], auto for isce
  mintpy.load.updateMode     = auto  #[yes / no], auto for yes, skip re-loading if HDF5 files are complete
  mintpy.load.compression    = auto  #[gzip / lzf / no], auto for no.
  ##---------interferogram datasets:
  mintpy.load.unwFile        = ../stack/unwrapStack.vrt
  mintpy.load.corFile        = ../stack/cohStack.vrt
  mintpy.load.connCompFile   = ../stack/connCompStack.vrt
  mintpy.load.magFile        = ../stack/ampStack.vrt        # optional
  ##---------geometry datasets:
  mintpy.load.demFile        = ../DEM/SRTM_3arcsec.dem
  mintpy.load.incAngleFile   = ../incidenceAngle/*.vrt
  mintpy.load.azAngleFile    = ../azimuthAngle/*.vrt
  mintpy.load.waterMaskFile  = ../mask/watermask.msk
  ##---------subset (optional):
  ## if both yx and lalo are specified, use lalo option
  mintpy.subset.yx           = auto    #[y0:y1,x0:x1 / no], auto for no
  mintpy.subset.lalo         = auto    #[lat0:lat1,lon0:lon1 / no], auto for no
  ##---------multilook (optional):
  ## multilook while loading data with the specified method, to reduce dataset size
  ## nearest, mean and median methods are applicable to interferogram/ionosphere/offset stack(s), except for:
  ## connected components and all geometry datasets, for which nearest is hardwired.
  mintpy.multilook.method    = auto    #[nearest, mean, median], auto for nearest - lines/rows skipping approach
  mintpy.multilook.ystep     = auto    #[int >= 1], auto for 1 - no multilooking
  mintpy.multilook.xstep     = auto    #[int >= 1], auto for 1 - no multilooking
"""

EXAMPLE = """example:
  prep_aria.py -t smallbaselineApp.cfg    # recommended
  prep_aria.py -t SanFranSenDT42.txt
  prep_aria.py -s ../stack/ -d ../DEM/SRTM_3arcsec.dem -i '../incidenceAngle/*.vrt'
  prep_aria.py -s ../stack/ -d ../DEM/SRTM_3arcsec.dem -i '../incidenceAngle/*.vrt' -a '../azimuthAngle/*.vrt' -w ../mask/watermask.msk
  prep_aria.py -s ../stack/ -d ../DEM/SRTM_3arcsec.dem -i '../incidenceAngle/*.vrt' --set '../stack/setStack.vrt' --tropo '../stack/troposphereTotal/HRRR_stack.vrt' --iono '../stack/ionStack.vrt'

  # download / extract / prepare inteferograms stack from ARIA using ARIA-tools:
  ariaDownload.py --track 71 --bbox "34.21 34.31 -118.60 -118.43" --start 20240101 --end 20240301
  ariaTSsetup.py -f "products/*.nc" --bbox "34.21 34.31 -118.60 -118.43" --mask Download --num_threads 4 --layers "solidEarthTide, troposphereTotal, ionosphere" --tropo_models HRRR
  prep_aria.py -s stack -d DEM/glo_90.dem -i incidenceAngle/*.vrt -a azimuthAngle/*.vrt -w mask/esa_world_cover_2021.msk --set stack/setStack.vrt --tropo stack/troposphereTotal/HRRRStack.vrt --iono stack/ionoStack.vrt
"""

def create_parser(subparsers=None):
    """Command line parser."""
    synopsis = 'Prepare ARIA processed products for MintPy.'
    epilog = TEMPLATE + '\n' + EXAMPLE
    name = __name__.split('.')[-1]
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=epilog, subparsers=subparsers)

    parser.add_argument('-t','--template', dest='template_file', type=str,
                        help='template file with the options')
    parser.add_argument('-o', '--output', type=str, nargs=2, dest='outfile',
                        default=['./inputs/ifgramStack.h5',
                                 './inputs/geometryGeo.h5'],
                        help='output HDF5 file')
    parser.add_argument('--update', dest='updateMode', action='store_true',
                        help='Enable the update mode: checking dataset already loaded.')
    parser.add_argument('--compression', choices={'gzip', 'lzf', None}, default=None,
                        help='HDF5 file compression, default: %(default)s')

    # ifgramStack
    stack = parser.add_argument_group('interferogram stack')
    stack.add_argument('-s','--stack-dir', dest='stackDir', type=str,
                       help='The directory which contains stack VRT files.')
    stack.add_argument('-u','--unwrap-stack-name', dest='unwFile', type=str,
                       default="unwrapStack.vrt",
                       help='Name of the stack VRT file of unwrapped data.\n'+
                            'default: %(default)s')
    stack.add_argument('-c','--coherence-stack-name', dest='corFile', type=str,
                       default="cohStack.vrt",
                       help='Name of the stack VRT file of coherence data.\n'+
                            'default: %(default)s')
    stack.add_argument('-l','--conn-comp-stack-name', dest='connCompFile', type=str,
                       default="connCompStack.vrt",
                       help='Name of the stack VRT file of connected component data.\n' +
                            'default: %(default)s')
    stack.add_argument('--amp-stack-name','--amplitude-stack-name', dest='magFile', type=str,
                       default="ampStack.vrt",
                       help='Name of the stack VRT file of interferogram amplitude data (optional).\n' +
                            'default: %(default)s')

    # geometryGeo
    geom = parser.add_argument_group('geometry')
    geom.add_argument('-d','--dem', dest='demFile', type=str,
                      help='Name of the DEM file')
    geom.add_argument('-i','--incidence-angle', dest='incAngleFile', type=str,
                      help='Name of the incidence angle file')
    geom.add_argument('-a','--az-angle','--azimuth-angle', dest='azAngleFile', type=str,
                      help='Name of the azimuth angle file.')
    geom.add_argument('-w','--water-mask', dest='waterMaskFile', type=str,
                      help='Name of the water mask file')

    # correction layers: troposphereTotal, ionosphere, solidEarthTides
    corr = parser.add_argument_group('corrections')
    corr.add_argument('-ct', '--tropo', dest='tropoFile', type=str,
                      help='Name of the Troposhere Delay stack file', default=None)
    corr.add_argument('-ci', '--iono', dest='ionoFile', type=str,
                    help='Name of the Ionosphere Delay stack file', default=None)
    corr.add_argument('-cs', '--set', dest='setFile', type=str,
                    help='Name of the Solid Earth Tides stack file', default=None)

    return parser


def cmd_line_parse(iargs = None):
    # parse
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # default: multilook options
    iDict = vars(inps)
    iDict['xstep'] = int(iDict.get('xstep', 1))
    iDict['ystep'] = int(iDict.get('ystep', 1))
    iDict['method'] = str(iDict.get('method', 'nearest'))

    # check: --template option
    if inps.template_file:
        inps = read_template2inps(inps.template_file, inps)
    print('multilook x/ystep: {}/{}'.format(iDict['xstep'], iDict['ystep']))
    print('multilook method : {}'.format(iDict['method']))

    # check: --stack-dir
    if inps.stackDir is not None:
        inps.stackDir     = os.path.abspath(inps.stackDir)
        inps.corFile      = os.path.join(inps.stackDir, os.path.basename(inps.corFile))
        inps.unwFile      = os.path.join(inps.stackDir, os.path.basename(inps.unwFile))
        inps.connCompFile = os.path.join(inps.stackDir, os.path.basename(inps.connCompFile))

    # check: all options end with "File"
    # translate wildcard path input with search result
    # if not exist, raise error for required datasets
    #               set to None for the other datasets
    print('search input data file info:')
    ds_keys = [key for key in list(iDict.keys()) if key.endswith('File')]
    required_ds_keys = ['unwFile', 'corFile', 'demFile', 'incAngleFile']
    max_digit = max(len(i) for i in ds_keys)

    for key in ds_keys:
        # search for wildcard pattern
        fnames = glob.glob(iDict[key]) if iDict[key] else []

        # return the first element if more than one exist
        # except for tropo, for which multiple inputs could be passed
        if len(fnames) > 0:
            if 'tropo' not in key:
                iDict[key] = fnames[0]
            else:
                iDict[key] = fnames
            print('{k:<{w}} : {f}'.format(k=key, w=max_digit, f=iDict[key]))

        elif key in required_ds_keys:
            # raise exception if any required DS is missing
            raise SystemExit(f'ERROR: no file found for {key} in input path: "{iDict[key]}"!')

        else:
            iDict[key] = None

    return inps


def read_template2inps(template_file, inps):
    """Read input template file into inps"""
    print(f'read options from template file: {os.path.basename(template_file)}')

    from mintpy.utils import readfile, utils1 as ut

    iDict = vars(inps)
    template = readfile.read_template(template_file)
    template = ut.check_template_auto_value(template)

    # ignore template options with default auto values
    # so that options from input arguments have higher priority
    # than template options with auto values
    for key in list(template.keys()):
        if template[key] == 'auto':
            template.pop(key)

    # pass options from template to inps
    # group - load
    key_prefix = 'mintpy.load.'
    keys = [i for i in list(iDict.keys()) if key_prefix+i in template.keys()]
    for key in keys:
        value = template[key_prefix+key]
        if key in ['updateMode', 'compression']:
            iDict[key] = value
        elif key in ['unwFile']:
            iDict['stackDir'] = os.path.dirname(value)
        elif value:
            iDict[key] = str(value)

    # group - multilook
    prefix = 'mintpy.multilook.'
    key_list = [i.split(prefix)[1] for i in template.keys() if i.startswith(prefix)]
    for key in key_list:
        value = template[prefix+key]
        if key in ['xstep', 'ystep']:
            iDict[key] = int(template[prefix+key])
        elif key in ['method']:
            iDict[key] = template[prefix+key]

    return inps


####################################################################################
def main(iargs=None):
    # parse
    inps = cmd_line_parse(iargs)

    # import
    from mintpy.prep_aria import load_aria

    # run
    load_aria(inps)


####################################################################################
if __name__=="__main__":
    main(sys.argv[1:])
