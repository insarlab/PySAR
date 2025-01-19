#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Antonio Valentino, Zhang Yunjun, Aug 2022        #
############################################################


import sys

from mintpy.utils.arg_utils import create_argument_parser

###########################################################################
TEMPLATE = """template
## if both yx and lalo are specified, use lalo option unless a) no lookup file AND b) dataset is in radar coord
mintpy.subset.yx       = auto    #[1800:2000,700:800 / no], auto for no
mintpy.subset.lalo     = auto    #[31.5:32.5,130.5:131.0 / no], auto for no
"""

EXAMPLE = """example:
  subset.py inputs/ifgramStack.h5 -y 400  1500 -x 200   600
  subset.py geo_velocity.h5       -l 30.5 30.8 -L 130.3 130.9
  subset.py velocity.h5   --lat 25.894 25.896  --lon -80.124 -80.122
  subset.py 030405_090801.unw     -t SinabungT495F50AlosA.template
  subset.py demLat*.dem.wgs84 --lat 32.5 33.0 --lon 130.2 130.6 -o srtm1.h5

  # subset to the same coverage as the reference file
  subset.py geo_incidence.h5 -r subset_geo_velocity.h5

  # multiple files input
  subset.py *velocity*.h5 timeseries*.h5  -y 400 1500  -x 200 600
  subset.py geometryRadar.h5 slcStack.h5  --lat 25.894 25.896  --lon -80.124 -80.122

  # crop to larger area with custom fill value
  subset.py geo_velocity.h5 -l 32.2 33.5  --outfill-nan
  subset.py Mask.h5 -x 500 3500 --outfill 0

  # "tight" subset for geocoded lookup table larger than data file
  subset.py geomap_4rlks.trans --tight
"""

def create_parser(subparsers=None):
    synopsis = 'Generate a subset from file/dataset'
    epilog = TEMPLATE + '\n' + EXAMPLE
    name = __name__.split('.')[-1]
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=epilog, subparsers=subparsers)

    parser.add_argument('file', nargs='+', help='File(s) to subset/crop')

    # subset range
    parser.add_argument('-x','--sub-x','--subset-x', dest='subset_x', type=int, nargs=2,
                        help='subset range in x/cross-track/column direction')
    parser.add_argument('-y','--sub-y','--subset-y', dest='subset_y', type=int, nargs=2,
                        help='subset range in y/along-track/row direction')
    parser.add_argument('-l', '--lat', '--sub-lat', '--subset-lat', dest='subset_lat',
                        type=float, nargs=2, help='subset range in latitude')
    parser.add_argument('-L', '--lon', '--sub-lon', '--subset-lon', dest='subset_lon',
                        type=float, nargs=2, help='subset range in column\n\n')

    parser.add_argument('-t', '--template', dest='template_file',
                        help='template file with subset setting.  i.e. \n'
                             'mintpy.subset.yx    = 300:800,1000:3500\n'
                             'mintpy.subset.lalo  = 30.2:30.5,130.1:131.3')
    parser.add_argument('-r', '--reference',
                        help='reference file, subset to the same lalo as reference file')
    parser.add_argument('--tight', action='store_true',
                        help='subset geomap_*.trans file based on non-zero values.\n' +
                             'For geocoded file(s) only'
                             'A convenient way to get rid of extra wide space due to "too large" DEM.\n\n')

    # extra options
    parser.add_argument('--outfill', dest='fill_value', type=float,
                        help="fill subset area out of data coverage with input value. i.e. \n"
                             "np.nan, 0, 1000, ... \n"
                             "By default, it's None for no-outfill.")

    parser.add_argument('-o', '--output', dest='outfile',
                        help='output file name\n' +
                             'add prefix "sub_" if input/output files are in the same directory;\n' +
                             'same filename otherwise.')

    dset_group = parser.add_argument_group('Datasets',
                                           'Create a subset of entire dataset in radar using y/x or lat/lon option\n' +
                                           'Including *.trans and *.dem in geo coord.')
    dset_group.add_argument('--lookup', dest='lookup_file',
                            help='calculate bounding box in geo/radar coord from input radar/geo subset range\n' +
                                 'using transformation file, i.e. geomap_4rlks.trans\n' +
                                 'All input radar coord file should be same size/coverage; same for all geo coord files.')
    return parser


def cmd_line_parse(iargs=None):
    # parse
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # import
    from mintpy.utils import utils1 as ut

    # check: existence of input file
    flist = ut.get_file_list(inps.file)
    if len(flist) == 0:
        raise FileNotFoundError(f'NO file found in: {inps.file}!')
    inps.file = flist

    # default: disable --output option for multiple input files
    if len(inps.file) > 1 and inps.outfile:
        inps.outfile = None
        print('WARNING: disable --output option for multiple input files.')

    return inps


###########################################################################
def main(iargs=None):
    # parse
    inps = cmd_line_parse(iargs)

    # import
    from mintpy.subset import read_aux_subset2inps, subset_file

    # run
    inps = read_aux_subset2inps(inps)

    for fname in inps.file:
        print('-'*30)
        subset_file(fname, vars(inps), out_file=inps.outfile)


###########################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
