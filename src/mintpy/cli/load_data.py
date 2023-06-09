#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Antonio Valentino, Zhang Yunjun, Aug 2022        #
############################################################

import os
import sys

from mintpy.defaults import auto_path
from mintpy.defaults.template import get_template_content
from mintpy.utils.arg_utils import create_argument_parser

#################################################################
DEFAULT_TEMPLATE = """template:
########## 1. Load Data (--load to exit after this step)
{}\n
{}\n
{}""".format(
    auto_path.AUTO_PATH_GAMMA,
    auto_path.AUTO_PATH_ISCE_STRIPMAP,
    auto_path.AUTO_PATH_ISCE_TOPS,
)

TEMPLATE = get_template_content('load_data')

NOTE = """NOTE:
  For interferogram, unwrapPhase is required, the other dataset are optional, including coherence, connectComponent, wrapPhase, etc.
  The unwrapPhase metadata file requires DATE12 attribute in YYMMDD-YYMMDD format.
  All path of data file must contain the reference and secondary date, either in file name or folder name.
"""

EXAMPLE = """example:
  # MUST run in the mintpy working directory!

  # show example template file for ISCE/ROI_PAC/GAMMA products
  load_data.py -H

  # load & write the following HDF5 files:
  # ./inputs/ifgramStack.h5   for interferogram        stack
  # ./inputs/offsetStack.h5   for range/azimuth offset stack
  # ./inputs/geometryRadar.h5 for geometry in radar coordinates
  # ./inputs/geometryGeo.h5   for geometry in geo   coordinates
  # ./inputs/ion.h5           for smooth ionosphere     time-series (from ISCE topStack)
  # ./inputs/ionBurstRamp.h5  for ionosphere burst ramp time-series (from ISCE topStack)
  load_data.py -t smallbaselineApp.cfg
  load_data.py -t smallbaselineApp.cfg GalapagosSenDT128.txt --project GalapagosSenDT128

  # load geometry ONLY
  smallbaselineApp.py SaltonSeaSenDT173.txt -g
  load_data.py -t smallbaselineApp.cfg -l geom

  # load ionoshpere time series ONLY
  smallbaselineApp.py SaltonSeaSenDT173.txt -g
  load_data.py -t smallbaselineApp.cfg -l ion
"""


def create_parser(subparsers=None):
    """Create command line parser."""
    synopsis = 'Load stacks of interferograms to HDF5 files'
    epilog = TEMPLATE + '\n' + NOTE + '\n' + EXAMPLE
    name = __name__.split('.')[-1]
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=epilog, subparsers=subparsers)

    # extra help
    parser.add_argument('-H', dest='print_example_template', action='store_true',
                        help='Print/Show the example template file for loading.')

    # input files
    parser.add_argument('-t', '--template', dest='template_file', type=str, nargs='+',
                        help='template file(s) with path info.')
    parser.add_argument('-l','--listDset', nargs='+', help='a list of datasets to be loadded (default: %(default)s)',
                        default=['ifg','geom','ion'], choices=['ifg','geom','ion'])

    # options from template file name & content
    parser.add_argument('--project', type=str, dest='PROJECT_NAME',
                        help='project name of dataset for INSARMAPS Web Viewer')
    parser.add_argument('--enforce', '-f', dest='updateMode', action='store_false',
                        help='Disable the update mode, or skip checking dataset already loaded.')
    parser.add_argument('--compression', choices={'gzip', 'lzf', None}, default=None,
                        help='compress loaded geometry while writing HDF5 file, default: None.')

    return parser


def cmd_line_parse(iargs=None):
    """Command line parser."""
    # parse
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # check: -H option
    if inps.print_example_template:
        print(DEFAULT_TEMPLATE)
        sys.exit(0)

    # check: -t/--template option
    # -t option is required AND
    # smallbaselineApp.cfg file is required
    if not inps.template_file:
        parser.print_usage()
        script_name = os.path.basename(__file__)
        print(f'{script_name}: error: -t/--template option is required.')
        print(f'run {script_name} -H to show the example template file.')
        sys.exit(1)

    elif all(not x.endswith('smallbaselineApp.cfg') for x in inps.template_file):
        script_name = os.path.basename(__file__)
        print(f'{script_name}: error: at least smallbaselineApp.cfg file is required for -t/--template option.')
        sys.exit(1)

    return inps



#################################################################
def main(iargs=None):
    # parse
    inps = cmd_line_parse(iargs)

    # import
    from mintpy.load_data import load_data

    # run
    load_data(inps)


#################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
