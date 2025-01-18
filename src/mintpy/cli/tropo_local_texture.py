#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Yang Qingyue, Hu Changyang, 2024                 #
############################################################

import argparse
import sys

from mintpy.utils.arg_utils import create_argument_parser

############################################################################
REFERENCE = """reference:
    Yang, Q., Yunjun, Z., Wang, R. Y. Heterogeneous InSAR Tropospheric Correction
    Based on Local Texture Correlation, IEEE Transactions on Geoscience and Remote Sensing, 62,
      doi:10.1109/TGRS.2024.3356749.
"""
#TODO
EXAMPLE = """example:
  tropo_local_texture.py  timeseries_ramp_demErr.h5 -g inputs/geometryRadar.h5  -m maskTempCoh.h5
"""

def create_parser(subparsers=None):
    #TODO
    synopsis = 'Correct Topo-correlated Stratified tropospheric delay'
    epilog = REFERENCE + '\n' + EXAMPLE
    name = __name__.split('.')[-1]
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=epilog, subparsers=subparsers)

    parser.add_argument('timeseries_file', help='time-series file to be corrected')
    parser.add_argument('-g', '--geometry', dest='geom_file', required=True,
                        help='DEM file used for correlation calculation.')
    parser.add_argument('-m', '--mask', dest='mask_file', required=True,
                        help='mask file for pixels used for correlation calculation')

    parser.add_argument('-w', '--windowsize', type=int, default=141,
                        help='window size (square window, must be odd number).')
    parser.add_argument('-r', '--overlapratio', type=float, default=0.4,
                        help='overlap ratio for window filtering')

    parser.add_argument('-o', '--outfile', help='output corrected timeseries file name')
    return parser

def cmd_line_parse(iargs=None):
    # parse
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # check: -r / --overlapration option (must be within [0,1])
    if inps.overlapratio and (not 0.0 <= inps.overlapratio <= 1.0):
        msg = f'overlap ratio {inps.overlapratio} is NOT within [0.0, 1.0]'
        raise argparse.ArgumentTypeError(msg)

    # check: -w / --windowsize option (must be odd number)
    if inps.windowsize and (inps.windowsize % 2 == 0):
        msg = f'window size {inps.windowsize} is NOT odd number'
        raise argparse.ArgumentTypeError(msg)

    return inps


############################################################################
def main(iargs=None):
    # parse
    inps = cmd_line_parse(iargs)

    # import
    from mintpy.tropo_local_texture import run_tropo_local_texture

    # run
    run_tropo_local_texture(inps)


############################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
