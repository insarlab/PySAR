#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Antonio Valentino, Zhang Yunjun, Aug 2022        #
############################################################


import os
import sys

from mintpy.utils.arg_utils import create_argument_parser

#####################################################################################
EXAMPLE = """example:
  # same file types
  diff.py  velocity.h5    velocity_demErr.h5
  diff.py  timeseries.h5  inputs/ERA5.h5  -o timeseries_ERA5.h5
  diff.py  timeseries.h5  inputs/ERA5.h5  -o timeseries_ERA5.h5  --force
  diff.py  timeseries_ERA5_ramp_demErr.h5  ../GIANT/Stack/LS-PARAMS.h5 -o mintpy_giant.h5
  diff.py  reconUnwrapIfgram.h5  ./inputs/ifgramStack.h5  -o diffUnwrapIfgram.h5

  # different resolutions
  diff.py timeseries.h5 inputs/SET.h5 -o timeseries_SET.h5
  diff.py timeseries_SET.h5 ion.h5 -o timeseries_SET_ion.h5

  # different file types
  diff.py  filt_20220905_20230220.unw  ./inputs/ERA5.h5 -o filt_20220905_20230220_ERA5.unw
  diff.py  timeseries.h5 ./inputs/ITRF14.h5 -o timeseries_ITRF14.h5

  # multiple files
  diff.py  waterMask.h5  maskSantiago.h5  maskFernandina.h5  -o maskIsabela.h5
"""


def create_parser(subparsers=None):
    synopsis = 'Generate the difference of two input files.'
    epilog = EXAMPLE
    name = __name__.split('.')[-1]
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=epilog, subparsers=subparsers)

    parser.add_argument('file1', help='file to be subtracted.')
    parser.add_argument('file2', nargs='+', help='file used to subtract')
    parser.add_argument('-o', '--output', dest='out_file',
                        help='output file name, default is file1_diff_file2.h5')
    parser.add_argument('--force','--force-diff', dest='force_diff', action='store_true',
                        help='Enforce the differencing for the shared dates only for time-series files')
    return parser


def cmd_line_parse(iargs=None):
    # parse
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # import
    from mintpy.utils import readfile

    # check
    # check: number of files == 2 for time-series and ifgram stack
    ftype = readfile.read_attribute(inps.file1)['FILE_TYPE']
    if ftype in ['timeseries', 'ifgramStack', '.unw']:
        if len(inps.file2) > 1:
            raise SystemExit(f'ERROR: ONLY ONE file2 is inputted for {ftype} type!')

    # check: --output (output file is required for number of files >=2)
    if not inps.out_file:
        if len(inps.file2) > 1:
            raise ValueError('--output is required for >=2 files!')

    # default: --output
    if not inps.out_file:
        fbase1, fext = os.path.splitext(inps.file1)
        fbase2 = os.path.splitext(os.path.basename(inps.file2[0]))[0]
        inps.out_file = f'{fbase1}_diff_{fbase2}{fext}'

    return inps


#####################################################################################
def main(iargs=None):
    # parse
    inps = cmd_line_parse(iargs)

    # import
    from mintpy.diff import diff_file

    # run
    diff_file(
        file1=inps.file1,
        file2=inps.file2,
        out_file=inps.out_file,
        force_diff=inps.force_diff,
    )


#####################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
