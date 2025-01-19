#!/usr/bin/env python3
#########################################################################
# Program is part of MintPy                                             #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi                      #
# Author: Author: Ashok Dahal, Alexander Handwerger, Eric Fielding.     #
# Mar 2024 Based on previous code by Antonio Valentino, Zhang Yunjun.   #
#########################################################################

import sys

from mintpy.utils.arg_utils import create_argument_parser

################################################################################
REFERENCE = """reference:
  Liu, L., Millar, C. I., Westfall, R. D., and Zebker, H. A.(2013). Surface motion of active
    rock glaciers in the Sierra Nevada, California, USA: inventory and a case study using InSAR.
    The Cryosphere, 7, 1109–1119.
    https://doi.org/10.5194/tc-7-1109-2013

  Handwerger, A.L., Huang, MH., Fielding, E.J. et al (2019). A shift from drought to extreme rainfall
    drives a stable landslide to catastrophic failure. Sci Rep 9, 1569.
    https://doi.org/10.1038/s41598-018-38300-0

"""

EXAMPLE = """example:
 # Geocode the data before running this step
 cd ./mintpy
 mask.py velocity.h5 -m maskTempCoh.h5
 geocode.py velocity_msk.h5 -l inputs/geometryRadar.h5 -x 0.00027778 -y -0.00027778 --bbox 32.0 32.5 130.1 130.5

 # To run this program do the following
 los2slope.py -file ./geo_velocity.h5 -g ./geo_geometryRadar.h5 -o ./geo_velocity_slp.h5 -m 15 --slp_thresh 1.0 --scaling_factor_thresh 10

 # Or simply
 los2slope.py -file ./geo_velocity.h5 -g ./geo_geometryRadar.h5 -o ./geo_velocity_slp.h5
"""


def create_parser(subparsers=None):
    synopsis = "Project Asc or Desc LOS displacement to downslope direction"
    epilog = REFERENCE + "\n" + EXAMPLE
    name = __name__.split(".")[-1]
    parser = create_argument_parser(
        name,
        synopsis=synopsis,
        description=synopsis,
        epilog=epilog,
        subparsers=subparsers,
    )

    # input files
    parser.add_argument(
        "-file",
        help="Ascending or descending files\n"
        "Files need to be geocoded in the same spatial resolution.",
    )
    parser.add_argument(
        "-d",
        "--dset",
        dest="ds_name",
        type=str,
        help="dataset to use, default: 1st dataset",
    )
    parser.add_argument(
        "-g",
        "--geom-file",
        dest="geom_file",
        help="Geometry files for the input data files.",
    )

    parser.add_argument(
        "-s",
        "--slp_thresh",
        dest="slp_thresh",
        type=float,
        default=5.0,
        help="Slope threshold below which the downslope projection is masked out"
        + "(default: %(default)s).",
    )

    parser.add_argument(
        "-a",
        "--asp_thresh",
        dest="asp_thresh",
        type=float,
        default=0.0,
        help="Hillslope Aspect value where downslope projection is masked out. This helps prevent erroneous values"
        + "(default: %(default)s).",
    )

    parser.add_argument(
        "-sf",
        "--scaling_factor_thresh",
        dest="scaling_factor_thresh",
        type=float,
        default=10,
        help="Threshold for divide factor G, this removes extremely large downslope values that are unrealistic"
        + "(default: %(default)s).",
    )

    parser.add_argument(
        "-m",
        "--median_kernel",
        dest="median_kernel",
        default=25,
        help="Smoothing kernel size for median filter of DEM, must be an odd number.",
    )

    parser.add_argument(
        "-si",
        "--gaussian_sigma",
        dest="gaussian_sigma",
        help="Standard deviation value for smoothing of DEM",
    )

    # output - data files
    parser.add_argument(
        "-o",
        "--output",
        dest="outfile",
        metavar=("DSLP_FILE"),
        default=["Dslp.h5"],
        help="output file name for slope direction components",
    )
    return parser


def cmd_line_parse(iargs=None):
    # parse
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # import
    from mintpy.utils import readfile

    # check
    atr = readfile.read_attribute(inps.file)

    # check: if input file type is supported
    ts_ftypes = ["timeseries", "velocity"]
    ftype = atr["FILE_TYPE"]
    if ftype not in ts_ftypes:
        raise Exception(
            f"input file type ({ftype}) contains UN-supported file types: {ts_ftypes}!"
        )

    # check: if input is in geo-coordinates
    if any("X_FIRST" not in i for i in [atr]):
        raise Exception("Not all input files are geocoded.")

    return inps


################################################################################
def main(iargs=None):
    # parse
    inps = cmd_line_parse(iargs)

    # import
    from los2slope import run_los2slope

    # run
    run_los2slope(inps)


################################################################################
if __name__ == "__main__":
    main(sys.argv[1:])
