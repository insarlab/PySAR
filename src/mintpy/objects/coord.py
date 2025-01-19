"""Class for (radar/geo) coordinates conversion."""
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, 2019                               #
############################################################
# Recommend import:
#   from mintpy.utils import utils as ut


from argparse import Namespace

import numpy as np

from mintpy.constants import EARTH_RADIUS
from mintpy.utils import readfile, utils0 as ut0, utils1 as ut1


#####################################  coordinate class begin ##############################################
class coordinate:
    """
    Coordinates conversion based lookup file in pixel-level accuracy

    Convention:
        radar coordinates or row and column numbers represent the whole pixel, starting from 0 as a python convention
        geo coordinates represent an infinite precise point without size:
            (list) of points indicate the lat/lon of the pixel center.
            (bounding) box of points indicate the lat/lon of the pixel UL corner.

    Example:
        from mintpy.utils import readfile, utils as ut
        atr = readfile.read('velocity.h5')
        coord = ut.coordinate(atr)                                          # geo coord
        coord = ut.coordinate(atr, lookup_file='inputs/geometryRadar.h5')   # radar coord
        coord = ut.coordinate(atr, lookup_file=['lat.rdr', 'lon.rdr'])      # radar coord in isce2 format
        y, x = coord.geo2radar(lat, lon)[0:2]
        lat, lon = coord.radar2geo(y, x)[0:2]
    """

    def __init__(self, metadata, lookup_file=None):
        """Define a coordinate object
        Parameters: metadata    - dict, source metadata
                    lookup_file - str / list of 2 str, lookup table file(s)
                                  'geometryRadar.h5'
                                  ['lat.rdr', 'lon.rdr']
        Returns:    mintpy.utils.utils.coordinate object
        """
        self.src_metadata = metadata
        if lookup_file is None:
            lookup_file = ut1.get_lookup_file(lookup_file, abspath=True, print_msg=False)
        if isinstance(lookup_file, str):
            lookup_file = [lookup_file, lookup_file]
        self.lookup_file = lookup_file
        self.lut_y = None
        self.lut_x = None


    def open(self):
        self.earth_radius = float(self.src_metadata.get('EARTH_RADIUS', EARTH_RADIUS))

        if 'Y_FIRST' in self.src_metadata.keys():
            self.geocoded = True
            self.lat0 = float(self.src_metadata['Y_FIRST'])
            self.lon0 = float(self.src_metadata['X_FIRST'])
            self.lat_step = float(self.src_metadata['Y_STEP'])
            self.lon_step = float(self.src_metadata['X_STEP'])
        else:
            self.geocoded = False
            if self.lookup_file:
                self.lut_metadata = readfile.read_attribute(self.lookup_file[0])

        # remove empty attributes [to facilitate checking laterward]
        key_list = ['UTM_ZONE']
        for key in key_list:
            if key in self.src_metadata and not self.src_metadata[key]:
                self.src_metadata.pop(key)


    def _clean_coord(self, coord1, coord2):
        """Ensure input coordinate as list type and same size."""

        def _to_list(x):
            # convert numpy array to list or scalar
            if isinstance(x, np.ndarray):
                x = x.tolist()
            # convert scalar / None to list
            # Note: np.float128 is not supported on Windows OS,
            #   use np.longdouble as a platform neutral syntax
            float_types = (float, np.float16, np.float32, np.float64, np.longdouble)
            int_types = (int, np.int16, np.int32, np.int64)
            if isinstance(x, int_types + float_types):
                x = [x]
            elif x is None:
                x = [None]
            x = list(x)
            return x

        # convert to list type
        coord1 = _to_list(coord1)
        coord2 = _to_list(coord2)

        # ensure the same size
        if len(coord1) != len(coord2):
            if len(coord1) == 1 and len(coord2) > 1:
                coord1 *= len(coord2)
            elif len(coord1) > 1 and len(coord2) == 1:
                coord2 *= len(coord1)
            else:
                raise ValueError('Input two coordinates do NOT have the same size!')

        return coord1, coord2


    def lalo2yx(self, lat_in, lon_in):
        """convert geo coordinates into radar coordinates for Geocoded file only.

        Parameters: lat_in    - list / tuple / 1D np.ndarray / float, coordinate(s) in latitude / northing
                    lon_in    - list / tuple / 1D np.ndarray / float, coordinate(s) in longitude / easting
        Returns:    coord_out - tuple(list / float / 1D np.ndarray), coordinates(s) in (row, col) numbers
        Example:    300, 1000 = coordinate.lalo2yx(32.1, 130.5)
                    300 = coordinate.lalo2yx(32.1, None)[0]
                    ([300, 301], [1000, 1001]) = coordinate.lalo2yx((32.1, 32.101), (130.5, 130.501))
        """
        self.open()
        if not self.geocoded:
            raise ValueError('Input file is NOT geocoded.')

        lat_in, lon_in = self._clean_coord(lat_in, lon_in)

        # attempts to convert lat/lon to utm coordinates if needed.
        if (lat_in is not None and lon_in is not None
                and 'UTM_ZONE' in self.src_metadata
                and np.max(np.abs(lat_in)) <= 90
                and np.max(np.abs(lon_in)) <= 360):
            lat_in, lon_in = ut0.latlon2utm(self.src_metadata, np.array(lat_in), np.array(lon_in))

        # convert coordinates
        y_out = []
        x_out = []
        for lat_i, lon_i in zip(lat_in, lon_in):
            # plus 0.01 to be more robust in practice
            y_i = None if lat_i is None else int(np.floor((lat_i - self.lat0) / self.lat_step + 0.01))
            x_i = None if lon_i is None else int(np.floor((lon_i - self.lon0) / self.lon_step + 0.01))
            y_out.append(y_i)
            x_out.append(x_i)

        # output format
        if len(y_out) == 1 and len(x_out) == 1:
            coord_out = tuple([y_out[0], x_out[0]])
        else:
            coord_out = tuple([y_out, x_out])

        return coord_out


    def yx2lalo(self, y_in, x_in):
        """convert radar coordinates into geo coordinates (pixel center)
            for Geocoded file only.

        Parameters: y_in      - list / tuple / 1D np.ndarray / int, coordinate(s) in row number
                    x_in      - list / tuple / 1D np.ndarray / int, coordinate(s) in col number
        Returns:    coord_out - tuple(list / 1D np.ndarray / int), coordinate(s) in (lat/northing, lon/easting)
        Example:    32.1, 130.5 = coordinate.yx2lalo(300, 1000)
                    32.1 = coordinate.yx2lalo(300, None)[0]
                    ([32.1, 32.101], [130.5, 130.501]) = coordinate.lalo2yx([(300, 301), (1000, 1001)])
        """
        self.open()
        if not self.geocoded:
            raise ValueError('Input file is NOT geocoded.')

        y_in, x_in = self._clean_coord(y_in, x_in)

        # convert coordinates
        lat_out = []
        lon_out = []
        for y_i, x_i in zip(y_in, x_in):
            lat_i = None if y_i is None else (y_i + 0.5) * self.lat_step + self.lat0
            lon_i = None if x_i is None else (x_i + 0.5) * self.lon_step + self.lon0
            lat_out.append(lat_i)
            lon_out.append(lon_i)

        # output format
        if len(lat_out) == 1 and len(lon_out) == 1:
            coord_out = tuple([lat_out[0], lon_out[0]])
        else:
            coord_out = tuple([lat_out, lon_out])

        return coord_out


    def _get_lookup_row_col(self, y, x, y_factor=10, x_factor=10, geo_coord=False, debug_mode=False):
        """Get row/col number in y/x value matrix from input y/x
        Use overlap mean value between y and x buffer;
        To support point outside of value pool/matrix, could use np.polyfit to fit a line
        for y and x value buffer and return the intersection point row/col
        """
        ymin = y - y_factor;  ymax = y + y_factor
        xmin = x - x_factor;  xmax = x + x_factor
        if not geo_coord:
            ymin = max(ymin, 0.5)
            xmin = max(xmin, 0.5)
        mask_y = np.multiply(self.lut_y >= ymin, self.lut_y <= ymax)
        mask_x = np.multiply(self.lut_x >= xmin, self.lut_x <= xmax)
        mask_yx = np.multiply(mask_y, mask_x)

        # for debugging only
        if debug_mode:
            print('Debug mode is ON.\nShow the row/col number searching result.')
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=[12, 5])
            kwargs = dict(cmap='gray', interpolation='nearest')
            axs[0].imshow(mask_y, **kwargs);  axs[0].set_title('Buffer in Y direction')
            axs[1].imshow(mask_x, **kwargs);  axs[1].set_title('Buffer in X direction')
            axs[2].imshow(mask_yx, **kwargs); axs[2].set_title('Y & X overlap (zoom in)')

            try:
                idx = np.where(np.sum(mask_yx, axis=0))[0]
                idy = np.where(np.sum(mask_yx, axis=1))[0]
                axs[2].set_xlim(idx[0], idx[-1])
                axs[2].set_ylim(idy[0], idy[-1])
            except:
                pass
            axs[1].set_yticklabels([])
            fig.tight_layout()
            plt.show()

        row, col = np.nanmean(np.where(mask_yx), axis=1)
        if any(np.isnan(i) for i in [row, col]):
            raise RuntimeError(f'No corresponding coordinate found for y/x: {y}/{x}')

        return row, col


    def read_lookup_table(self, print_msg=True):
        ds_name_x = 'rangeCoord' if 'Y_FIRST' in self.lut_metadata.keys() else 'longitude'
        ds_name_y = 'azimuthCoord' if 'Y_FIRST' in self.lut_metadata.keys() else 'latitude'
        self.lut_y = readfile.read(self.lookup_file[0], datasetName=ds_name_y, print_msg=print_msg)[0]
        self.lut_x = readfile.read(self.lookup_file[1], datasetName=ds_name_x, print_msg=print_msg)[0]
        return self.lut_y, self.lut_x


    def _read_geo_lut_metadata(self):
        """Read lat/lon0, lat/lon_step_deg, lat/lon_step into a Namespace - lut"""
        lut = Namespace()
        lut.lat0 = float(self.lut_metadata['Y_FIRST'])
        lut.lon0 = float(self.lut_metadata['X_FIRST'])
        lut.lat_step_deg = float(self.lut_metadata['Y_STEP'])
        lut.lon_step_deg = float(self.lut_metadata['X_STEP'])

        # Get lat/lon resolution/step in meter
        length = int(self.lut_metadata['LENGTH'])
        lat_c = lut.lat0 + lut.lat_step_deg * length / 2.
        lut.lat_step = lut.lat_step_deg * np.pi/180.0 * self.earth_radius
        lut.lon_step = lut.lon_step_deg * np.pi/180.0 * self.earth_radius * np.cos(lat_c * np.pi/180)
        return lut


    def geo2radar(self, lat, lon, print_msg=True, debug_mode=False):
        """Convert geo coordinates into radar coordinates.

        Parameters: lat/lon   - np.array / float, latitude/longitude
        Returns:    az/rg     - np.array / int, range/azimuth pixel number
                    az/rg_res - float, residul/uncertainty of coordinate conversion
        """
        # check 1: ensure longitude convention to be consistent with src_metadata [for WGS84 coordinates]
        if 'X_FIRST' in self.src_metadata.keys() and 'UTM_ZONE' not in self.src_metadata.keys():
            width = int(self.src_metadata['WIDTH'])
            lon_step = float(self.src_metadata['X_STEP'])
            min_lon = float(self.src_metadata['X_FIRST'])
            max_lon = min_lon + lon_step * width

            # skip if larger than (-180, 180)
            # e.g. IONEX file in (-182.25, 182.25)
            if np.all(np.abs([min_lon, max_lon]) > 180):
                pass

            # ensure longitude within [0, 360)
            elif max_lon > 180:
                if np.isscalar(lon):
                    lon = lon + 360 if lon < 0. else lon
                else:
                    lon[lon < 0.] += 360

            # ensure longitude within (-180, 180]
            else:
                if np.isscalar(lon):
                    lon = lon - 360. if lon > 180. else lon
                else:
                    lon[lon > 180.] -= 360

        # check 2: attempts to convert lat/lon to utm coordinates if needed
        if ('UTM_ZONE' in self.src_metadata
                and np.max(np.abs(lat)) <= 90
                and np.max(np.abs(lon)) <= 360):
            lat, lon = ut0.latlon2utm(self.src_metadata, np.array(lat), np.array(lon))

        self.open()
        if self.geocoded:
            az, rg = self.lalo2yx(lat, lon)
            return az, rg, 0, 0

        if not isinstance(lat, np.ndarray):
            lat = np.array(lat)
            lon = np.array(lon)

        # read lookup table
        if self.lookup_file is None:
            raise FileNotFoundError('No lookup table file found!')
        if self.lut_y is None or self.lut_x is None:
            self.read_lookup_table(print_msg=print_msg)

        # For lookup table in geo-coord, read value directly (GAMMA and ROI_PAC)
        if 'Y_FIRST' in self.lut_metadata.keys():
            lut = self._read_geo_lut_metadata()

            # if source data file is subsetted before
            az0 = 0
            rg0 = 0
            if 'SUBSET_YMIN' in self.src_metadata.keys():
                az0 = int(self.src_metadata['SUBSET_YMIN'])
            if 'SUBSET_XMIN' in self.src_metadata.keys():
                rg0 = int(self.src_metadata['SUBSET_XMIN'])

            # uncertainty due to different resolution between src and lut file
            try:
                az_step = ut0.azimuth_ground_resolution(self.src_metadata)
                rg_step = ut0.range_ground_resolution(self.src_metadata, print_msg=False)
                x_factor = np.ceil(abs(lut.lon_step) / rg_step).astype(int)
                y_factor = np.ceil(abs(lut.lat_step) / az_step).astype(int)
            except:
                x_factor = 2
                y_factor = 2

            # read y/x value from lookup table
            row = np.rint((lat - lut.lat0) / lut.lat_step_deg).astype(int)
            col = np.rint((lon - lut.lon0) / lut.lon_step_deg).astype(int)
            rg = np.rint(self.lut_x[row, col]).astype(int) - rg0
            az = np.rint(self.lut_y[row, col]).astype(int) - az0

        # For lookup table in radar-coord, search the buffer and use center pixel (ISCE)
        else:
            # get resolution in degree in range/azimuth direction
            az_step = ut0.azimuth_ground_resolution(self.src_metadata)
            rg_step = ut0.range_ground_resolution(self.src_metadata, print_msg=False)
            lat_c = (np.nanmax(lat) + np.nanmin(lat)) / 2.
            az_step_deg = 180./np.pi * az_step / (self.earth_radius)
            rg_step_deg = 180./np.pi * rg_step / (self.earth_radius * np.cos(lat_c * np.pi/180.))

            az, rg = np.zeros(lat.shape), np.zeros(lat.shape)
            x_factor = 10
            y_factor = 10

            # search the overlap area of buffer in x/y direction and use the cross center
            if lat.size == 1:
                az, rg = self._get_lookup_row_col(
                    lat, lon,
                    y_factor*az_step_deg,
                    x_factor*rg_step_deg,
                    geo_coord=True,
                    debug_mode=debug_mode,
                )
            else:
                for i in range(rg.size):
                    az[i], rg[i] = self._get_lookup_row_col(
                        lat[i], lon[i],
                        y_factor*az_step_deg,
                        x_factor*rg_step_deg,
                        geo_coord=True,
                        debug_mode=debug_mode,
                    )
            az = np.floor(az).astype(int)
            rg = np.floor(rg).astype(int)

        rg_resid = x_factor
        az_resid = y_factor
        return az, rg, az_resid, rg_resid


    def radar2geo(self, az, rg, print_msg=True, debug_mode=False):
        """Convert radar coordinates into geo coordinates (pixel center)
        Parameters: rg/az      - np.array / int, range/azimuth pixel number
        Returns:    lon/lat    - np.array / float, longitude/latitude of input point (rg,az);
                                 nan if not found.
                    latlon_res - float, residul/uncertainty of coordinate conversion
        """
        self.open()
        if self.geocoded:
            lat, lon = self.yx2lalo(az, rg)
            return lat, lon, 0, 0

        if not isinstance(az, np.ndarray):
            az = np.array(az)
            rg = np.array(rg)

        # read lookup table file
        if self.lookup_file is None:
            raise FileNotFoundError('No lookup table file found!')
        if self.lut_y is None or self.lut_x is None:
            self.read_lookup_table(print_msg=print_msg)

        # For lookup table in geo-coord, search the buffer and use center pixel
        if 'Y_FIRST' in self.lut_metadata.keys():
            lut = self._read_geo_lut_metadata()

            # Get buffer ratio from range/azimuth ground resolution/step
            try:
                az_step = ut0.azimuth_ground_resolution(self.src_metadata)
                rg_step = ut0.range_ground_resolution(self.src_metadata, print_msg=False)
                x_factor = 2 * np.ceil(abs(lut.lon_step) / rg_step)
                y_factor = 2 * np.ceil(abs(lut.lat_step) / az_step)
            except:
                x_factor = 10
                y_factor = 10

            if 'SUBSET_XMIN' in self.src_metadata.keys():
                rg += int(self.src_metadata['SUBSET_XMIN'])
                az += int(self.src_metadata['SUBSET_YMIN'])

            lut_row = np.zeros(rg.shape)
            lut_col = np.zeros(rg.shape)
            if rg.size == 1:
                lut_row, lut_col = self._get_lookup_row_col(
                    az, rg,
                    y_factor,
                    x_factor,
                    debug_mode=debug_mode,
                )
            else:
                for i in range(rg.size):
                    lut_row[i], lut_col[i] = self._get_lookup_row_col(
                        az[i], rg[i],
                        y_factor,
                        x_factor,
                        debug_mode=debug_mode,
                    )
            lat = (lut_row + 0.5) * lut.lat_step_deg + lut.lat0
            lon = (lut_col + 0.5) * lut.lon_step_deg + lut.lon0
            lat_resid = abs(y_factor * lut.lat_step_deg)
            lon_resid = abs(x_factor * lut.lon_step_deg)

        # For lookup table in radar-coord, read the value directly.
        else:
            lat = self.lut_y[az, rg]
            lon = self.lut_x[az, rg]
            lat_resid = 0.
            lon_resid = 0.
        return lat, lon, lat_resid, lon_resid


    def box_pixel2geo(self, pixel_box):
        """Convert pixel_box to geo_box in UL corner
        Parameters: pixel_box - list/tuple of 4 int   in (x0, y0, x1, y1)
        Returns:    geo_box   - tuple      of 4 float in (W, N, E, S)
        """
        try:
            lat, lon = self.yx2lalo([pixel_box[1], pixel_box[3]], [pixel_box[0], pixel_box[2]])
            # shift lat from pixel center to the UL corner
            lat = [i - self.lat_step / 2.0 for i in lat]
            lon = [i - self.lon_step / 2.0 for i in lon]
            geo_box = (lon[0], lat[0], lon[1], lat[1])
        except:
            geo_box = None
        return geo_box


    def box_geo2pixel(self, geo_box):
        """Convert geo_box to pixel_box
        Parameters: geo_box   - tuple      of 4 float in (W, N, E, S)
        Returns:    pixel_box - list/tuple of 4 int   in (x0, y0, x1, y1)
        """
        try:
            y, x = self.lalo2yx([geo_box[1], geo_box[3]], [geo_box[0], geo_box[2]])
            pixel_box = (x[0], y[0], x[1], y[1])
        except:
            pixel_box = None
        return pixel_box


    def bbox_radar2geo(self, pix_box, print_msg=False):
        """Calculate bounding box in lat/lon for file in geo coord, based on input radar/pixel box
        Parameters: pix_box - tuple of 4 int, in (x0, y0, x1, y1)
        Returns:    geo_box - tuple of 4 float, in (W, N, E, S)
        """
        x = np.array([pix_box[0], pix_box[2], pix_box[0], pix_box[2]])
        y = np.array([pix_box[1], pix_box[1], pix_box[3], pix_box[3]])
        lat, lon, lat_res, lon_res = self.radar2geo(y, x, print_msg=print_msg)
        buf = 2 * np.max(np.abs([lat_res, lon_res]))
        geo_box = (np.min(lon) - buf, np.max(lat) + buf,
                   np.max(lon) + buf, np.min(lat) - buf)
        return geo_box


    def bbox_geo2radar(self, geo_box, buf=None, print_msg=False):
        """Calculate bounding box in x/y for file in radar coord, based on input geo box.
        Parameters: geo_box - tuple of 4 float, indicating the UL/LR lon/lat
                    buf     - buffer [number of pixel]
        Returns:    pix_box - tuple of 4 int, indicating the UL/LR x/y of the bounding box in radar coord
                              for the corresponding lat/lon coverage.
        """
        lat = np.array([geo_box[3], geo_box[3], geo_box[1], geo_box[1]])
        lon = np.array([geo_box[0], geo_box[2], geo_box[0], geo_box[2]])
        y, x, y_res, x_res = self.geo2radar(lat, lon, print_msg=print_msg)

        if buf is None:
            buf = 2 * np.max(np.abs([x_res, y_res]))

        pix_box = (np.min(x) - buf, np.min(y) - buf,
                   np.max(x) + buf, np.max(y) + buf)
        return pix_box


    def check_box_within_data_coverage(self, pixel_box, print_msg=True):
        """Check the subset box's conflict with data coverage
        Parameters:  pixel_box - 4-tuple of int, indicating y/x coordinates of subset
        Returns:     out_box   - 4-tuple of int
        """
        self.open()
        length, width = int(self.src_metadata['LENGTH']), int(self.src_metadata['WIDTH'])
        if pixel_box is None:
            pixel_box = (0, 0, width, length)
        sub_x = [pixel_box[0], pixel_box[2]]
        sub_y = [pixel_box[1], pixel_box[3]]

        if sub_y[0] >= length or sub_y[1] <= 0 or sub_x[0] >= width or sub_x[1] <= 0:
            data_box = (0, 0, width, length)
            msg = 'ERROR: input index is out of data range!\n'
            msg += f'\tdata   range in (x0,y0,x1,y1): {data_box}\n'
            msg += f'\tsubset range in (x0,y0,x1,y1): {pixel_box}\n'
            msg += f'\tdata   range in (W, N, E, S): {self.box_pixel2geo(data_box)}\n'
            msg += f'\tsubset range in (W, N, E, S): {self.box_pixel2geo(pixel_box)}\n'
            raise ValueError(msg)

        # Check Y/Azimuth/Latitude subset range
        if sub_y[0] < 0:
            sub_y[0] = 0
            if print_msg:
                print('WARNING: input y < min (0)! Set it to min.')
        if sub_y[1] > length:
            sub_y[1] = length
            if print_msg:
                print(f'WARNING: input y > max ({length})! Set it to max.')

        # Check X/Range/Longitude subset range
        if sub_x[0] < 0:
            sub_x[0] = 0
            if print_msg:
                print('WARNING: input x < min (0)! Set it to min.')
        if sub_x[1] > width:
            sub_x[1] = width
            if print_msg:
                print(f'WARNING: input x > max ({width})! Set it to max.')

        out_box = (sub_x[0], sub_y[0], sub_x[1], sub_y[1])
        return out_box

#####################################  coordinate class end ##############################################
