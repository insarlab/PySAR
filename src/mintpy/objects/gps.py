############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Jul 2018                           #
############################################################
# Utility scripts for GPS handling
# Recommend import:
#     from mintpy.objects.gps import GPS


import os
import csv
import glob
import datetime as dt
import numpy as np
from pyproj import Geod
from urllib.request import urlretrieve
import pandas as pd
import zipfile
import matplotlib.pyplot as plt

from mintpy.objects.coord import coordinate
from mintpy.utils import ptime, time_func, readfile, utils1 as ut


supported_sources = ['UNR', 'ESESES']

UNR_site_list_file_url = 'http://geodesy.unr.edu/NGLStationPages/DataHoldings.txt'

ESESES_site_list_file_url = 'http://garner.ucsd.edu/pub/measuresESESES_products/Velocities/ESESES_Velocities.txt'



def dload_site_list(out_file=None, source='UNR', print_msg=True) -> str:
    """Download single file with list of GPS site locations.
    """
    # Check source is supported
    assert source in supported_sources, \
        f'Source {source:s} not supported. Use one of {supported_sources}'

    # Determine URL
    if source == 'UNR':
        site_list_file_url = UNR_site_list_file_url
    elif source == 'ESESES':
        site_list_file_url = ESESES_site_list_file_url

    # Handle output file
    if out_file is None:
        out_file = os.path.basename(site_list_file_url)

    # Report if requested
    if print_msg:
        print(f'Downloading site list from {source}: {site_list_file_url} to {out_file}')

    # Download file
    urlretrieve(site_list_file_url, out_file)

    return out_file


def search_gps(SNWE, source='UNR', start_date=None, end_date=None,
               site_list_file=None, min_num_solution=None, print_msg=True):
    """Search available GPS sites within the geo bounding box from UNR website
    Parameters: SNWE       : tuple of 4 float, indicating (South, North, West, East) in degrees
                start_date : string in YYYYMMDD format
                end_date   : string in YYYYMMDD format
                site_list_file : string.
                min_num_solution : int, minimum number of solutions available
    Returns:    site_names : 1D np.array of string for GPS station names
                site_lats  : 1D np.array for lat
                site_lons  : 1D np.array for lon
    """
    print('Searching!', source)
    # Check start and end dates if provided
    if start_date is not None:
        start_date = dt.datetime.strptime(start_date, '%Y%m%d')
    if end_date is not None:
        end_date = dt.datetime.strptime(end_date, '%Y%m%d')
    if start_date is not None and end_date is not None:
        assert(start_date < end_date), 'Start date must be before end date'

    # Check file name
    if site_list_file is None:
        if source == 'UNR':
            print('Using UNR!')
            site_list_file = os.path.basename(UNR_site_list_file_url)
        elif source == 'ESESES':
            site_list_file = os.path.basename(ESESES_site_list_file_url)

    # Check whether site list file is in current directory
    if not os.path.isfile(site_list_file):
        # Download file
        dload_site_list(site_list_file, print_msg=print_msg)

    # Parse data from file
    if source == 'UNR':
        site_data = read_UNR_station_list(site_list_file)
    elif source == 'ESESES':
        site_data = read_ESESES_station_list(site_list_file)

    if print_msg == True:
        print('Loaded data for fields: {:s}'.\
            format(' '.join(list(site_data.columns))))

    # Parse bounding box
    lat_min, lat_max, lon_min, lon_max = SNWE
    assert (lon_min < lon_max) and (lat_min < lat_max), \
        'Check bounding box'

    if print_msg == True:
        print('Cropping to')
        print(f'lon range: {lon_min:.5f} to {lon_max:.5f}')
        print(f'lat range: {lat_min:.5f} to {lat_max:.5f}')

    # Ensure lon values in (-180, 180]
    site_data['lon'] = [lon - 360 if lon > 180 else lon \
                        for lon in site_data['lon']]

    # Limit in space
    drop_ndx = (site_data.lat < lat_min) \
                | (site_data.lat > lat_max) \
                | (site_data.lon < lon_min) \
                | (site_data.lon > lon_max)
    site_data.drop(site_data[drop_ndx].index, inplace=True)

    # Limit in time
    if start_date is not None:
        if hasattr(site_data, 'start_date'):
            drop_ndx = site_data.start_date > start_date
            site_data.drop(site_data[drop_ndx].index, inplace=True)
        else:
            print('No date information available--date range not applied to GPS site selection')

    if end_date is not None:
        if hasattr(site_data, 'end_date'):
            drop_ndx = site_data.end_date < end_date
            site_data.drop(site_data[drop_ndx].index, inplace=True)
        else:
            print('No date information available--date range not applied to GPS site selection')

    # Limit based on number of solutions
    if hasattr(site_data, 'num_solution'):
        drop_ndx = site_data.num_solution < min_num_solution
        site_data.drop(site_data[drop_ndx].index, inplace=True)

    # Final reporting
    if print_msg == True:
        print('{:d} stations available'.format(site_data.shape[0]))

    return (site_data.site.to_numpy(),
            site_data.lat.to_numpy(),
            site_data.lon.to_numpy())

def read_UNR_station_list(site_list_file, print_msg=True):
    """Return names and lon/lat values for UNR GNSS stations.
    """
    if print_msg == True:
        print('Parsing UNR site list file')

    # Read file contents
    site_data = pd.read_fwf(site_list_file,
                            widths=(4, 9, 12, 9, 14, 14, 14, 11, 11, 11, 7))

    # Rename columns for uniformity
    site_data.rename(columns={'Sta': 'site',
        'Lat(deg)': 'lat', 'Long(deg)': 'lon',
        'Dtbeg': 'start_date', 'Dtend': 'end_date',
        'NumSol': 'num_solution'}, inplace=True)

    # Format dates
    site_data['start_date'] = [dt.datetime.strptime(date, '%Y-%m-%d') \
                                for date in site_data.start_date]
    site_data['end_date'] = [dt.datetime.strptime(date, '%Y-%m-%d') \
                                for date in site_data.end_date]

    return site_data

def read_ESESES_station_list(site_list_file, print_msg=True):
    """Return names and lon/lat values for JPL GNSS stations.
    """
    if print_msg == True:
        print('Parsing ESESES site list file')

    # Read file contents
    site_data = pd.read_csv(site_list_file, header = 14, delim_whitespace=True)

    # Rename columns for uniformity
    site_data.rename(columns={'Site': 'site',
        'Latitude': 'lat', 'Longitude': 'lon'}, inplace=True)

    return site_data


def get_baseline_change(dates1, pos_x1, pos_y1, pos_z1,
                        dates2, pos_x2, pos_y2, pos_z2):
    """Calculate the baseline change between two GPS displacement time-series
    Parameters: dates1/2     : 1D np.array of datetime.datetime object
                pos_x/y/z1/2 : 1D np.ndarray of displacement in meters in float32
    Returns:    dates        : 1D np.array of datetime.datetime object for the common dates
                bases        : 1D np.ndarray of displacement in meters in float32 for the common
                               dates
    """
    dates = np.array(sorted(list(set(dates1) & set(dates2))))
    bases = np.zeros(dates.shape, dtype=float)
    for i in range(len(dates)):
        idx1 = np.where(dates1 == dates[i])[0][0]
        idx2 = np.where(dates2 == dates[i])[0][0]
        basei = ((pos_x1[idx1] - pos_x2[idx2]) ** 2
               + (pos_y1[idx1] - pos_y2[idx2]) ** 2
               + (pos_z1[idx1] - pos_z2[idx2]) ** 2) ** 0.5
        bases[i] = basei
    bases -= bases[0]
    bases = np.array(bases, dtype=float)

    return dates, bases


def get_gps_los_obs(meta, obs_type, site_names, start_date, end_date, source='UNR',
                    gps_comp='enu2los', horz_az_angle=-90., model=None,
                    print_msg=True,redo=False):
    """Get the GPS LOS observations given the query info.

    Parameters: meta       - dict, dictionary of metadata of the InSAR file
                obs_type   - str, GPS observation data type, displacement or velocity.
                site_names - list of str, GPS sites, output of search_gps()
                start_date - str, date in YYYYMMDD format
                end_date   - str, date in YYYYMMDD format
                gps_comp   - str, flag of projecting 2/3D GPS into LOS
                             e.g. enu2los, hz2los, up2los
                horz_az_angle - float, azimuth angle of the horizontal motion in degree
                             measured from the north with anti-clockwise as positive
                model         - dict, time function model, e.g. {'polynomial': 1, 'periodic': [1.0, 0.5]}
                print_msg  - bool, print verbose info
                redo       - bool, ignore existing CSV file and re-calculate
    Returns:    site_obs   - 1D np.ndarray(), GPS LOS velocity or displacement in m or m/yr
    Examples:   from mintpy.objects import gps
                from mintpy.utils import readfile, utils as ut
                meta = readfile.read_attribute('geo/geo_velocity.h5')
                SNWE = ut.four_corners(meta)
                site_names = gps.search_gps(SNWE, start_date='20150101', end_date='20190619')
                vel = gps.get_gps_los_obs(meta, 'velocity',     site_names, start_date='20150101', end_date='20190619')
                dis = gps.get_gps_los_obs(meta, 'displacement', site_names, start_date='20150101', end_date='20190619')
    """
    vprint = print if print_msg else lambda *args, **kwargs: None
    num_site = len(site_names)

    # obs_type --> obs_ind
    obs_types = ['displacement', 'velocity']
    if obs_type not in obs_types:
        raise ValueError(f'un-supported obs_type: {obs_type}')
    obs_ind = 3 if obs_type.lower() == 'displacement' else 4

    # GPS CSV file info
    file_dir = os.path.dirname(meta['FILE_PATH'])
    csv_file = os.path.join(file_dir, f'gps_{gps_comp}')
    csv_file += f'{horz_az_angle:.0f}' if gps_comp == 'horz' else ''
    csv_file += '.csv'
    col_names = ['Site', 'Lon', 'Lat', 'Displacement', 'Velocity']
    col_types = ['U10'] + ['f8'] * (len(col_names) - 1)
    vprint(f'default GPS observation file name: {csv_file}')

    # skip re-calculate GPS if:
    # 1. redo is False AND
    # 2. csv_file exists (equivalent to num_row > 0) AND
    # 3. num_row >= num_site
    num_row = 0
    if os.path.isfile(csv_file):
        fc = np.genfromtxt(csv_file, dtype=col_types, delimiter=',', names=True)
        num_row = fc.size

    if not redo and os.path.isfile(csv_file) and num_row >= num_site:
        # read from existing CSV file
        vprint('read GPS observations from file: {}'.format(csv_file))
        fc = np.genfromtxt(csv_file, dtype=col_types, delimiter=',', names=True)
        site_obs = fc[col_names[obs_ind]]

        # get obs for the input site names only
        # in case the site_names are not consistent with the CSV file.
        if num_row != num_site:
            temp_names = fc[col_names[0]]
            temp_obs = np.array(site_obs, dtype=float)
            site_obs = np.zeros(num_site, dtype=float) * np.nan
            for i, site_name in enumerate(site_names):
                if site_name in temp_names:
                    site_obs[i] = temp_obs[temp_names == site_name][0]

    else:
        # calculate and save to CSV file
        data_list = []
        vprint('calculating GPS observation ...')

        # get geom_obj (meta / geom_file)
        geom_file = ut.get_geometry_file(['incidenceAngle','azimuthAngle'],
                                         work_dir=file_dir, coord='geo')
        if geom_file:
            geom_obj = geom_file
            vprint('use incidence / azimuth angle from file: {}'.\
                   format(os.path.basename(geom_file)))
        else:
            geom_obj = meta
            vprint('use incidence / azimuth angle from metadata')

        # loop for calculation
        prog_bar = ptime.progressBar(maxValue=num_site, print_msg=print_msg)
        for i, site_name in enumerate(site_names):
            prog_bar.update(i+1, suffix='{}/{} {}'.format(i+1, num_site, site_name))

            # calculate gps data value
            obj = GPS(site_name, source=source)
            obj.open(print_msg=print_msg)
            vel, dis_ts = obj.get_gps_los_velocity(
                geom_obj,
                start_date=start_date,
                end_date=end_date,
                gps_comp=gps_comp,
                horz_az_angle=horz_az_angle,
                model=model)

            # ignore time-series if the estimated velocity is nan
            dis = np.nan if np.isnan(vel) else dis_ts[-1] - dis_ts[0]

            # save data to list
            data_list.append([obj.site, obj.site_lon, obj.site_lat, dis, vel])
        prog_bar.close()

        # # discard invalid sites
        # flag = np.isnan([x[-1] for x in data_list])
        # vprint('discard extra {} stations due to limited overlap/observations in time:'.format(np.sum(flag)))
        # vprint('  {}'.format(np.array(data_list)[flag][:,0].tolist()))
        # data_list = [x for x in data_list if not np.isnan(x[-1])]

        # write to CSV file
        vprint('write GPS observations to file: {}'.format(csv_file))
        with open(csv_file, 'w') as fc:
            fcw = csv.writer(fc)
            fcw.writerow(col_names)
            fcw.writerows(data_list)

        # prepare API output
        site_obs = np.array([x[obs_ind] for x in data_list])

    return site_obs



#################################### Beginning of GPS-GSI utility functions ########################
def read_pos_file(fname):
    import codecs
    fcp = codecs.open(fname, encoding = 'cp1252')
    fc = np.loadtxt(fcp, skiprows=20, dtype=str, comments=('*','-DATA'))

    ys = fc[:,0].astype(int)
    ms = fc[:,1].astype(int)
    ds = fc[:,2].astype(int)
    dates = [dt.datetime(year=y, month=m, day=d) for y,m,d in zip(ys, ms, ds)]

    X = fc[:,4].astype(float64).tolist()
    Y = fc[:,5].astype(float64).tolist()
    Z = fc[:,6].astype(float64).tolist()

    return dates, X, Y, Z


def get_pos_years(gps_dir, site):
    fnames = glob.glob(os.path.join(gps_dir, '{}.*.pos'.format(site)))
    years = [os.path.basename(i).split('.')[1] for i in fnames]
    years = ptime.yy2yyyy(years)
    return years


def read_GSI_F3(gps_dir, site, start_date=None, end_date=None):
    year0 = int(start_date[0:4])
    year1 = int(end_date[0:4])
    num_year = year1 - year0 + 1

    dates, X, Y, Z = [], [], [], []
    for i in range(num_year):
        yeari = str(year0 + i)
        fname = os.path.join(gps_dir, '{}.{}.pos'.format(site, yeari[2:]))
        datesi, Xi, Yi, Zi = read_pos_file(fname)
        dates += datesi
        X += Xi
        Y += Yi
        Z += Zi
    dates = np.array(dates)
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    date0 = dt.datetime.strptime(start_date, "%Y%m%d")
    date1 = dt.datetime.strptime(end_date, "%Y%m%d")
    flag = np.ones(X.shape, dtype=np.bool_)
    flag[dates < date0] = False
    flag[dates > date1] = False
    return dates[flag], X[flag], Y[flag], Z[flag]


#################################### End of GPS-GSI utility functions ##############################



#################################### Beginning of GPS class ########################################
class GPS:
    """GPS class for GPS time-series of daily solution.
    """

    def __init__(self, site: str, source='UNR', data_dir='./GPS',
                 version='IGS14'):
        # Check inputs
        assert source in supported_sources, \
            f'Source {source:s} not supported. Use one of {supported_sources}'

        # Record properties
        self.site = site
        self.source = source
        self.version = version

        # Create data directory if not exist
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        return None


    def open(self, file=None, print_msg=True):
        """Read the lat/lon and displacement data of the station.
        Download if necessary.
        """
        # Download file if not present
        if not hasattr(self, 'file'):
            self.dload_site(print_msg=print_msg)

        # Retrieve data from file
        self.get_stat_lat_lon(print_msg=print_msg)
        self.read_displacement(print_msg=print_msg)

        return None


    def dload_site(self, print_msg=True):
        """Download the station displacement data from the
        specified source.

        Modifies:   self.file     : str for local file path/name
                    self.file_url : str for file URL

        Returns:    self.file
        """
        if print_msg == True:
            print(f"Downloading data for site {self.site:s} from {self.source:s}")

        # Download displacement data based on source
        if self.source == 'UNR':
            self.__download_unr_file__(print_msg=print_msg)
        elif self.source == 'ESESES':
            self.__download_eseses_file__(print_msg=print_msg)

        return self.file

    def __download_unr_file__(self, print_msg):
        """Download GPS displacement data from UNR.
        """
        # URL and file name specs
        url_prefix = 'http://geodesy.unr.edu/gps_timeseries/tenv3'
        if self.version == 'IGS08':
            self.file = os.path.join(self.data_dir,
                                     '{site:s}.{version:s}.tenv3'.\
                                     format(site=self.site))
        elif self.version == 'IGS14':
            self.file = os.path.join(self.data_dir,
                                    '{site:s}.tenv3'.\
                                    format(site=self.site))
        self.file_url = os.path.join(url_prefix, self.version,
                                     os.path.basename(self.file))

        # Download file if not present
        if os.path.exists(self.file):
            print(f'File {self.file} exists--reading')
        else:
            if print_msg == True:
                print(f'... downloading {self.file_url:s} to {self.file:s}')
            urlretrieve(self.file_url, self.file)

        return None

    def __download_eseses_file__(self, print_msg):
        """Download GPS displacement data from ESESES.
        """
        # URL and file name specs
        url_prefix = 'http://garner.ucsd.edu/pub/measuresESESES_products/Timeseries/CurrentUntarred/Clean_TrendNeuTimeSeries_comb_20240320'
        self.file = os.path.join(self.data_dir,
                                 '{site:s}CleanTrend.neu.Z'.\
                                 format(site=self.site.lower()))
        self.file_url = os.path.join(url_prefix, os.path.basename(self.file))

        # Download file if not present
        if os.path.exists(self.file):
            print(f'File {self.file} exists--reading')
        else:
            if print_msg == True:
                print(f'... downloading {self.file_url:s} to {self.file:s}')
            urlretrieve(self.file_url, self.file)

        # Unzip file
        with zipfile.ZipFile(self.file, 'r') as Zfile:
            Zfile.extractall(self.data_dir)

        # Update file name
        self.file = self.file.strip('.Z')
        if print_msg == True:
            print(f'... extracted to {self.file:s}')

        return None


    def get_stat_lat_lon(self, print_msg=True):
        """Get station lat/lon based on processing source.
        Retrieve data from the displacement file.
        """
        if print_msg == True:
            print('calculating station lat/lon')
        if not os.path.isfile(self.file):
            self.dload_site(print_msg=print_msg)

        # Retrieve lat/lon based on processing source
        if self.source == 'UNR':
            self.__get_unr_lat_lon__()
        elif self.source == 'ESESES':
            self.__get_eseses_lat_lon__()

        if print_msg == True:
            print(f'\t{self.site_lat:f}, {self.site_lon:f}')

        return self.site_lat, self.site_lon

    def __get_unr_lat_lon__(self):
        """Get station lat/lon for UNR data.
        """
        data = np.loadtxt(self.file, dtype=bytes, skiprows=1).astype(str)
        ref_lon, ref_lat = float(data[0, 6]), 0.
        e0, e_off, n0, n_off = data[0, 7:11].astype(float)
        e0 += e_off
        n0 += n_off

        az = np.arctan2(e0, n0) / np.pi * 180.
        dist = np.sqrt(e0**2 + n0**2)
        g = Geod(ellps='WGS84')
        self.site_lon, self.site_lat = g.fwd(ref_lon, ref_lat, az, dist)[0:2]

        return None

    @staticmethod
    def lon_360to180(lon: float) -> float:
        """Convert longitude in the range [0, 360) to
        range (-180, 180].
        """
        if lon > 180:
            lon -= 360
        return lon

    def __get_eseses_lat_lon__(self):
        """Get station lat/lon for ESESES data.
        """
        with open(self.file, 'r') as data_file:
            # Read raw file contents
            lines = data_file.readlines()

            # Determine reference latitude
            lat_line = [line for line in lines \
                        if line.find('# Latitude') != -1]
            lat_line = lat_line[0].strip('\n')
            self.site_lat = float(lat_line.split()[-1])

            # Determine reference longitude
            lon_line = [line for line in lines \
                        if line.find('# East Longitude') != -1]
            lon_line = lon_line[0].strip('\n')
            site_lon = float(lon_line.split()[-1])
            self.site_lon = self.lon_360to180(site_lon)

        return None


    def read_displacement(self, start_date=None, end_date=None, print_msg=True,
                          display=False):
        """Read GPS displacement time-series (defined by start/end_date)
        Parameters: start/end_date : str in YYYYMMDD format
        Returns:    dates : 1D np.ndarray of datetime.datetime object
                    dis_e/n/u : 1D np.ndarray of displacement in meters in float32
                    std_e/n/u : 1D np.ndarray of displacement STD in meters in float32
        """
        # Download file if it does not exist
        if not os.path.isfile(self.file):
            self.dload_site(print_msg=print_msg)

        # Read dates, dis_e, dis_n, dis_u
        if print_msg == True:
            print('reading time and displacement in east/north/vertical direction')

        if self.source == 'UNR':
            self.__read_unr_displacement__()
        elif self.source == 'ESESES':
            self.__read_eseses_displacement__()

        # Cut out the specified time range
        self.__crop_to_date_range__(start_date, end_date)

        # Display if requested
        if display == True:
            self.display_data()

        return (self.dates,
                self.dis_e, self.dis_n, self.dis_u,
                self.std_e, self.std_n, self.std_u)

    def __read_unr_displacement__(self):
        """Read GPS displacement time-series processed by UNR.
        """
        # Read data from file
        data = np.loadtxt(self.file, dtype=bytes, skiprows=1).astype(str)

        # Parse dates
        self.dates = np.array([dt.datetime.strptime(i, "%y%b%d") \
                               for i in data[:,1]])
        self.date_list = [x.strftime('%Y%m%d') for x in self.dates]

        # Parse displacement data
        (self.dis_e,
         self.dis_n,
         self.dis_u,
         self.std_e,
         self.std_n,
         self.std_u) = data[:, (8,10,12,14,15,16)].astype(np.float32).T

        return None

    def __read_eseses_displacement__(self):
        """Read GPS displacement time-series processed by ESESES.
        """
        # Read data from file
        data = np.loadtxt(self.file, usecols=tuple(range(0,12)))
        n_data = data.shape[0]

        # Parse dates
        dates = [dt.datetime(int(data[i,1]), 1, 1) \
                 + dt.timedelta(days=int(data[i,2])) \
                 for i in range(n_data)]
        self.dates = np.array(dates)
        self.date_list = [date.strftime('%Y%m%d') for date in self.dates]

        # Parse displacement data
        (self.dis_n,
         self.dis_e,
         self.dis_u,
         self.std_n,
         self.std_e,
         self.std_u) = data[:, 3:9].astype(np.float32).T / 1000

        return None

    def __crop_to_date_range__(self, start_date: str, end_date: str):
        """Cut out the specified time range.
        start/end_date in format YYYYMMDD
        """
        t_flag = np.ones(len(self.dates), bool)
        if start_date:
            t0 = ptime.date_list2vector([start_date])[0][0]
            t_flag[self.dates < t0] = 0
        if end_date:
            t1 = ptime.date_list2vector([end_date])[0][0]
            t_flag[self.dates > t1] = 0
        self.dates = self.dates[t_flag]
        self.dis_e = self.dis_e[t_flag]
        self.dis_n = self.dis_n[t_flag]
        self.dis_u = self.dis_u[t_flag]
        self.std_e = self.std_e[t_flag]
        self.std_n = self.std_n[t_flag]
        self.std_u = self.std_u[t_flag]

        return None


    def display_data(self):
        """Display displacement data.
        """
        # Instantiate figure and axes
        fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

        # Plot data
        ax[0].scatter(self.dates, self.dis_e, s=2**2,
                      c='k', label='East')
        ax[1].scatter(self.dates, self.dis_n, s=2**2,
                      c='k', label='North')
        ax[2].scatter(self.dates, self.dis_u, s=2**2,
                      c='k', label='Up')

        # Format plot
        fig.suptitle(f'{self.site:s} ({self.source:s})')

        plt.show()

        return fig, ax


    #####################################  Utility Functions ###################################
    def displacement_enu2los(self, inc_angle:float, az_angle:float, gps_comp='enu2los',
                             horz_az_angle=-90., display=False, model=None):
        """Convert displacement in ENU to LOS direction.

        Parameters: inc_angle     - float, LOS incidence angle in degree
                    az_angle      - float, LOS aziuth angle in degree
                                    from the north, defined as positive in clock-wise direction
                    gps_comp      - str, GPS components used to convert to LOS direction
                    horz_az_angle - float, fault azimuth angle used to convert horizontal to fault-parallel
                                    measured from the north with anti-clockwise as positive
        Returns:    dis_los       - 1D np.array for displacement in LOS direction
                    std_los       - 1D np.array for displacement standard deviation in LOS direction
        """
        # get unit vector for the component of interest
        unit_vec = ut.get_unit_vector4component_of_interest(
            los_inc_angle=inc_angle,
            los_az_angle=az_angle,
            comp=gps_comp.lower(),
            horz_az_angle=horz_az_angle,
        )

        # convert ENU to LOS direction
        self.dis_los = (  self.dis_e * unit_vec[0]
                        + self.dis_n * unit_vec[1]
                        + self.dis_u * unit_vec[2])
        # assuming ENU component are independent with each other
        self.std_los = (   (self.std_e * unit_vec[0])**2
                         + (self.std_n * unit_vec[1])**2
                         + (self.std_u * unit_vec[2])**2 ) ** 0.5

        # display if requested
        if display == True:
            # Instantiate figure and axes
            fig, ax = plt.subplots(sharex=True)

            # Plot LOS displacement
            ax.scatter(self.dates, self.dis_los, s=2**2,
                       c='k', label='LOS')

            # Plot fit if model specified
            if model is not None:
                # specific time_func model
                date_list = [dt.datetime.strftime(i, '%Y%m%d') for i in dates]
                A = time_func.get_design_matrix4time_func(date_list, model=model)
                estm_dis = np.dot(np.linalg.pinv(A), self.dis_los)

        return self.dis_los, self.std_los


    def get_los_geometry(self, geom_obj, print_msg=False):
        """Get the Line-of-Sight geometry info in incidence and azimuth angle in degrees."""
        lat, lon = self.get_stat_lat_lon(print_msg=print_msg)

        # get LOS geometry
        if isinstance(geom_obj, str):
            # geometry file
            atr = readfile.read_attribute(geom_obj)
            coord = coordinate(atr, lookup_file=geom_obj)
            y, x = coord.geo2radar(lat, lon, print_msg=print_msg)[0:2]
            # check against image boundary
            y = max(0, y);  y = min(int(atr['LENGTH'])-1, y)
            x = max(0, x);  x = min(int(atr['WIDTH'])-1, x)
            box = (x, y, x+1, y+1)
            inc_angle = readfile.read(geom_obj, datasetName='incidenceAngle', box=box, print_msg=print_msg)[0][0,0]
            az_angle  = readfile.read(geom_obj, datasetName='azimuthAngle',   box=box, print_msg=print_msg)[0][0,0]

        elif isinstance(geom_obj, dict):
            # use mean inc/az_angle from metadata
            inc_angle = ut.incidence_angle(geom_obj, dimension=0, print_msg=print_msg)
            az_angle  = ut.heading2azimuth_angle(float(geom_obj['HEADING']))

        else:
            raise ValueError(f'input geom_obj is neither str nor dict: {geom_obj}')

        return inc_angle, az_angle


    def read_gps_los_displacement(self, geom_obj, start_date=None, end_date=None, ref_site=None,
                                  gps_comp='enu2los', horz_az_angle=-90., print_msg=False):
        """Read GPS displacement in LOS direction.

        Parameters: geom_obj      - dict / str, metadata of InSAR file, or geometry file path
                    start_date    - str in YYYYMMDD format
                    end_date      - str in YYYYMMDD format
                    ref_site      - str, reference GPS site
                    gps_comp      - str, GPS components used to convert to LOS direction
                    horz_az_angle - float, fault azimuth angle used to convert horizontal to fault-parallel
        Returns:    dates         - 1D np.array of datetime.datetime object
                    dis/std       - 1D np.array of displacement / uncertainty in meters
                    site_lalo     - tuple of 2 float, lat/lon of GPS site
                    ref_site_lalo - tuple of 2 float, lat/lon of reference GPS site
        """
        # read GPS object
        inc_angle, az_angle = self.get_los_geometry(geom_obj)
        dates = self.read_displacement(start_date, end_date, print_msg=print_msg)[0]
        dis, std = self.displacement_enu2los(inc_angle, az_angle, gps_comp=gps_comp, horz_az_angle=horz_az_angle)
        site_lalo = self.get_stat_lat_lon(print_msg=print_msg)

        # get LOS displacement relative to another GPS site
        if ref_site:
            ref_obj = GPS(site=ref_site, data_dir=self.data_dir)
            ref_obj.read_displacement(start_date, end_date, print_msg=print_msg)
            inc_angle, az_angle = ref_obj.get_los_geometry(geom_obj)
            ref_obj.displacement_enu2los(inc_angle, az_angle, gps_comp=gps_comp, horz_az_angle=horz_az_angle)
            ref_site_lalo = ref_obj.get_stat_lat_lon(print_msg=print_msg)

            # get relative LOS displacement on common dates
            dates = np.array(sorted(list(set(self.dates) & set(ref_obj.dates))))
            dis = np.zeros(dates.shape, np.float32)
            std = np.zeros(dates.shape, np.float32)
            for i, date_i in enumerate(dates):
                idx1 = np.where(self.dates == date_i)[0][0]
                idx2 = np.where(ref_obj.dates == date_i)[0][0]
                dis[i] = self.dis_los[idx1] - ref_obj.dis_los[idx2]
                std[i] = (self.std_los[idx1]**2 + ref_obj.std_los[idx2]**2)**0.5
        else:
            ref_site_lalo = None

        return dates, dis, std, site_lalo, ref_site_lalo


    def get_gps_los_velocity(self, geom_obj, start_date=None, end_date=None,
                             ref_site=None, gps_comp='enu2los',
                             horz_az_angle=-90., model=None,
                             print_msg=True):
        """Convert the three-component displacement data into LOS
        velocity.

        Parameters: geom_obj        : dict / str, metadata of InSAR file, or
                                      geometry file path
                    start_date      : string in YYYYMMDD format
                    end_date        : string in YYYYMMDD format
                    ref_site        : string, reference GPS site
                    gps_comp        : string, GPS components used to convert to
                                      LOS direction
                    horz_az_angle   : float, fault azimuth angle used to convert
                                 horizontal to fault-parallel
                    model           : dict, time function model, e.g.
                                      {'polynomial': 1, 'periodic': [1.0, 0.5]}
        Returns:    dates : 1D np.array of datetime.datetime object
                    dis   : 1D np.array of displacement in meters
                    std   : 1D np.array of displacement uncertainty in meters
                    site_lalo : tuple of 2 float, lat/lon of GPS site
                    ref_site_lalo : tuple of 2 float, lat/lon of reference GPS site
        """
        # Retrieve displacement data
        dates, dis = self.read_gps_los_displacement(geom_obj,
                                                    start_date=start_date,
                                                    end_date=end_date,
                                                    ref_site=ref_site,
                                                    gps_comp=gps_comp,
                                                    horz_az_angle=horz_az_angle)[:2]

        # displacement -> velocity
        # if:
        # 1. num of observations > 2 AND
        # 2. time overlap > 1/4
        dis2vel = True
        if len(dates) <= 2:
            dis2vel = False
        elif start_date and end_date:
            t0 = ptime.date_list2vector([start_date])[0][0]
            t1 = ptime.date_list2vector([end_date])[0][0]
            if dates[-1] - dates[0] <= (t1 - t0) / 4:
                dis2vel = False

        if dis2vel:
            # specific time_func model
            date_list = [dt.datetime.strftime(i, '%Y%m%d') for i in dates]
            A = time_func.get_design_matrix4time_func(date_list, model=model)
            self.velocity = np.dot(np.linalg.pinv(A), dis)[1]
        else:
            self.velocity = np.nan
            if print_msg == True:
                print(f'Velocity calculation failed for site {self.site}')

        return self.velocity, dis


#################################### End of GPS class ####################################
