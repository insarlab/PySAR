"""Classes for HDF5/MintPy file operations."""
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, 2018               #
############################################################
# Recommend import:
#   from mintpy.objects import timeseries, ifgramStack, geometry


import datetime as dt
import itertools
import os
import time

import h5py
import numpy as np
from scipy import ndimage

from mintpy.utils import ptime

##------------------ Global Variables ---------------------##

DATA_TYPE_DICT = {
    'bool': np.bool_, 'byte': np.bool_, 'flag': np.bool_,
    'int': np.int16, 'int16': np.int16, 'short': np.int16, 'int32': np.int32,
    'int64': np.int64, 'long': np.int64,
    'float': np.float32, 'float32': np.float32,
    'float_': np.float64, 'float64': np.float64,
    'complex': np.complex64, 'complex64': np.complex64, 'cpx_float32': np.complex64,
    'cfloat': np.complex64, 'cfloat32': np.complex64,
    'complex128': np.complex128, 'complex_': np.complex128, 'cpx_float64': np.complex128,
}

TIMESERIES_KEY_NAMES = ['timeseries', 'HDFEOS', 'giantTimeseries']

TIMESERIES_DSET_NAMES = [
    'timeseries',
    'raw',
    'troposphericDelay',
    'topographicResidual',
    'ionosphericDelay',
    'ionosphericBurstRamp',
    'ramp',
    'displacement',
]

GEOMETRY_DSET_NAMES = [
    # coordinates
    'height',
    'latitude',
    'longitude',
    'rangeCoord',
    'azimuthCoord',
    # others
    'incidenceAngle',
    'azimuthAngle',
    'slantRangeDistance',
    'shadowMask',
    'waterMask',
    'commonMask',
    'bperp',
]

IFGRAM_DSET_NAMES = [
    # interferogram
    'unwrapPhase',
    'unwrapPhase_bridging_phaseClosure',
    'unwrapPhase_bridging',
    'unwrapPhase_phaseClosure',
    'coherence',
    'connectComponent',
    'wrapPhase',
    'magnitude',
    # offset
    'azimuthOffset',
    'azimuthOffsetStd',
    'rangeOffset',
    'rangeOffsetStd',
    'offsetSNR',
    'refPhase',
]

DSET_UNIT_DICT = {
    # interferogram
    'unwrapPhase'      : 'radian',
    'coherence'        : '1',
    'connectComponent' : '1',
    'wrapPhase'        : 'radian',
    'magnitude'        : '1',

    # offset
    'azimuthOffset'    : 'pixel',
    'azimuthOffsetStd' : 'pixel',
    'rangeOffset'      : 'pixel',
    'rangeOffsetStd'   : 'pixel',
    'offsetSNR'        : '1',

    # geometry
    'height'             : 'm',
    'latitude'           : 'degree',
    'longitude'          : 'degree',
    'rangeCoord'         : '1',
    'azimuthCoord'       : '1',
    'incidenceAngle'     : 'degree',
    'azimuthAngle'       : 'degree',
    'slantRangeDistance' : 'm',
    'shadowMask'         : '1',
    'waterMask'          : '1',
    'commonMask'         : '1',
    'bperp'              : 'm',

    # time-series
    'timeseries'          : 'm',
    'raw'                 : 'm',
    'troposphericDelay'   : 'm',
    'topographicResidual' : 'm',
    'ramp'                : 'm',
    'displacement'        : 'm',

    # others
    'temporalCoherence' : '1',
    'velocity'          : 'm/year',
    'acceleration'      : 'm/year^2',
    'mask'              : '1',

    # giant
    'giantTimeseries' : 'mm',
    'recons'          : 'mm',
    'rawts'           : 'mm',
    'sar_aps'         : 'mm',
    'igram_aps'       : 'mm',
    'figram'          : 'mm',
    'igram'           : 'mm',
    'cmask'           : '1',
    'ifgcnt'          : '1',

    # binary
    'unw'       : 'radian',
    'int'       : 'radian',
    'flat'      : 'radian',
    'cor'       : '1',
    'dem'       : 'm',
    'hgt'       : 'm',
    'hgt_sim'   : 'm',
    'intensity' : '1',
}



################################ timeseries class begin ################################
class timeseries:
    """
    Time-series object for displacement of a set of SAR images from the same platform and track.
    It contains three datasets in root level: date, bperp and timeseries.

    File structure: https://mintpy.readthedocs.io/en/latest/api/data_structure/#timeseries
    """

    def __init__(self, file=None):
        self.file = file
        self.name = 'timeseries'

    def close(self, print_msg=True):
        try:
            self.f.close()
            if print_msg:
                print(f'close timeseries file: {os.path.basename(self.file)}')
        except:
            pass
        return None

    def open(self, print_msg=True):
        if print_msg:
            print(f'open {self.name} file: {os.path.basename(self.file)}')
        self.get_metadata()
        self.get_size()
        self.get_date_list()
        self.numPixel = self.length * self.width

        with h5py.File(self.file, 'r') as f:
            try:
                self.pbase = f['bperp'][:]
                self.pbase -= self.pbase[self.refIndex]
            except:
                self.pbase = None

        # time info
        self.dateFormat = ptime.get_date_str_format(self.dateList[0])
        self.times = np.array([dt.datetime.strptime(i, self.dateFormat) for i in self.dateList])
        # add hh/mm/ss info to the datetime objects
        if 'T' not in self.dateFormat or all(i.hour==0 and i.minute==0 for i in self.times):
            if 'CENTER_LINE_UTC' in self.metadata.keys():
                utc_sec = float(self.metadata['CENTER_LINE_UTC'])
                self.times = np.array([i + dt.timedelta(seconds=utc_sec) for i in self.times])
        self.tbase = np.array([(i.days + i.seconds / (24 * 60 * 60))
                               for i in (self.times - self.times[self.refIndex])],
                              dtype=np.float32)

        # list of float for year, 2014.95
        self.yearList = [i.year + (i.timetuple().tm_yday-1)/365.25 for i in self.times]
        self.sliceList = [f'{self.name}-{i}' for i in self.dateList]
        return None

    def get_metadata(self):
        with h5py.File(self.file, 'r') as f:
            self.metadata = dict(f.attrs)
            dates = f['date'][:]
        for key, value in self.metadata.items():
            try:
                self.metadata[key] = value.decode('utf8')
            except:
                self.metadata[key] = value

        # ref_date/index
        dateList = [i.decode('utf8') for i in dates]
        if 'REF_DATE' not in self.metadata.keys():
            self.metadata['REF_DATE'] = dateList[0]
        self.refIndex = dateList.index(self.metadata['REF_DATE'])
        self.metadata['START_DATE'] = dateList[0]
        self.metadata['END_DATE'] = dateList[-1]
        return self.metadata

    def get_size(self):
        with h5py.File(self.file, 'r') as f:
            self.numDate, self.length, self.width = f[self.name].shape[-3:]
        return self.numDate, self.length, self.width

    def get_date_list(self):
        with h5py.File(self.file, 'r') as f:
            self.dateList = [i.decode('utf8') for i in f['date'][:]]
        return self.dateList

    def read(self, datasetName=None, box=None, squeeze=True, print_msg=True):
        """Read dataset from timeseries file
        Parameters: self : timeseries object
                    datasetName : (list of) string in YYYYMMDD format
                    box : tuple of 4 int, indicating x0,y0,x1,y1 of range
        Returns:    data : 2D or 3D dataset
        Examples:   from mintpy.objects import timeseries
                    tsobj = timeseries('timeseries_ERA5_demErr.h5')
                    data = tsobj.read(datasetName='20161020')
                    data = tsobj.read(datasetName='20161020', box=(100,300,500,800))
                    data = tsobj.read(datasetName=['20161020','20161026','20161101'])
                    data = tsobj.read(box=(100,300,500,800))
        """
        if print_msg:
            print(f'reading {self.name} data from file: {self.file} ...')
        self.open(print_msg=False)

        # convert input datasetName into list of dates
        if not datasetName or datasetName == 'timeseries':
            datasetName = []
        elif isinstance(datasetName, str):
            datasetName = [datasetName]
        datasetName = [i.replace('timeseries', '').replace('-', '') for i in datasetName]

        with h5py.File(self.file, 'r') as f:
            ds = f[self.name]
            if isinstance(ds, h5py.Group):  # support for old mintpy files
                ds = ds[self.name]

            # Get dateFlag - mark in time/1st dimension
            dateFlag = np.zeros((self.numDate), dtype=np.bool_)
            if not datasetName:
                dateFlag[:] = True
            else:
                for e in datasetName:
                    dateFlag[self.dateList.index(e)] = True

            # Get Index in space/2_3 dimension
            if box is None:
                box = [0, 0, self.width, self.length]
            xsize = box[2] - box[0]
            ysize = box[3] - box[1]

            # read
            num_slice = np.sum(dateFlag)
            inds = np.where(dateFlag)[0].tolist()

            if num_slice / dateFlag.size < 0.05:
                # single indexing if only a small fraction is read
                data = np.zeros((num_slice, ysize, xsize), dtype=ds.dtype)
                for i, ind in enumerate(inds):
                    data[i] = ds[ind,
                                 box[1]:box[3],
                                 box[0]:box[2]]
            else:
                data = ds[:,
                          box[1]:box[3],
                          box[0]:box[2]][dateFlag]

            if squeeze and any(i == 1 for i in data.shape):
                data = np.squeeze(data)
        return data


    def write2hdf5(self, data, outFile=None, dates=None, bperp=None, metadata=None, refFile=None, compression=None):
        """
        Parameters: data  : 3D array of float32
                    dates : 1D array/list of string in YYYYMMDD format
                    bperp : 1D array/list of float32 (optional)
                    metadata : dict
                    outFile : string
                    refFile : string
                    compression : string or None
        Returns: outFile : string
        Examples:
            from mintpy.objects import timeseries

            ##Generate a new timeseries file
            tsobj = timeseries('timeseries.h5')
            timeseries.write(data, dates=dateList, bperp=bperp, metadata=atr)

            ##Generate a timeseries with same attributes and same date/bperp info
            tsobj = timeseries('timeseries_demErr.h5')
            timeseries.write(data, refFile='timeseries.h5')
        """
        if not outFile:
            outFile = self.file
        if refFile:
            refobj = timeseries(refFile)
            refobj.open(print_msg=False)
            if metadata is None:
                metadata = refobj.metadata
            if dates is None:
                dates = refobj.dateList
            if bperp is None:
                bperp = refobj.pbase
            # get ref file compression type if input compression is None
            if compression is None:
                with h5py.File(refFile, 'r') as rf:
                    compression = rf[TIMESERIES_DSET_NAMES[0]].compression
            refobj.close(print_msg=False)
        data = np.array(data, dtype=np.float32)
        dates = np.array(dates, dtype=np.bytes_)
        bperp = np.array(bperp, dtype=np.float32)
        metadata = dict(metadata)
        metadata['FILE_TYPE'] = self.name

        # directory
        outDir = os.path.dirname(os.path.abspath(outFile))
        if not os.path.isdir(outDir):
            os.makedirs(outDir)
            print(f'create directory: {outDir}')

        # 3D dataset - timeseries
        print(f'create timeseries HDF5 file: {outFile} with w mode')
        with h5py.File(outFile, 'w') as f:
            print(('create dataset /timeseries of {t:<10} in size of {s} '
                   'with compression={c}').format(t=str(data.dtype),
                                                  s=data.shape,
                                                  c=compression))
            f.create_dataset('timeseries',
                             data=data,
                             chunks=True,
                             compression=compression)

            # 1D dataset - date / bperp
            print(f'create dataset /dates      of {str(dates.dtype):<10} in size of {dates.shape}')
            f.create_dataset('date', data=dates)

            if bperp.shape != ():
                print(f'create dataset /bperp      of {str(bperp.dtype):<10} in size of {bperp.shape}')
                f.create_dataset('bperp', data=bperp)

            # Attributes
            for key, value in metadata.items():
                f.attrs[key] = str(value)
        print(f'finished writing to {outFile}')
        return outFile

    def timeseries_std(self, maskFile=None, outFile=None):
        """Calculate the standard deviation (STD) for acquisition of time-series,
           output result to a text file.
        """
        # Calculate STD
        data = self.read()
        if maskFile:
            mask = singleDataset(maskFile).read()
            print('read mask from file: '+maskFile)
            data[:, mask == 0] = np.nan
        self.std = np.nanstd(data, axis=(1, 2))

        # Write text file
        header = 'Standard Deviation in space for each acquisition of time-series\n'
        header += f'Timeseries file: {self.file}\n'
        header += f'Mask file: {maskFile}\n'
        header += 'Date\t\tSTD (m)'
        if not outFile:
            outFile = os.path.join(os.path.dirname(os.path.abspath(self.file)),
                                   f'std_{os.path.splitext(os.path.basename(self.file))[0]}.txt')
        np.savetxt(outFile, np.hstack((np.array(self.dateList).reshape(-1, 1), self.std.reshape(-1, 1))),
                   fmt='%s', delimiter='\t', header=header)
        print(f'save timeseries STD to text file: {outFile}')
        return outFile

    def timeseries_rms(self, maskFile=None, outFile=None):
        """Calculate the Root Mean Square for each acquisition of time-series
            and output result to a text file.
        """
        # Get date list
        date_list = self.get_date_list()
        num_date  = len(date_list)

        # Get mask
        if maskFile and os.path.isfile(maskFile):
            print('read mask from file: '+maskFile)
            mask = singleDataset(maskFile).read()

        # Calculate RMS one date at a time
        self.rms = np.zeros(num_date) * np.nan
        print(f'reading {self.name} data from file: {self.file} ...')
        prog_bar = ptime.progressBar(maxValue=num_date)
        for i in range(num_date):
            data = self.read(datasetName=f'{date_list[i]}', print_msg=False)
            if maskFile and os.path.isfile(maskFile):
                data[mask == 0] = np.nan
            self.rms[i] = np.sqrt(np.nanmean(np.square(data), axis=(0, 1)))
            prog_bar.update(i+1, suffix=f'{i+1}/{num_date}')
        prog_bar.close()

        # Write text file
        header = 'Root Mean Square in space for each acquisition of time-series\n'
        header += f'Timeseries file: {self.file}\n'
        header += f'Mask file: {maskFile}\n'
        header += 'Date\t\tRMS (m)'
        if not outFile:
            outFile = os.path.join(os.path.dirname(os.path.abspath(self.file)),
                                   f'rms_{os.path.splitext(os.path.basename(self.file))[0]}.txt')
        np.savetxt(outFile, np.hstack((np.array(self.dateList).reshape(-1, 1), self.rms.reshape(-1, 1))),
                   fmt='%s', delimiter='\t', header=header)
        print(f'save timeseries RMS to text file: {outFile}')
        return outFile

    def spatial_average(self, maskFile=None, box=None, reverseMask=False, threshold=None):
        self.open(print_msg=False)
        data = self.read(box=box)
        if maskFile and os.path.isfile(maskFile):
            print('read mask from file: '+maskFile)
            mask = singleDataset(maskFile).read(box=box)
            data[:, mask == int(reverseMask)] = np.nan

        # calculate area ratio if threshold is specified
        # percentage of pixels with value above the threshold
        if threshold is not None:
            data[data > threshold] = 1
            data[data <= threshold] = 0

        dmean = np.nanmean(data, axis=(1, 2))
        return dmean, self.dateList

    def temporal_average(self):
        print(f'calculating the temporal average of timeseries file: {self.file}')
        self.open(print_msg=False)
        data = self.read(squeeze=False)
        dmean = np.nanmean(data, axis=0)
        return dmean


    def temporal_derivative(self, out_file):
        print(f'calculating the temporal derivative of timeseries file: {self.file}')

        # read
        print('reading timeseries data')
        self.open(print_msg=False)
        ts_data = self.read(print_msg=False)

        # calculate
        print('calculate the 1st derivative of timeseries data')
        ts_data_1d = np.zeros(ts_data.shape, np.float32)
        ts_data_1d[1:, :, :] = np.diff(ts_data, n=1, axis=0)

        # write
        if not out_file:
            fbase = os.path.splitext(self.file)[0]
            out_file = f'{fbase}_1stDiff.h5'
        self.write2hdf5(ts_data_1d, outFile=out_file, refFile=self.file)

        return out_file


    def temporal_filter(self, time_win=1.0, filter_type='guassian', out_file=None):
        """Filter the time-series in time with a moving window.

        Parameters: time_win    - float, sigma of Gaussian distribution in time in months (30.4 days)
                    filter_type - str, filter type: gaussian, median
                    out_file    - str, output file name
        Returns:    out_file    - str, output file name
        """
        # use integer type if possible for shorter default output file name
        time_win = int(time_win) if float(time_win).is_integer() else time_win

        # output file
        if not out_file:
            fbase = os.path.splitext(self.file)[0]
            out_file = f'{fbase}_temp{filter_type.capitalize()}{time_win}.h5'
        print(f'output file: {out_file}')

        # read
        self.open()
        ts_data = self.read().reshape(self.numDate, -1)

        print('-'*50)
        ts_data_filt = np.zeros(ts_data.shape, np.float32)

        if filter_type == 'gaussian':
            print(f'temporal filtering via a Gaussian window of {time_win} months')
            tbase = self.tbase.reshape(-1, 1) / 365.25 * 12  # months (30.4 days)
            prog_bar = ptime.progressBar(maxValue=self.numDate)
            for i in range(self.numDate):

                # calc weight from Gaussian (normal) distribution in time
                tbase_diff = tbase[i] - tbase
                weight = np.exp(-0.5 * (tbase_diff**2) / (time_win**2))
                weight /= np.sum(weight)

                # smooth the current acquisition via Gaussian weighting
                ts_data_filt[i, :] = np.sum(ts_data * weight, axis=0)

                prog_bar.update(i+1, suffix=self.dateList[i])
            prog_bar.close()

        elif filter_type == 'median':
            print(f'temporal filtering via scipy.ndimage.median_filter of {time_win} acquisitions')
            ts_data_filt = ndimage.median_filter(
                ts_data,
                size=(time_win, 1),
                mode='nearest',
            )

        else:
            raise ValueError(f'un-supported temporal filter: {filter_type}!')
        del ts_data

        # prepare for writing: temporal referencing + reshape
        ts_data_filt -= ts_data_filt[self.refIndex, :]
        ts_data_filt = np.reshape(ts_data_filt, (self.numDate, self.length, self.width))

        # write
        self.write2hdf5(ts_data_filt, outFile=out_file, refFile=self.file)

        return


    def save2bl_list_file(self, out_file='bl_list.txt'):
        """Generate bl_list.txt file from timeseries h5 file."""
        self.open(print_msg=False)
        date6_list = [i[2:8] for i in self.dateList]
        pbase_list = self.pbase.tolist()
        print(f'write baseline list info to file: {out_file}')
        with open(out_file, 'w') as f:
            for d, pbase in zip(date6_list, pbase_list):
                f.write(f'{d}\t{pbase}\n')
        return out_file

################################ timeseries class end ##################################




################################# geometry class begin #################################
class geometry:
    """ Geometry object.

    File structure: https://mintpy.readthedocs.io/en/latest/api/data_structure/#geometry
    """

    def __init__(self, file=None):
        self.file = file
        self.name = 'geometry'

    def close(self, print_msg=True):
        try:
            self.f.close()
            if print_msg:
                print(f'close geometry file: {os.path.basename(self.file)}')
        except:
            pass

    def open(self, print_msg=True):
        if print_msg:
            print(f'open {self.name} file: {os.path.basename(self.file)}')
        self.get_metadata()
        self.get_size()
        self.numPixel = self.length * self.width
        self.geocoded = False
        if 'Y_FIRST' in self.metadata.keys():
            self.geocoded = True

        with h5py.File(self.file, 'r') as f:
            self.datasetNames = [i for i in f.keys() if isinstance(f[i], h5py.Dataset)]
            self.sliceList = list(self.datasetNames)
            if 'bperp' in f.keys():
                self.dateList = [i.decode('utf8') for i in f['date'][:]]
                self.numDate = len(self.dateList)
                # Update bperp datasetNames
                try:
                    self.sliceList.remove('bperp')
                except:
                    pass
                self.sliceList += ['bperp-'+d for d in self.dateList]
            else:
                self.dateList = None

    def get_size(self):
        with h5py.File(self.file, 'r') as f:
            dsName = [i for i in f.keys() if i in GEOMETRY_DSET_NAMES][0]
            dsShape = f[dsName].shape
            if len(dsShape) == 3:
                self.length, self.width = dsShape[1:3]
            else:
                self.length, self.width = dsShape
        return self.length, self.width

    def get_metadata(self):
        with h5py.File(self.file, 'r') as f:
            self.metadata = dict(f.attrs)
        for key, value in self.metadata.items():
            try:
                self.metadata[key] = value.decode('utf8')
            except:
                self.metadata[key] = value
        return self.metadata

    def read(self, datasetName=GEOMETRY_DSET_NAMES[0], box=None, print_msg=True):
        """Read 2D / 3D dataset with bounding box in space
        Parameters: datasetName : (list of) string, to point to specific 2D dataset, e.g.:
                        height
                        incidenceAngle
                        bperp
                        ...
                        bperp-20161020
                        bperp-20161026
                        bperp-...
                    box : tuple of 4 int, for (x0,y0,x1,y1)
                    print_msg : bool
        Returns: data : 2D or 3D array
        Example:
            obj = geometry('./inputs/geometryRadar.h5')
            obj.read(datasetName='height')
            obj.read(datasetName='incidenceAngle')
            obj.read(datasetName='bperp')
            obj.read(datasetName='bperp-20161020')
            obj.read(datasetName=['bperp-20161020',
                                  'bperp-20161026'])
        """
        self.open(print_msg=False)
        if box is None:
            box = (0, 0, self.width, self.length)

        if datasetName is None:
            datasetName = GEOMETRY_DSET_NAMES[0]
        elif isinstance(datasetName, str):
            datasetName = [datasetName]

        with h5py.File(self.file, 'r') as f:
            familyName = datasetName[0].split('-')[0]
            ds = f[familyName]
            if print_msg:
                print(f'reading {familyName:<15} data from file: {self.file} ...')

            if len(ds.shape) == 1:
                data = ds[:]
            elif len(ds.shape) == 2:
                data = ds[box[1]:box[3], box[0]:box[2]]
            else:
                # get dateFlag - mark in time/1st dimension
                dateFlag = np.zeros((ds.shape[0]), dtype=np.bool_)
                datasetName = [i.replace(familyName, '').replace('-', '') for i in datasetName]
                if any(not i for i in datasetName):
                    dateFlag[:] = True
                else:
                    for e in datasetName:
                        dateFlag[self.dateList.index(e)] = True

                # read
                data = ds[:,
                          box[1]:box[3],
                          box[0]:box[2]][dateFlag]

                if any(i == 1 for i in data.shape):
                    data = np.squeeze(data)
        return data
################################# geometry class end ###################################



################################# ifgramStack class begin ##############################
class ifgramStack:
    """ Interferograms Stack object.

    File structure: https://mintpy.readthedocs.io/en/latest/api/data_structure/#ifgramstack
    """

    def __init__(self, file=None):
        self.file = file
        self.name = 'ifgramStack'

    def close(self, print_msg=True):
        try:
            self.f.close()
            if print_msg:
                print(f'close {self.name} file: {os.path.basename(self.file)}')
        except:
            pass

    def open(self, print_msg=True):
        """
        Time format/rules:
            All datetime.datetime objects named with time
            All string in YYYYMMDD        named with date (following roipac)
        """
        if print_msg:
            print(f'open {self.name} file: {os.path.basename(self.file)}')
        self.get_metadata()
        self.get_size()
        self.read_datetimes()
        self.numPixel = self.length * self.width

        # time info
        self.date12List = [f'{i}_{j}' for i, j in zip(self.mDates, self.sDates)]
        self.tbaseIfgram = np.array([i.days + i.seconds / (24 * 60 * 60)
                                     for i in (self.sTimes - self.mTimes)],
                                    dtype=np.float32)

        with h5py.File(self.file, 'r') as f:
            self.dropIfgram = f['dropIfgram'][:]
            self.pbaseIfgram = f['bperp'][:]

            # get existed datasetNames in the order of IFGRAM_DSET_NAMES
            dsNames = [i for i in f.keys()
                       if (isinstance(f[i], h5py.Dataset)
                           and f[i].shape[-2:] == (self.length, self.width))]
            self.datasetNames = [i for i in IFGRAM_DSET_NAMES if i in dsNames]
            self.datasetNames += [i for i in dsNames if i not in IFGRAM_DSET_NAMES]

        # Get sliceList for self.read()
        self.sliceList = []
        for dsName in self.datasetNames:
            self.sliceList += [f'{dsName}-{i}' for i in self.date12List]

        # Time in timeseries domain
        self.dateList = self.get_date_list(dropIfgram=False)
        self.numDate = len(self.dateList)

        # Reference pixel
        try:
            self.refY = int(self.metadata['REF_Y'])
            self.refX = int(self.metadata['REF_X'])
        except:
            self.refY = None
            self.refX = None
        try:
            self.refLat = float(self.metadata['REF_LAT'])
            self.refLon = float(self.metadata['REF_LON'])
        except:
            self.refLat = None
            self.refLon = None

    def get_metadata(self):
        # read metadata from root level
        with h5py.File(self.file, 'r') as f:
            self.metadata = dict(f.attrs)
            dates = f['date'][:].flatten()

        # decode metadata
        for key, value in self.metadata.items():
            try:
                self.metadata[key] = value.decode('utf8')
            except:
                self.metadata[key] = value

        # START/END_DATE
        dateList = sorted(i.decode('utf8') for i in dates)
        self.metadata['START_DATE'] = dateList[0]
        self.metadata['END_DATE'] = dateList[-1]
        return self.metadata

    def get_size(self, dropIfgram=False, datasetName=None):
        with h5py.File(self.file, 'r') as f:
            # get default datasetName
            if datasetName is None:
                datasetName = [i for i in ['unwrapPhase', 'rangeOffset', 'azimuthOffset'] if i in f.keys()][0]

            # get 3D size
            self.numIfgram, self.length, self.width = f[datasetName].shape

            # update 1st dimension size
            if dropIfgram:
                self.numIfgram = np.sum(f['dropIfgram'][:])
        return self.numIfgram, self.length, self.width

    def read_datetimes(self):
        """Read date1/2 into array of datetime.datetime objects"""
        with h5py.File(self.file, 'r') as f:
            dates = f['date'][:]

        # grab the date string format
        self.dateFormat = ptime.get_date_str_format(dates[0, 0])

        # convert date from str to datetime.datetime objects
        self.mDates = np.array([i.decode('utf8') for i in dates[:, 0]])
        self.sDates = np.array([i.decode('utf8') for i in dates[:, 1]])
        self.mTimes = np.array([dt.datetime.strptime(i, self.dateFormat) for i in self.mDates])
        self.sTimes = np.array([dt.datetime.strptime(i, self.dateFormat) for i in self.sDates])

    def read(self, datasetName='unwrapPhase', box=None, print_msg=True, dropIfgram=False):
        """Read 3D dataset with bounding box in space
        Parameters: datasetName : string, to point to specific 2D dataset, e.g.:
                        unwrapPhase
                        coherence
                        connectComponent
                        ...
                        unwrapPhase-20161020_20161026
                        unwrapPhase-...
                        coherence-20161020_20161026
                        ...
                        ['unwrapPhase-20161020_20161026',
                         'unwrapPhase-20161020_20161101',
                         ...]
                    box : tuple of 4 int, for (x0,y0,x1,y1)
                    print_msg : bool
        Returns: data : 2D or 3D array
        Example:
            obj = ifgramStack('./inputs/ifgramStack.h5')
            obj.read(datasetName='unwrapPhase')
            obj.read(datasetName='coherence')
            obj.read(datasetName='unwrapPhase-20161020_20161026')
            obj.read(datasetName=['unwrapPhase-20161020_20161026',
                                  'unwrapPhase-20161020_20161101'])
        """
        self.get_size(dropIfgram=False)
        date12List = self.get_date12_list(dropIfgram=False)

        # convert input datasetName into list
        if datasetName is None:
            datasetName = ['unwrapPhase']
        elif isinstance(datasetName, str):
            datasetName = [datasetName]

        with h5py.File(self.file, 'r') as f:
            familyName = datasetName[0].split('-')[0]
            ds = f[familyName]
            if print_msg:
                print(f'reading {familyName} data from file: {self.file} ...')

            # get dateFlag - mark in time/1st dimension
            dateFlag = np.zeros((self.numIfgram), dtype=np.bool_)
            datasetName = [i.replace(familyName, '').replace('-', '') for i in datasetName]
            if any(not i for i in datasetName):
                if dropIfgram:
                    dateFlag = f['dropIfgram'][:]
                else:
                    dateFlag[:] = True
            else:
                for e in datasetName:
                    dateFlag[date12List.index(e)] = True

            # get index in space/2-3 dimension
            if box is None:
                box = (0, 0, self.width, self.length)

            # read
            data = ds[:,
                      box[1]:box[3],
                      box[0]:box[2]][dateFlag]

            if any(i == 1 for i in data.shape):
                data = np.squeeze(data)
        return data

    def spatial_average(self, datasetName='coherence', maskFile=None, box=None, useMedian=False,
                        reverseMask=False, threshold=None):
        """ Calculate the spatial average."""
        if datasetName is None:
            datasetName = 'coherence'

        if useMedian:
            print(f'calculating spatial median of {datasetName} in file {self.file} ...')
        else:
            print(f'calculating spatial mean of {datasetName} in file {self.file} ...')

        # read mask
        if maskFile and os.path.isfile(maskFile):
            print('read mask from file: '+maskFile)
            mask = singleDataset(maskFile).read(box=box)
        else:
            maskFile = None

        # calculation
        with h5py.File(self.file, 'r') as f:
            dset = f[datasetName]
            numIfgram = dset.shape[0]
            dmean = np.zeros((numIfgram), dtype=np.float32)

            prog_bar = ptime.progressBar(maxValue=numIfgram)
            for i in range(numIfgram):
                prog_bar.update(i+1, suffix=f'{i+1}/{numIfgram}')

                # read
                data = dset[i, box[1]:box[3], box[0]:box[2]]
                if maskFile:
                    data[mask == int(reverseMask)] = np.nan

                # ignore ZERO value for coherence
                if datasetName == 'coherence':
                    data[data == 0] = np.nan

                # calculate area ratio if threshold is specified
                # percentage of pixels with value above the threshold
                if threshold is not None:
                    data[data > threshold] = 1
                    data[data <= threshold] = 0

                if useMedian:
                    dmean[i] = np.nanmedian(data)
                else:
                    dmean[i] = np.nanmean(data)
            prog_bar.close()
        return dmean, self.date12List

    # Functions considering dropIfgram value
    def get_date12_list(self, dropIfgram=True):
        with h5py.File(self.file, 'r') as f:
            dates = f['date'][:]
            if dropIfgram:
                dates = dates[f['dropIfgram'][:], :]
        mDates = np.array([i.decode('utf8') for i in dates[:, 0]])
        sDates = np.array([i.decode('utf8') for i in dates[:, 1]])
        date12List = [f'{i}_{j}' for i, j in zip(mDates, sDates)]
        return date12List

    def get_drop_date12_list(self):
        with h5py.File(self.file, 'r') as f:
            dates = f['date'][:]
            dates = dates[~f['dropIfgram'][:], :]
        mDates = np.array([i.decode('utf8') for i in dates[:, 0]])
        sDates = np.array([i.decode('utf8') for i in dates[:, 1]])
        date12List = [f'{i}_{j}' for i, j in zip(mDates, sDates)]
        return date12List

    def get_date_list(self, dropIfgram=False):
        with h5py.File(self.file, 'r') as f:
            dates = f['date'][:]
            if dropIfgram:
                dates = dates[f['dropIfgram'][:], :]
        mDates = [i.decode('utf8') for i in dates[:, 0]]
        sDates = [i.decode('utf8') for i in dates[:, 1]]
        dateList = sorted(list(set(mDates + sDates)))
        return dateList

    def get_reference_phase(self, unwDatasetName='unwrapPhase', skip_reference=False, dropIfgram=False):
        """Get reference value
        Parameters: unwDatasetName : string, unwrapPhase, or unwrapPhase_unwCor
                    skip_reference : bool, skip reference value (for simulation only)
                    dropIfgram : bool, skip ifgrams marked as dropped or not
        Returns:    ref_phase : 1D np.array in size of (num_ifgram,) in float32
        """
        self.open(print_msg=False)
        if skip_reference:
            ref_phase = np.zeros(self.get_size(dropIfgram=dropIfgram)[0], np.float32)
            print('skip checking reference pixel info - This is for offset and testing ONLY.')
        elif 'REF_Y' not in self.metadata.keys():
            raise ValueError('No REF_X/Y found!\nrun reference_point.py to select reference pixel.')
        else:
            print(f'reference pixel in y/x: ({self.refY}, {self.refX}) from dataset: {unwDatasetName}')
            ref_phase = self.read(datasetName=unwDatasetName,
                                  box=(self.refX, self.refY, self.refX+1, self.refY+1),
                                  dropIfgram=dropIfgram,
                                  print_msg=False)
        return ref_phase

    def nonzero_mask(self, datasetName=None, print_msg=True, dropIfgram=True):
        """Return the common mask of pixels with non-zero value in dataset of all ifgrams.
           Ignoring dropped ifgrams
        """
        self.open(print_msg=False)
        with h5py.File(self.file, 'r') as f:
            if datasetName is None:
                datasetName = [i for i in ['connectComponent', 'unwrapPhase']
                               if i in f.keys()][0]
            print(f'calculate the common mask of pixels with non-zero {datasetName} value')

            dset = f[datasetName]
            mask = np.ones(dset.shape[1:3], dtype=np.bool_)
            dropIfgramFlag = np.ones(dset.shape[0], dtype=np.bool_)
            if dropIfgram:
                dropIfgramFlag = self.dropIfgram
            num2read = np.sum(dropIfgramFlag)
            idx2read = np.where(dropIfgramFlag)[0]

            # Loop to save memory usage
            prog_bar = ptime.progressBar(maxValue=num2read)
            for i in range(num2read):
                prog_bar.update(i+1, suffix=f'{i+1}/{num2read}')
                data = dset[idx2read[i], :, :]
                mask[data == 0.] = 0
                mask[np.isnan(data)] = 0
            prog_bar.close()
        return mask

    def temporal_average(self, datasetName='coherence', dropIfgram=True, max_memory=4):
        self.open(print_msg=False)
        if datasetName is None:
            datasetName = 'coherence'
        print(f'calculate the temporal average of {datasetName} in file {self.file} ...')

        # index of pairs to read
        ifgram_flag = np.ones(self.numIfgram, dtype=np.bool_)
        if dropIfgram:
            ifgram_flag = self.dropIfgram
            if np.all(ifgram_flag == 0.):
                raise Exception('ALL interferograms are marked as dropped, '
                                'can not calculate temporal average.')

        # temporal baseline for phase
        # with unit of years (float64 for very short tbase of UAVSAR data)
        if 'unwrapPhase' in datasetName:
            phase2range = -1 * float(self.metadata['WAVELENGTH']) / (4.0 * np.pi)
            tbase = np.array(self.tbaseIfgram, dtype=np.float64) / 365.25
            tbase = tbase[ifgram_flag]

        with h5py.File(self.file, 'r') as f:
            dset = f[datasetName]

            # reference value for phase
            ref_val = None
            if ('unwrapPhase' in datasetName
                   and self.refY is not None and 0 <= self.refY <= self.width
                   and self.refX is not None and 0 <= self.refX <= self.length):
                ref_val = dset[:, self.refY, self.refX][ifgram_flag]

            # get step size and number
            ds_size = np.sum(ifgram_flag, dtype=np.int64) * self.length * self.width * 4
            num_step = int(np.ceil(ds_size * 3 / (max_memory * 1024**3)))
            row_step = int(np.rint(self.length / num_step / 10) * 10)
            num_step = int(np.ceil(self.length / row_step))

            # calculate lines by lines
            dmean = np.zeros(dset.shape[1:3], dtype=np.float32)
            prog_bar = ptime.progressBar(maxValue=num_step)
            for i in range(num_step):
                r0 = i * row_step
                r1 = min(r0 + row_step, self.length)
                prog_bar.update(i+1, suffix=f'lines {r1}/{self.length}')

                # read
                data = dset[:, r0:r1, :][ifgram_flag]

                # referencing / normalizing for phase
                if 'unwrapPhase' in datasetName:
                    # spatial referencing
                    if ref_val is not None:
                        data -= np.tile(ref_val.reshape(-1, 1, 1), (1, data.shape[1], data.shape[2]))
                    # phase to phase velocity
                    for j in range(data.shape[0]):
                        data[j,:,:] *= (phase2range / tbase[j])

                # use nanmean to better handle NaN values
                dmean[r0:r1, :] = np.nanmean(data, axis=0)
            prog_bar.close()
        return dmean

    def get_max_connection_number(self):
        date12_list = self.get_date12_list()
        A = self.get_design_matrix4timeseries(date12_list, refDate=0)[0]
        num_conn = np.zeros(A.shape[0], dtype=np.int16)
        for i in range(A.shape[0]):
            Ai = A[i, :]
            num_conn[i] = np.where(Ai == 1)[0] - np.where(Ai == -1)[0]
        return np.max(num_conn)


    def split2boxes(self, max_memory=4, dim0_size=None, print_msg=True):
        """Split into chunks in rows to reduce memory usage.

        Parameters: max_memory - float, max memory to use in GB
                    dim0_size  - the 1st dimension size of all used datasets
                                 e.g., dim0_size = num_pair * 2 + num_date
                    print_msg  - bool
        Returns:    box_list   - list of tuple of 4 int
                    num_box    - int, number of boxes
        """
        self.open(print_msg=False)
        length = self.length
        width = self.width

        # dimension in time: phase/offset, weight, timeseries, etc.
        if not dim0_size:
            # for time series estimation
            dim0_size = self.numIfgram * 2 + self.numDate
        ds_size = dim0_size * length * width * 4

        num_box = int(np.ceil(ds_size * 1.5 / (max_memory * 1024**3)))
        y_step = int(np.ceil((length / num_box) / 10) * 10)
        num_box = int(np.ceil(length / y_step))
        if print_msg and num_box > 1:
            print('maximum memory size: %.1E GB' % max_memory)
            print('split %d lines into %d patches for processing' % (length, num_box))
            print('    with each patch up to %d lines' % y_step)

        # y_step / num_box --> box_list
        box_list = []
        for i in range(num_box):
            y0 = i * y_step
            y1 = min([length, y0 + y_step])
            box = (0, y0, width, y1)
            box_list.append(box)

        return box_list, num_box


    # Functions for closure phase bias
    def get_closure_phase_index(self, conn, dropIfgram=True):
        """Get the indices of interferograms that forms the given connection level closure loop.

        Parameters: conn       - int, connection level
                    dropIfgram - bool, exclude the dropped interferograms.
        Returns:    cp_idx     - 2D np.ndarray in int16 in size of (num_cp, conn + 1)
                                 Each row for the indices of interferograms for one closure loop.
                                 num_cp <= num_date - conn
        """
        date12_list = self.get_date12_list(dropIfgram=False)
        date_list = self.get_date_list(dropIfgram=dropIfgram)
        num_date = len(date_list)

        # get the closure index
        cp_idx = []
        for i in range(num_date - conn):
            # compose the connection-n pairs
            cp_date12_list = []
            for j in range(conn):
                cp_date12_list.append(f'{date_list[i+j]}_{date_list[i+j+1]}')
            cp_date12_list.append(f'{date_list[i]}_{date_list[i+conn]}')

            # add to cp_idx, ONLY IF all pairs exist for this closure loop
            if all(x in date12_list for x in cp_date12_list):
                cp_idx.append([date12_list.index(x) for x in cp_date12_list])

        # list(list) to 2D array
        cp_idx = np.array(cp_idx, dtype=np.int16)
        cp_idx = np.unique(cp_idx, axis=0)

        return cp_idx


    def get_sequential_closure_phase(self, box, conn, post_proc=None):
        """Computes wrapped sequential closure phases for a given connection level.

        Reference: Equation (21) in Zheng et al. (2022, TGRS)
        For conn = 5, seq_closure_phase = p12 + p23 + p34 + p45 + p56 - p16.

        Parameters: box       - tuple of 4 int, bounding box in (x0, y0, x1, y1)
                    conn      - int, connection level of the closure phase
                    post_proc - str, post processing of the closure phase:
                                None - 3D array in float32, seq closure phase
                                sum  - 2D array in complex64, sum  in time of the complex seq closure phase
                                mean - 2D array in complex64, mean in time of the complex seq closure phase
        Returns:    cp_w      - 3D np.ndarray in float32 in size of (num_cp, box_len, box_wid)
                                wrapped sequential  closure phase for the given connection level.
                    sum_cp    - None or 2D np.ndarray in complex64 in size of (box_len, box_width)
                                wrapped average seq closure phase for the given connection level,
                                controlled by post_proc.
                    num_cp    - int, number of  seq closure phase for the given connection level.
        """
        # basic info
        num_date = len(self.get_date_list(dropIfgram=True))
        box_wid = box[2] - box[0]
        box_len = box[3] - box[1]

        ## get the closure index
        cp_idx = self.get_closure_phase_index(conn=conn, dropIfgram=True)
        num_cp = cp_idx.shape[0]
        print(f'number of closure measurements expected: {num_date - conn}')
        print(f'number of closure measurements found   : {num_cp}')

        if not post_proc:
            if num_cp < num_date - conn:
                msg = f'num_cp ({num_cp}) < num_date - conn ({num_date - conn})'
                msg += ' --> some interferograms are missing!'
                raise Exception(msg)
        else:
            if num_cp < 1:
                raise Exception(f"No triplets found at connection level: {conn}!")

        ## read data
        # Note by Yujie Zheng on Dec 2, 2022: I removed the lines for spatial referencing,
        # because no moisture change should have a naturally zero closure phase. On the other
        # hand, choosing a reference point in the wrong place can lead to misinterpretations.
        # Tests in both Central Valley and Bristol Dry lakes confirms this.
        ds_name = 'wrapPhase' if 'wrapPhase' in self.datasetNames else 'unwrapPhase'
        print(f'reading {ds_name} to compute closure phases')
        phase = self.read(datasetName=ds_name, box=box, print_msg=False)

        # apply spatial referencing (for ARIA only)
        # to avoid the abnormal result as shown in https://github.com/insarlab/MintPy/pull/1063
        # This may be caused by the phase stitching during product preparation via ARIA-tools,
        # which could have broken the temporal consistency of the native unwrapped phase.
        processor = self.metadata.get('mintpy.load.processor', 'isce')
        if ds_name == 'unwrapPhase' and processor in ['aria']:
            print(f'apply spatial referencing to {processor} products')
            ref_phase = self.get_reference_phase(dropIfgram=False)
            for i in range(phase.shape[0]):
                mask = phase[i] != 0.
                phase[i][mask] -= ref_phase[i]

        ## calculate the 3D complex seq closure phase
        cp_w = np.zeros((num_cp, box_len, box_wid), dtype=np.complex64)
        for i in range(num_cp):

            # calculate (un)wrapped closure phase
            idx_plus, idx_minor = cp_idx[i, :-1], cp_idx[i, -1]
            cp0_w = np.sum(phase[idx_plus], axis=0) - phase[idx_minor]

            # re-wrapping to ensure the wrapped closure phase
            cp_w[i] = np.exp(1j * cp0_w)

        ## post-processing
        if not post_proc:
            sum_cp = None

        elif post_proc == 'sum':
            sum_cp = np.sum(cp_w, axis=0)

        elif post_proc == 'mean':
            sum_cp = np.mean(cp_w, axis=0)

        else:
            raise ValueError(f'un-recognized post_proc={post_proc}! Available choices: sum, mean.')

        return np.angle(cp_w), sum_cp, num_cp


    # Functions for unwrapping error correction
    @staticmethod
    def get_design_matrix4triplet(date12_list):
        """Generate the design matrix of ifgram triangle for unwrap error correction using phase closure

        Parameters: date12_list : list of string in YYYYMMDD_YYYYMMDD format
        Returns:    C : 2D np.array in size of (num_tri, num_ifgram) consisting 0, 1, -1
                        for 3 SAR acquisition in t1, t2 and t3 in time order,
                        ifg1 for (t1, t2) with 1
                        ifg2 for (t1, t3) with -1
                        ifg3 for (t2, t3) with 1
        Examples:   obj = ifgramStack('./inputs/ifgramStack.h5')
                    date12_list = obj.get_date12_list(dropIfgram=True)
                    C = ifgramStack.get_design_matrix4triplet(date12_list)
        """
        # Date info
        date12_list = list(date12_list)
        # Use tuples of the dates, which are hashable for the lookup dict
        date12_tuples = [tuple(d.split('_')) for d in date12_list]

        # Create an inverse map from tuple(date1, date1) -> index in ifg list
        ifg_to_idx = {ifg: idx for idx, ifg in enumerate(date12_tuples)}

        # Get the unique SAR dates present in the interferogram list
        date_list = sorted(set(itertools.chain.from_iterable(date12_tuples)))

        # Start with all possible triplets, narrow down based on ifgs present
        closure_list = itertools.combinations(date_list, 3)

        M = len(date12_tuples)  # Number of igrams, number of rows
        C_list = []
        for date1, date2, date3 in closure_list:
            ifg12 = (date1, date2)
            ifg23 = (date2, date3)
            ifg13 = (date1, date3)
            # Check if any ifg is not available in the current triple. Skip if so
            try:
                idx12 = ifg_to_idx[ifg12]
                idx23 = ifg_to_idx[ifg23]
                idx13 = ifg_to_idx[ifg13]
            except KeyError:
                continue

            # Add the +/-1 row of the matrix to our list
            row = np.zeros(M, dtype=np.int8)
            row[idx12] = 1
            row[idx23] = 1
            row[idx13] = -1
            C_list.append(row)

        if len(C_list) == 0:
            print(f'\nWARNING: No triangles found from input date12_list:\n{date12_list}!\n')
            return None

        return np.stack(C_list).astype(np.float32)


    # Functions for network inversion / time series estimation
    @staticmethod
    def get_design_matrix4timeseries(date12_list, refDate=None):
        """Return design matrix of the input ifgramStack for timeseries estimation

        Parameters: date12_list - list of string in YYYYMMDD_YYYYMMDD format
                    refDate     - str, date in YYYYMMDD format
                                  set to None for the 1st date
                                  set to 'no' to disable reference date
        Returns:    A - 2D array of float32 in size of (num_ifgram, num_date-1)
                    B - 2D array of float32 in size of (num_ifgram, num_date-1)
        Examples:   obj = ifgramStack('./inputs/ifgramStack.h5')
                    A, B = obj.get_design_matrix4timeseries(obj.get_date12_list(dropIfgram=True))
                    A = ifgramStack.get_design_matrix4timeseries(date12_list, refDate='20101022')[0]
                    A = ifgramStack.get_design_matrix4timeseries(date12_list, refDate=0)[0] #do not omit the 1st column
        """
        # Date info
        date12_list = list(date12_list)
        date1s = [i.split('_')[0] for i in date12_list]
        date2s = [i.split('_')[1] for i in date12_list]
        date_list = sorted(list(set(date1s + date2s)))
        num_ifgram = len(date12_list)
        num_date = len(date_list)

        # tbase in the unit of years
        date_format = ptime.get_date_str_format(date_list[0])
        dates = np.array([dt.datetime.strptime(i, date_format) for i in date_list])
        tbase = [i.days + i.seconds / (24 * 60 * 60) for i in (dates - dates[0])]
        tbase = np.array(tbase, dtype=np.float32) / 365.25

        # calculate design matrix
        # A for minimizing the residual of phase
        # B for minimizing the residual of phase velocity
        A = np.zeros((num_ifgram, num_date), np.float32)
        B = np.zeros((num_ifgram, num_date), np.float32)
        for i in range(num_ifgram):
            ind1, ind2 = (date_list.index(d) for d in date12_list[i].split('_'))
            A[i, ind1] = -1
            A[i, ind2] = 1
            # support date12_list with the first date NOT being the earlier date
            if ind1 < ind2:
                B[i, ind1:ind2] = tbase[ind1 + 1:ind2 + 1] - tbase[ind1:ind2]
            else:
                B[i, ind2:ind1] = tbase[ind2:ind1] - tbase[ind2 + 1:ind1 + 1]

        # Remove reference date as it can not be resolved
        if refDate != 'no':
            # default refDate
            if refDate is None:
                # for single   reference network, use the same reference date
                # for multiple reference network, use the first date
                if len(set(date1s)) == 1:
                    refDate = date1s[0]
                else:
                    refDate = date_list[0]

            # apply refDate
            if refDate:
                ind_r = date_list.index(refDate)
                A = np.hstack((A[:, 0:ind_r], A[:, (ind_r+1):]))
                B = B[:, :-1]

        return A, B


    def get_perp_baseline_timeseries(self, dropIfgram=True):
        """Get spatial perpendicular baseline in timeseries from ifgramStack, ignoring dropped ifgrams"""
        # read pbase of interferograms
        with h5py.File(self.file, 'r') as f:
            pbaseIfgram = f['bperp'][:]
            if dropIfgram:
                pbaseIfgram = pbaseIfgram[f['dropIfgram'][:]]

        # estimate pbase of time-series
        date12List = self.get_date12_list(dropIfgram=dropIfgram)
        A = self.get_design_matrix4timeseries(date12List)[0]
        pbaseTimeseries = np.zeros(A.shape[1]+1, dtype=np.float32)
        pbaseTimeseries[1:] = np.linalg.lstsq(A, pbaseIfgram, rcond=None)[0]
        return pbaseTimeseries

    def update_drop_ifgram(self, date12List2Drop):
        """Update dropIfgram dataset based on input date12List2Drop"""
        if date12List2Drop is None:
            return
        date12ListAll = self.get_date12_list(dropIfgram=False)

        # check date12 list to drop: old vs new
        date12ListKeptOld = self.get_date12_list(dropIfgram=True)
        date12List2DropOld = sorted(list(set(date12ListAll) - set(date12ListKeptOld)))
        if date12List2DropOld == date12List2Drop:
            print('The same date12List2Drop / dropIfgram is already marked in the file, skip updating dropIfgram.')
            return

        with h5py.File(self.file, 'r+') as f:
            print(f'open file {self.file} with r+ mode')
            print('update HDF5 dataset "/dropIfgram".')
            f['dropIfgram'][:] = np.array([i not in date12List2Drop for i in date12ListAll], dtype=np.bool_)

            # update MODIFICATION_TIME for all datasets in IFGRAM_DSET_NAMES
            for dsName in IFGRAM_DSET_NAMES:
                if dsName in f.keys():
                    print(f'update MODIFICATION_TIME in HDF5 dataset "/{dsName}"')
                    f[dsName].attrs['MODIFICATION_TIME'] = str(time.time())
                    time.sleep(1)   #to distinguish the modification time of input files

################################# ifgramStack class end ################################



########################################################################################
class singleDataset:
    def __init__(self, file=None):
        self.file = file

    def read(self, box=None):
        with h5py.File(self.file, 'r') as f:
            dsName = list(f.keys())[0]
            data = f[dsName][:]

        if box is not None:
            data = data[box[1]:box[3],
                        box[0]:box[2]]
        return data


################################# HDF-EOS5 class begin #################################
class HDFEOS:
    """
    Time-series object in HDF-EOS5 format for Univ of Miami's InSAR Time-series Web Viewer
        Link: http://insarmaps.miami.edu
    It contains a "timeseries" group and three datasets: date, bperp and timeseries.

    File structure: https://mintpy.readthedocs.io/en/latest/hdfeos5/#file_structure
    """

    def __init__(self, file=None):
        self.file = file
        self.name = 'HDFEOS'
        self.datasetGroupNameDict = {'displacement'       : 'observation',
                                     'raw'                : 'observation',
                                     'troposphericDelay'  : 'observation',
                                     'topographicResidual': 'observation',
                                     'ramp'               : 'observation',
                                     'date'               : 'observation',
                                     'temporalCoherence'  : 'quality',
                                     'mask'               : 'quality',
                                     'coherence'          : 'quality',
                                     'variance'           : 'quality',
                                     'uncertainty'        : 'quality',
                                     'height'             : 'geometry',
                                     'incidenceAngle'     : 'geometry',
                                     'slantRangeDistance' : 'geometry',
                                     'azimuthAngle'       : 'geometry',
                                     'shadowMask'         : 'geometry',
                                     'waterMask'          : 'geometry',
                                     'bperp'              : 'geometry'
                                    }

    def close(self, print_msg=True):
        try:
            self.f.close()
            if print_msg:
                print(f'close timeseries file: {os.path.basename(self.file)}')
        except:
            pass

    def open(self, print_msg=True):
        if print_msg:
            print(f'open {self.name} file: {os.path.basename(self.file)}')
        self.get_metadata()
        self.length = int(self.metadata['LENGTH'])
        self.width = int(self.metadata['WIDTH'])

        self.sliceList = []
        with h5py.File(self.file, 'r') as f:
            gname = 'HDFEOS/GRIDS/timeseries/observation'
            g = f[gname]
            self.dateList = [i.decode('utf8') for i in g['date'][:]]
            self.pbase = g['bperp'][:]
            self.numDate = len(self.dateList)

            # get slice list
            self.sliceList += [f'{gname}/displacement-{i}' for i in self.dateList]
            for gname in ['HDFEOS/GRIDS/timeseries/quality',
                          'HDFEOS/GRIDS/timeseries/geometry']:
                g = f[gname]
                for key in g.keys():
                    if isinstance(g[key], h5py.Dataset) and len(g[key].shape) == 2:
                        self.sliceList.append(f'{gname}/{key}')

    def get_metadata(self):
        with h5py.File(self.file, 'r') as f:
            self.metadata = dict(f.attrs)
            dates = f['HDFEOS/GRIDS/timeseries/observation/date'][:]
        for key, value in self.metadata.items():
            try:
                self.metadata[key] = value.decode('utf8')
            except:
                self.metadata[key] = value
        self.metadata['FILE_TYPE'] = self.name

        # ref_date/index
        dateList = [i.decode('utf8') for i in dates]
        if 'REF_DATE' not in self.metadata.keys():
            self.metadata['REF_DATE'] = dateList[0]
        self.refIndex = dateList.index(self.metadata['REF_DATE'])
        return self.metadata

    def get_date_list(self):
        with h5py.File(self.file, 'r') as f:
            g = f['HDFEOS/GRIDS/timeseries/observation']
            self.dateList = [i.decode('utf8') for i in g['date'][:]]
        return self.dateList

    def read(self, datasetName=None, box=None, print_msg=True):
        """Read dataset from HDF-EOS5 file
        Parameters: self : HDFEOS object
                    datasetName : (list of) str
                    box : tuple of 4 int, for (x0, y0, x1, y1)
                    print_msg : bool
        Returns:    data: 2D or 3D array
        Example:    obj = HDFEOS('S1_IW1_128_0593_0597_20141213_20171221.he5')
                    obj.read(datasetName='displacement')
                    obj.read(datasetName='displacement-20150915')
                    obj.read(datasetName=['displacement-20150915',
                                          'displacement-20150921'])
                    obj.read(datasetName='incidenceAngle')
        """
        self.open(print_msg=False)
        if box is None:
            box = [0, 0, self.width, self.length]

        if datasetName is None:
            datasetName = [TIMESERIES_DSET_NAMES[-1]]
        elif isinstance(datasetName, str):
            datasetName = [datasetName]

        with h5py.File(self.file, 'r') as f:
            familyName = datasetName[0].split('-')[0]
            groupName = self.datasetGroupNameDict[familyName]
            ds = f[f'HDFEOS/GRIDS/timeseries/{groupName}/{familyName}']
            if print_msg:
                print(f'reading {familyName} data from file: {self.file} ...')

            if len(ds.shape) == 1:
                data = ds[:]
            elif len(ds.shape) == 2:
                data = ds[box[1]:box[3], box[0]:box[2]]
            else:
                # get dateFlag - mark in time/1st dimension
                dateFlag = np.zeros((ds.shape[0]), dtype=np.bool_)
                datasetName = [i.replace(familyName, '').replace('-', '') for i in datasetName]
                if any(not i for i in datasetName):
                    dateFlag[:] = True
                else:
                    for e in datasetName:
                        dateFlag[self.dateList.index(e)] = True

                # read
                data = ds[:,
                          box[1]:box[3],
                          box[0]:box[2]][dateFlag]

                # squeeze/shrink dimension whenever it is possible
                if any(i == 1 for i in data.shape):
                    data = np.squeeze(data)
        return data
################################# HDF-EOS5 class end ###################################
