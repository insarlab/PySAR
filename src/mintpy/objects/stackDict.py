"""Classes for HDF5/MintPy file creation / writing."""
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Heresh Fattahi, Zhang Yunjun, 2017               #
############################################################
# Recommend import:
#   from mintpy.objects.stackDict import (
#       geometryDict,
#       ifgramStackDict,
#       ifgramDict,
#   )


import os
import time
import warnings

import h5py
import numpy as np
from skimage.transform import resize

from mintpy.multilook import multilook_data
from mintpy.objects import (
    DATA_TYPE_DICT,
    GEOMETRY_DSET_NAMES,
    IFGRAM_DSET_NAMES,
    TIMESERIES_DSET_NAMES,
)
from mintpy.utils import attribute as attr, ptime, readfile, utils0 as ut


########################################################################################
class ifgramStackDict:
    """
    IfgramStack object for a set of InSAR pairs from the same platform and track.

    Example:
        from mintpy.objects.insarobj import ifgramStackDict
        pairsDict = {('20160524','20160530'):ifgramObj1,
                     ('20160524','20160605'):ifgramObj2,
                     ('20160524','20160611'):ifgramObj3,
                     ('20160530','20160605'):ifgramObj4,
                     ...
                     }
        stackObj = ifgramStackDict(pairsDict=pairsDict)
        stackObj.write2hdf5(outputFile='ifgramStack.h5', box=(200,500,300,600))
    """

    def __init__(self, name='ifgramStack', pairsDict=None, dsName0=IFGRAM_DSET_NAMES[0]):
        self.name = name
        self.pairsDict = pairsDict
        self.dsName0 = dsName0        #reference dataset name, unwrapPhase OR azimuthOffset OR rangeOffset

    def get_size(self, box=None, xstep=1, ystep=1, geom_obj=None):
        """Get size in 3D"""
        num_ifgram = len(self.pairsDict)
        ifgramObj = [v for v in self.pairsDict.values()][0]
        length, width = ifgramObj.get_size(family=self.dsName0)

        # use the reference geometry obj size
        # for low-reso ionosphere from isce2/topsStack
        if geom_obj:
            length, width = geom_obj.get_size()

        # update due to subset
        if box:
            length, width = box[3] - box[1], box[2] - box[0]

        # update due to multilook
        length = length // ystep
        width = width // xstep

        return num_ifgram, length, width

    def get_date12_list(self):
        pairs = [pair for pair in self.pairsDict.keys()]
        self.date12List = [f'{i[0]}_{i[1]}' for i in pairs]
        return self.date12List

    def get_dataset_list(self):
        ifgramObj = [x for x in self.pairsDict.values()][0]
        dsetList = [x for x in ifgramObj.datasetDict.keys()]
        return dsetList

    def get_metadata(self):
        ifgramObj = [v for v in self.pairsDict.values()][0]
        self.metadata = ifgramObj.get_metadata(family=self.dsName0)
        if 'UNIT' in self.metadata.keys():
            self.metadata.pop('UNIT')
        return self.metadata

    def get_dataset_data_type(self, dsName):
        ifgramObj = [v for v in self.pairsDict.values()][0]
        dsFile = ifgramObj.datasetDict[dsName]
        metadata = readfile.read_attribute(dsFile)
        dataType = DATA_TYPE_DICT[metadata.get('DATA_TYPE', 'float32').lower()]
        return dataType

    def write2hdf5(self, outputFile='ifgramStack.h5', access_mode='w', box=None, xstep=1, ystep=1, mli_method='nearest',
                   compression=None, extra_metadata=None, geom_obj=None):
        """Save/write an ifgramStackDict object into an HDF5 file with the structure defined in:

        https://mintpy.readthedocs.io/en/latest/api/data_structure/#ifgramstack

        Parameters: outputFile     - str, Name of the HDF5 file for the InSAR stack
                    access_mode    - str, access mode of output File, e.g. w, r+
                    box            - tuple, subset range in (x0, y0, x1, y1)
                    x/ystep        - int, multilook number in x/y direction
                    mli_method     - str, multilook method, nearest, mean or median
                    compression    - str, HDF5 dataset compression method, None, lzf or gzip
                    extra_metadata - dict, extra metadata to be added into output file
                    geom_obj       - geometryDict object, size reference to determine the resizing operation.
        Returns:    outputFile     - str, Name of the HDF5 file for the InSAR stack
        """
        print('-'*50)

        # output directory
        output_dir = os.path.dirname(outputFile)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            print(f'create directory: {output_dir}')

        self.pairs = sorted(pair for pair in self.pairsDict.keys())
        self.dsNames = list(self.pairsDict[self.pairs[0]].datasetDict.keys())
        self.dsNames = [i for i in IFGRAM_DSET_NAMES if i in self.dsNames]
        maxDigit = max(len(i) for i in self.dsNames)
        numIfgram, length, width = self.get_size(
            box=box,
            xstep=xstep,
            ystep=ystep)

        # check if resize is needed for a lower resolution stack, e.g. ionosphere from isce2/topsStack
        resize2shape = None
        if geom_obj and os.path.basename(outputFile).startswith('ion'):
            # compare the original data size between ionosphere and geometry w/o subset/multilook
            ion_size = self.get_size()[1:]
            geom_size = geom_obj.get_size()
            if ion_size != geom_size:
                msg = 'lower resolution ionosphere file detected'
                msg += f' --> resize from {ion_size} to {geom_size} via skimage.transform.resize ...'
                print(msg)

                # matrix shape for the original geometry size w/o subset/multilook
                resize2shape = geom_size
                # data size of the output HDF5 file w/ resize/subset/multilook
                length, width = self.get_size(
                    box=box,
                    xstep=xstep,
                    ystep=ystep,
                    geom_obj=geom_obj)[1:]

        # write HDF5 file
        with h5py.File(outputFile, access_mode) as f:
            print(f'create HDF5 file {outputFile} with {access_mode} mode')

            ###############################
            # 3D datasets containing unwrapPhase, magnitude, coherence, connectComponent, wrapPhase, etc.
            for dsName in self.dsNames:
                dsShape = (numIfgram, length, width)
                dsDataType = np.float32
                dsCompression = compression
                if dsName in ['connectComponent']:
                    dsDataType = np.int16
                    dsCompression = 'lzf'
                    mli_method = 'nearest'

                print(('create dataset /{d:<{w}} of {t:<25} in size of {s}'
                       ' with compression = {c}').format(d=dsName,
                                                         w=maxDigit,
                                                         t=str(dsDataType),
                                                         s=dsShape,
                                                         c=dsCompression))
                ds = f.create_dataset(dsName,
                                      shape=dsShape,
                                      maxshape=(None, dsShape[1], dsShape[2]),
                                      dtype=dsDataType,
                                      chunks=True,
                                      compression=dsCompression)

                # set no-data value - printout msg
                if dsName.endswith('OffsetVar'):
                    print(f'set no-data value for {dsName} from 99 to NaN.')
                    dsFile = self.pairsDict[self.pairs[0]].datasetDict[dsName]
                    if dsFile.endswith('cov.bip'):
                        print('convert variance to standard deviation.')

                # msg
                if xstep * ystep > 1:
                    print(f'apply {xstep} x {ystep} multilooking/downsampling via {mli_method} ...')

                prog_bar = ptime.progressBar(maxValue=numIfgram)
                for i, pair in enumerate(self.pairs):
                    prog_bar.update(i+1, suffix=f'{pair[0]}_{pair[1]}')

                    # read and/or resize
                    ifgramObj = self.pairsDict[pair]
                    data = ifgramObj.read(dsName,
                                          box=box,
                                          xstep=xstep,
                                          ystep=ystep,
                                          mli_method=mli_method,
                                          resize2shape=resize2shape)[0]

                    # special handling for offset covariance file
                    if dsName.endswith('OffsetStd'):
                        # set no-data value to np.nan
                        data[data == 99.] = np.nan

                        # convert variance to std. dev.
                        dsFile = ifgramObj.datasetDict[dsName]
                        if dsFile.endswith('cov.bip'):
                            data = np.sqrt(data)

                    # write
                    ds[i, :, :] = data

                ds.attrs['MODIFICATION_TIME'] = str(time.time())
                prog_bar.close()

            ###############################
            # 2D dataset containing reference and secondary dates of all pairs
            dsName = 'date'
            dsDataType = np.bytes_
            dsShape = (numIfgram, 2)
            print('create dataset /{d:<{w}} of {t:<25} in size of {s}'.format(d=dsName,
                                                                              w=maxDigit,
                                                                              t=str(dsDataType),
                                                                              s=dsShape))
            data = np.array(self.pairs, dtype=dsDataType)
            f.create_dataset(dsName, data=data)

            ###############################
            # 1D dataset containing perpendicular baseline of all pairs
            dsName = 'bperp'
            dsDataType = np.float32
            dsShape = (numIfgram,)
            print('create dataset /{d:<{w}} of {t:<25} in size of {s}'.format(d=dsName,
                                                                              w=maxDigit,
                                                                              t=str(dsDataType),
                                                                              s=dsShape))
            # get bperp
            data = np.zeros(numIfgram, dtype=dsDataType)
            for i in range(numIfgram):
                ifgramObj = self.pairsDict[self.pairs[i]]
                data[i] = ifgramObj.get_perp_baseline(family=self.dsName0)
            # write
            f.create_dataset(dsName, data=data)

            ###############################
            # 1D dataset containing bool value of dropping the interferograms or not
            dsName = 'dropIfgram'
            dsDataType = np.bool_
            dsShape = (numIfgram,)
            print('create dataset /{d:<{w}} of {t:<25} in size of {s}'.format(d=dsName,
                                                                              w=maxDigit,
                                                                              t=str(dsDataType),
                                                                              s=dsShape))
            data = np.ones(dsShape, dtype=dsDataType)
            f.create_dataset(dsName, data=data)

            ###############################
            # Attributes
            # read metadata from original data file w/o resize/subset/multilook
            meta = self.get_metadata()
            if extra_metadata:
                meta.update(extra_metadata)
                print(f'add extra metadata: {extra_metadata}')

            # update metadata due to resize
            # for low resolution ionosphere from isce2/topsStack
            if resize2shape:
                print('update metadata due to resize')
                meta = attr.update_attribute4resize(meta, resize2shape)

            # update metadata due to subset
            if box:
                print('update metadata due to subset')
                meta = attr.update_attribute4subset(meta, box)

            # update metadata due to multilook
            if xstep * ystep > 1:
                print('update metadata due to multilook')
                meta = attr.update_attribute4multilook(meta, ystep, xstep)

            # write metadata to HDF5 file at the root level
            meta['FILE_TYPE'] = self.name
            for key, value in meta.items():
                f.attrs[key] = value

        print(f'Finished writing to {outputFile}')
        return outputFile


########################################################################################
class ifgramDict:
    """
    Ifgram object for a single InSAR pair of interferogram. It includes dataset name (family) of:
        'unwrapPhase','coherence','connectComponent','wrapPhase','rangeOffset','azimuthOffset', etc.

    Example:
        from mintpy.objects.insarobj import ifgramDict
        datasetDict = {'unwrapPhase'     :'$PROJECT_DIR/merged/interferograms/20151220_20160206/filt_fine.unw',
                       'coherence'       :'$PROJECT_DIR/merged/interferograms/20151220_20160206/filt_fine.cor',
                       'connectComponent':'$PROJECT_DIR/merged/interferograms/20151220_20160206/filt_fine.unw.conncomp',
                       'wrapPhase'       :'$PROJECT_DIR/merged/interferograms/20151220_20160206/filt_fine.int',
                       'magnitude'       :'$PROJECT_DIR/merged/interferograms/20151220_20160206/filt_fine.unw',
                       ...
                      }
        ifgramObj = ifgramDict(datasetDict=datasetDict)
        data, atr = ifgramObj.read('unwrapPhase')
    """

    def __init__(self, name='ifgram', datasetDict={}, metadata=None):
        self.name = name
        self.datasetDict = datasetDict

        self.platform = None
        self.track = None
        self.processor = None
        # platform, track and processor can get values from metadat if they exist
        if metadata is not None:
            for key, value in metadata.items():
                setattr(self, key, value)

    def read(self, family, box=None, xstep=1, ystep=1, mli_method='nearest', resize2shape=None):
        """Read data for the given dataset name.

        Parameters: self         - ifgramDict object
                    family       - str, dataset name
                    box          -  tuple of 4 int, in (x0, y0, x1, y1) with respect to the full resolution
                    x/ystep      - int, number of pixels to skip, with respect to the full resolution
                    mli_method   - str, interpolation method, nearest, mean, median
                    resize2shape - tuple of 2 int, resize the native matrix to the given shape
                                   Set to None for not resizing
        Returns:    data         - 2D np.ndarray
                    meta         - dict, metadata
        """
        self.file = self.datasetDict[family]
        box2read = None if resize2shape else box

        # 1. read input file
        data, meta = readfile.read(self.file,
                                   datasetName=family,
                                   box=box2read,
                                   xstep=1,
                                   ystep=1)

        # 2. resize
        if resize2shape:
            # link: https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
            data = resize(data,
                          output_shape=resize2shape,
                          order=1,
                          mode='constant',
                          anti_aliasing=True,
                          preserve_range=True)

            # 3. subset by box
            if box:
                data = data[box[1]:box[3],
                            box[0]:box[2]]

        # 4. multilook
        if xstep * ystep > 1:
            if mli_method == 'nearest':
                # multilook - nearest resampling
                # output data size
                xsize = int(data.shape[1] / xstep)
                ysize = int(data.shape[0] / ystep)
                # sampling
                data = data[int(ystep/2)::ystep,
                            int(xstep/2)::xstep]
                data = data[:ysize, :xsize]

            else:
                # multilook - mean or median resampling
                data = multilook_data(data,
                                      lks_y=ystep,
                                      lks_x=xstep,
                                      method=mli_method)

        return data, meta

    def get_size(self, family=IFGRAM_DSET_NAMES[0]):
        self.file = self.datasetDict[family]
        metadata = readfile.read_attribute(self.file)
        length = int(metadata['LENGTH'])
        width = int(metadata['WIDTH'])
        return length, width

    def get_perp_baseline(self, family=IFGRAM_DSET_NAMES[0]):
        self.file = self.datasetDict[family]
        metadata = readfile.read_attribute(self.file)
        self.bperp_top = float(metadata['P_BASELINE_TOP_HDR'])
        self.bperp_bottom = float(metadata['P_BASELINE_BOTTOM_HDR'])
        self.bperp = (self.bperp_top + self.bperp_bottom) / 2.0
        return self.bperp

    def get_metadata(self, family=IFGRAM_DSET_NAMES[0]):
        self.file = self.datasetDict[family]
        self.metadata = readfile.read_attribute(self.file)
        self.length = int(self.metadata['LENGTH'])
        self.width = int(self.metadata['WIDTH'])

        if self.track:
            self.metadata['TRACK'] = self.track

        if self.platform:
            self.metadata['PLATFORM'] = self.platform

        return self.metadata


########################################################################################
class timeseriesDict:
    """
    timeseriesStack object for a set of InSAR acquisitions from the same platform and track.

    Example:
        from mintpy.objects.insarobj import timeseriesDict
        datesDict = {('20160524'):acqObj1,
                     ('20160524'):acqObj2,
                     ('20160524'):acqObj3,
                     ('20160530'):acqObj4,
                     ...
                     }
        tsObj = timeseriesDict(datesDict=datesDict)
        tsObj.write2hdf5(outputFile='timeseries.h5', box=(200,500,300,600))
    """

    def __init__(self, name='timeseries', datesDict=None, dsName0=TIMESERIES_DSET_NAMES[0]):
        self.name = name
        self.datesDict = datesDict
        self.dsName0 = dsName0        #reference dataset name, unwrapPhase OR azimuthOffset OR rangeOffset

    def get_size(self, box=None, xstep=1, ystep=1, geom_obj=None):
        """Get size in 3D"""
        num_date = len(self.datesDict)
        acqObj = [v for v in self.datesDict.values()][0]
        length, width = acqObj.get_size(family=self.dsName0)

        # use the reference geometry obj size
        # for low-reso ionosphere from isce2/topsStack
        if geom_obj:
            length, width = geom_obj.get_size()

        # update due to subset
        if box:
            length, width = box[3] - box[1], box[2] - box[0]

        # update due to multilook
        length = length // ystep
        width = width // xstep

        return num_date, length, width

    def get_date_list(self):
        self.dateList = list(self.datesDict.keys())
        return self.dateList

    def get_dataset_list(self):
        acqObj = [x for x in self.datesDict.values()][0]
        dsetList = [x for x in acqObj.datasetDict.keys()]
        return dsetList

    def get_metadata(self):
        acqObj = [v for v in self.datesDict.values()][0]
        self.metadata = acqObj.get_metadata(family=self.dsName0)
        return self.metadata

    def get_dataset_data_type(self, dsName):
        acqObj = [v for v in self.datesDict.values()][0]
        dsFile = acqObj.datasetDict[dsName]
        metadata = readfile.read_attribute(dsFile)
        dataType = DATA_TYPE_DICT[metadata.get('DATA_TYPE', 'float32').lower()]
        return dataType

    def write2hdf5(self, outputFile='timeseries.h5', access_mode='w', box=None, xstep=1, ystep=1, mli_method='nearest',
                   compression=None, extra_metadata=None, geom_obj=None):
        """Save/write an timeseriesDict object into an HDF5 file with the structure defined in:

        https://mintpy.readthedocs.io/en/latest/api/data_structure/#ifgramstack (Kai need to update the doc?)

        Parameters: outputFile     - str, Name of the HDF5 file for the InSAR stack
                    access_mode    - str, access mode of output File, e.g. w, r+
                    box            - tuple, subset range in (x0, y0, x1, y1)
                    x/ystep        - int, multilook number in x/y direction
                    mli_method     - str, multilook method, nearest, mean or median
                    compression    - str, HDF5 dataset compression method, None, lzf or gzip
                    extra_metadata - dict, extra metadata to be added into output file
                    geom_obj       - geometryDict object, size reference to determine the resizing operation.
        Returns:    outputFile     - str, Name of the HDF5 file for the InSAR stack
        """
        print('-'*50)

        # output directory
        output_dir = os.path.dirname(outputFile)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            print(f'create directory: {output_dir}')

        self.dates = sorted(self.get_date_list())                                  # Kai
        self.dsNames = list(self.datesDict[self.dates[0]].datasetDict.keys())      # Kai
        self.dsNames = [i for i in TIMESERIES_DSET_NAMES if i in self.dsNames]
        maxDigit = max(len(i) for i in self.dsNames)
        num_date, length, width = self.get_size(
            box=box,
            xstep=xstep,
            ystep=ystep)

        # check if resize is needed for a lower resolution stack, e.g. ionosphere from isce2/topsStack
        resize2shape = None
        if geom_obj and os.path.basename(outputFile).startswith('ion'):
            # compare the original data size between ionosphere and geometry w/o subset/multilook
            ion_size = self.get_size()[1:]
            geom_size = geom_obj.get_size()
            if ion_size != geom_size:
                msg = 'lower resolution ionosphere file detected'
                msg += f' --> resize from {ion_size} to {geom_size} via skimage.transform.resize ...'
                print(msg)

                # matrix shape for the original geometry size w/o subset/multilook
                resize2shape = geom_size
                # data size of the output HDF5 file w/ resize/subset/multilook
                length, width = self.get_size(
                    box=box,
                    xstep=xstep,
                    ystep=ystep,
                    geom_obj=geom_obj)[1:]

        # write HDF5 file
        with h5py.File(outputFile, access_mode) as f:
            print(f'create HDF5 file {outputFile} with {access_mode} mode')

            ###############################
            # 3D datasets containing unwrapPhase, magnitude, coherence, connectComponent, wrapPhase, etc.
            for dsName in self.dsNames:
                dsShape = (num_date, length, width)
                dsDataType = np.float32
                dsCompression = compression
                if dsName in ['connectComponent']:
                    dsDataType = np.int16
                    dsCompression = 'lzf'
                    mli_method = 'nearest'

                print(('create dataset /{d:<{w}} of {t:<25} in size of {s}'
                       ' with compression = {c}').format(d=dsName,
                                                         w=maxDigit,
                                                         t=str(dsDataType),
                                                         s=dsShape,
                                                         c=dsCompression))
                ds = f.create_dataset(dsName,
                                      shape=dsShape,
                                      maxshape=(None, dsShape[1], dsShape[2]),
                                      dtype=dsDataType,
                                      chunks=True,
                                      compression=dsCompression)

                # set no-data value - printout msg
                if dsName.endswith('OffsetVar'):
                    print(f'set no-data value for {dsName} from 99 to NaN.')
                    dsFile = self.datesDict[self.dates[0]].datasetDict[dsName]
                    if dsFile.endswith('cov.bip'):
                        print('convert variance to standard deviation.')

                # msg
                if xstep * ystep > 1:
                    print(f'apply {xstep} x {ystep} multilooking/downsampling via {mli_method} ...')

                prog_bar = ptime.progressBar(maxValue=num_date)
                for i, date in enumerate(self.dates):
                    prog_bar.update(i+1, suffix=f'{date[0]}')

                    # read and/or resize
                    acqObj = self.datesDict[date]
                    data = acqObj.read(dsName,
                                          box=box,
                                          xstep=xstep,
                                          ystep=ystep,
                                          mli_method=mli_method,
                                          resize2shape=resize2shape)[0]

                    # write
                    ds[i, :, :] = data

                ds.attrs['MODIFICATION_TIME'] = str(time.time())
                prog_bar.close()

            ###############################
            # 2D dataset containing reference and secondary dates of all dates
            dsName = 'date'
            dsDataType = np.string_
            dsShape = (num_date, 2)
            print('create dataset /{d:<{w}} of {t:<25} in size of {s}'.format(d=dsName,
                                                                              w=maxDigit,
                                                                              t=str(dsDataType),
                                                                              s=dsShape))
            data = np.array(self.dates, dtype=dsDataType)
            f.create_dataset(dsName, data=data)

            ###############################
            # 1D dataset containing perpendicular baseline of all dates
            if False: # Mute this since we don't have it when reading timeseries dataset
                dsName = 'bperp'
                dsDataType = np.float32
                dsShape = (num_date,)
                print('create dataset /{d:<{w}} of {t:<25} in size of {s}'.format(d=dsName,
                                                                                  w=maxDigit,
                                                                                  t=str(dsDataType),
                                                                                  s=dsShape))
                # get bperp
                data = np.zeros(num_date, dtype=dsDataType)
                for i in range(num_date):
                    acqObj = self.datesDict[self.dates[i]]
                    data[i] = acqObj.get_perp_baseline(family=self.dsName0)
                # write
                f.create_dataset(dsName, data=data)

            ###############################
            # 1D dataset containing bool value of dropping the interferograms or not
            dsName = 'dropIfgram' # Kai: need to delete this?
            dsDataType = np.bool_
            dsShape = (num_date,)
            print('create dataset /{d:<{w}} of {t:<25} in size of {s}'.format(d=dsName,
                                                                              w=maxDigit,
                                                                              t=str(dsDataType),
                                                                              s=dsShape))
            data = np.ones(dsShape, dtype=dsDataType)
            f.create_dataset(dsName, data=data)

            ###############################
            # Attributes
            # read metadata from original data file w/o resize/subset/multilook
            meta = self.get_metadata()
            if extra_metadata:
                meta.update(extra_metadata)
                print(f'add extra metadata: {extra_metadata}')

            # update metadata due to resize
            # for low resolution ionosphere from isce2/topsStack
            if resize2shape:
                print('update metadata due to resize')
                meta = attr.update_attribute4resize(meta, resize2shape)

            # update metadata due to subset
            if box:
                print('update metadata due to subset')
                meta = attr.update_attribute4subset(meta, box)

            # update metadata due to multilook
            if xstep * ystep > 1:
                print('update metadata due to multilook')
                meta = attr.update_attribute4multilook(meta, ystep, xstep)

            # write metadata to HDF5 file at the root level
            meta['FILE_TYPE'] = self.name
            for key, value in meta.items():
                f.attrs[key] = value

        print(f'Finished writing to {outputFile}')
        return outputFile


########################################################################################
class timeseriesAcqDict:
    """
    Timeseries object for timeseries, date, bperp, ... from the same platform and track.

    Example:
        from mintpy.utils import readfile
        from mintpy.utils.insarobj import timeseriesAcqDict
        datasetDict = {'timeseries'    :'$PROJECT_DIR/ion/*.rdr',
                       'date'          :'$PROJECT_DIR/',
                       'bperp'         :bperpDict
                       ...
                      }
        bperpDict = {'20160406':'$PROJECT_DIR/merged/baselines/20160406/bperp',
                     '20160418':'$PROJECT_DIR/merged/baselines/20160418/bperp',
                     ...
                    }
        metadata = readfile.read_attribute('$PROJECT_DIR/merged/interferograms/20160629_20160723/filt_fine.unw')
        acqObj = timeseriesAcqDict(processor='isce', datasetDict=datasetDict, extraMetadata=metadata)
    """

    def __init__(self, name='timeseries', datasetDict={}, metadata=None):
        self.name = name
        self.datasetDict = datasetDict

        self.platform = None
        self.track = None
        self.processor = None
        # platform, track and processor can get values from metadata if they exist
        if metadata is not None:
            for key, value in metadata.items():
                setattr(self, key, value)

    def read(self, family, box=None, xstep=1, ystep=1, mli_method='nearest', resize2shape=None):
        """Read data for the given dataset name.

        Parameters: self         - ifgramDict object
                    family       - str, dataset name
                    box          -  tuple of 4 int, in (x0, y0, x1, y1) with respect to the full resolution
                    x/ystep      - int, number of pixels to skip, with respect to the full resolution
                    mli_method   - str, interpolation method, nearest, mean, median
                    resize2shape - tuple of 2 int, resize the native matrix to the given shape
                                   Set to None for not resizing
        Returns:    data         - 2D np.ndarray
                    meta         - dict, metadata
        """
        self.file = self.datasetDict[family]
        dirname = os.path.dirname(self.file).split('/')[-1]
        box2read = None if resize2shape else box

        # 1. read input file
        data, meta = readfile.read(self.file,
                                   datasetName=family,
                                   box=box2read,
                                   xstep=1,
                                   ystep=1)

        # 2. resize
        if resize2shape:
            # link: https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize
            data = resize(data,
                          output_shape=resize2shape,
                          order=1,
                          mode='constant',
                          anti_aliasing=True,
                          preserve_range=True)

            # 3. subset by box
            if box:
                data = data[box[1]:box[3],
                            box[0]:box[2]]

        # 4. multilook
        if xstep * ystep > 1:
            if mli_method == 'nearest':
                # multilook - nearest resampling
                # output data size
                xsize = int(data.shape[1] / xstep)
                ysize = int(data.shape[0] / ystep)
                # sampling
                data = data[int(ystep/2)::ystep,
                            int(xstep/2)::xstep]
                data = data[:ysize, :xsize]

            else:
                # multilook - mean or median resampling
                data = multilook_data(data,
                                      lks_y=ystep,
                                      lks_x=xstep,
                                      method=mli_method)

        # 5. check the input unit
        if self.data_unit:
            if   self.data_unit ==  'm':    data *= 1e0
            elif self.data_unit == 'cm':    data *= 1e-2
            elif self.data_unit == 'mm':    data *= 1e-3
            elif self.data_unit == 'radian':
                if dirname in ['ion_dates', 'ion_burst_ramp_merged_dates']:
                    phase2range = 1 * float(meta['WAVELENGTH']) / (4.*np.pi)
                else:
                    phase2range = -1 * float(meta['WAVELENGTH']) / (4.*np.pi)
                data *= phase2range

        return data, meta

    def get_size(self, family=TIMESERIES_DSET_NAMES[0]):
        self.file = self.datasetDict[family]
        metadata = readfile.read_attribute(self.file)
        length = int(metadata['LENGTH'])
        width = int(metadata['WIDTH'])
        return length, width

    def get_perp_baseline(self, family=TIMESERIES_DSET_NAMES[0]):
        self.file = self.datasetDict[family]
        metadata = readfile.read_attribute(self.file)
        self.bperp_top = float(metadata['P_BASELINE_TOP_HDR'])
        self.bperp_bottom = float(metadata['P_BASELINE_BOTTOM_HDR'])
        self.bperp = (self.bperp_top + self.bperp_bottom) / 2.0
        return self.bperp

    def get_metadata(self, family=TIMESERIES_DSET_NAMES[0]):
        self.file = self.datasetDict[family]
        self.metadata = readfile.read_attribute(self.file)
        self.length = int(self.metadata['LENGTH'])
        self.width = int(self.metadata['WIDTH'])
        self.metadata['UNIT'] = 'm'

        if self.track:
            self.metadata['TRACK'] = self.track

        if self.platform:
            self.metadata['PLATFORM'] = self.platform

        return self.metadata


    def get_dataset_list(self):
        self.datasetList = list(self.datasetDict.keys())
        return self.datasetList


########################################################################################
class geometryDict:
    """
    Geometry object for Lat, Lon, Height, Incidence, Heading, Bperp, ... from the same platform and track.

    Example:
        from mintpy.utils import readfile
        from mintpy.utils.insarobj import geometryDict
        datasetDict = {'height'        :'$PROJECT_DIR/merged/geom_reference/hgt.rdr',
                       'latitude'      :'$PROJECT_DIR/merged/geom_reference/lat.rdr',
                       'longitude'     :'$PROJECT_DIR/merged/geom_reference/lon.rdr',
                       'incidenceAngle':'$PROJECT_DIR/merged/geom_reference/los.rdr',
                       'heandingAngle' :'$PROJECT_DIR/merged/geom_reference/los.rdr',
                       'shadowMask'    :'$PROJECT_DIR/merged/geom_reference/shadowMask.rdr',
                       'waterMask'     :'$PROJECT_DIR/merged/geom_reference/waterMask.rdr',
                       'bperp'         :bperpDict
                       ...
                      }
        bperpDict = {'20160406':'$PROJECT_DIR/merged/baselines/20160406/bperp',
                     '20160418':'$PROJECT_DIR/merged/baselines/20160418/bperp',
                     ...
                    }
        metadata = readfile.read_attribute('$PROJECT_DIR/merged/interferograms/20160629_20160723/filt_fine.unw')
        geomObj = geometryDict(processor='isce', datasetDict=datasetDict, extraMetadata=metadata)
        geomObj.write2hdf5(outputFile='geometryRadar.h5', access_mode='w', box=(200,500,300,600))
    """

    def __init__(self, name='geometry', processor=None, datasetDict={}, extraMetadata=None):
        self.name = name
        self.processor = processor
        self.datasetDict = datasetDict
        self.extraMetadata = extraMetadata

        # get extra metadata from geometry file if possible
        self.dsNames = list(self.datasetDict.keys())
        if not self.extraMetadata:
            dsFile = self.datasetDict[self.dsNames[0]]
            metadata = readfile.read_attribute(dsFile)
            if all(i in metadata.keys() for i in ['STARTING_RANGE', 'RANGE_PIXEL_SIZE']):
                self.extraMetadata = metadata

    def read(self, family, box=None, xstep=1, ystep=1):
        self.file = self.datasetDict[family]
        # relax dataset name constraint for HDF5 file
        # to support reading waterMask from waterMask.h5 file with /mask dataset
        if self.file.endswith('.h5'):
            dsName = None
        else:
            dsName = family
        data, metadata = readfile.read(self.file,
                                       datasetName=dsName,
                                       box=box,
                                       xstep=xstep,
                                       ystep=ystep)
        return data, metadata

    def get_slant_range_distance(self, box=None, xstep=1, ystep=1):
        """Generate 2D slant range distance if missing from input template file"""
        if not self.extraMetadata:
            return None

        print('prepare slantRangeDistance ...')
        if 'Y_FIRST' in self.extraMetadata.keys():
            # for dataset in geo-coordinates, use:
            # 1) incidenceAngle matrix if available OR
            # 2) constant value from SLANT_RANGE_DISTANCE.
            ds_name = 'incidenceAngle'
            key = 'SLANT_RANGE_DISTANCE'
            if ds_name in self.dsNames:
                print(f'    geocoded input, use incidenceAngle from file: {os.path.basename(self.datasetDict[ds_name])}')
                inc_angle = self.read(family=ds_name)[0].astype(np.float32)
                atr = readfile.read_attribute(self.file)
                if atr.get('PROCESSOR', 'isce') == 'hyp3' and atr.get('UNIT', 'degrees').startswith('rad'):
                    print('    convert incidence angle from Gamma to MintPy convention.')
                    inc_angle[inc_angle == 0] = np.nan              # convert the no-data-value from 0 to nan
                    inc_angle = 90. - (inc_angle * 180. / np.pi)    # hyp3/gamma to mintpy/isce2 convention
                # inc angle -> slant range distance
                data = ut.incidence_angle2slant_range_distance(self.extraMetadata, inc_angle)

            elif key in self.extraMetadata.keys():
                print(f'geocoded input, use constant value from metadata {key}')
                length = int(self.extraMetadata['LENGTH'])
                width = int(self.extraMetadata['WIDTH'])
                range_dist = float(self.extraMetadata[key])
                data = np.ones((length, width), dtype=np.float32) * range_dist
            else:
                return None

        else:
            # for dataset in radar-coordinates, calculate 2D pixel-wise value from geometry
            data = ut.range_distance(self.extraMetadata,
                                     dimension=2,
                                     print_msg=False)

        # subset
        if box is not None:
            data = data[box[1]:box[3],
                        box[0]:box[2]]

        # multilook
        if xstep * ystep > 1:
            # output size if x/ystep > 1
            xsize = int(data.shape[1] / xstep)
            ysize = int(data.shape[0] / ystep)

            # sampling
            data = data[int(ystep/2)::ystep,
                        int(xstep/2)::xstep]
            data = data[:ysize, :xsize]

        return data

    def get_incidence_angle(self, box=None, xstep=1, ystep=1):
        """Generate 2D slant range distance if missing from input template file"""
        if not self.extraMetadata:
            return None

        if 'Y_FIRST' in self.extraMetadata.keys():
            # for dataset in geo-coordinates, use constant value from INCIDENCE_ANGLE.
            key = 'INCIDENCE_ANGLE'
            print(f'geocoded input, use constant value from metadata {key}')
            if key in self.extraMetadata.keys():
                length = int(self.extraMetadata['LENGTH'])
                width = int(self.extraMetadata['WIDTH'])
                inc_angle = float(self.extraMetadata[key])
                data = np.ones((length, width), dtype=np.float32) * inc_angle

            else:
                return None

        else:
            # read DEM if available for more previse calculation
            if 'height' in self.dsNames:
                dem = readfile.read(self.datasetDict['height'], datasetName='height')[0]
            else:
                dem = None

            # for dataset in radar-coordinates, calculate 2D pixel-wise value from geometry
            data = ut.incidence_angle(self.extraMetadata,
                                      dem=dem,
                                      dimension=2,
                                      print_msg=False)

        # subset
        if box is not None:
            data = data[box[1]:box[3],
                        box[0]:box[2]]

        # multilook
        if xstep * ystep > 1:
            # output size if x/ystep > 1
            xsize = int(data.shape[1] / xstep)
            ysize = int(data.shape[0] / ystep)

            # sampling
            data = data[int(ystep/2)::ystep,
                        int(xstep/2)::xstep]
            data = data[:ysize, :xsize]

        return data

    def get_size(self, family=None, box=None, xstep=1, ystep=1):
        if not family:
            family = [i for i in self.datasetDict.keys() if i != 'bperp'][0]
        self.file = self.datasetDict[family]
        metadata = readfile.read_attribute(self.file)

        # update due to subset
        if box:
            length = box[3] - box[1]
            width = box[2] - box[0]
        else:
            length = int(metadata['LENGTH'])
            width = int(metadata['WIDTH'])

        # update due to multilook
        length = length // ystep
        width = width // xstep

        return length, width

    def get_dataset_list(self):
        self.datasetList = list(self.datasetDict.keys())
        return self.datasetList

    def get_metadata(self, family=None):
        if not family:
            family = [i for i in self.datasetDict.keys() if i != 'bperp'][0]
        self.file = self.datasetDict[family]
        self.metadata = readfile.read_attribute(self.file)
        self.length = int(self.metadata['LENGTH'])
        self.width = int(self.metadata['WIDTH'])
        if 'UNIT' in self.metadata.keys():
            self.metadata.pop('UNIT')

        return self.metadata

    def write2hdf5(self, outputFile='geometryRadar.h5', access_mode='w', box=None, xstep=1, ystep=1,
                   compression='lzf', extra_metadata=None):
        """Save/write to HDF5 file with structure defined in:
            https://mintpy.readthedocs.io/en/latest/api/data_structure/#geometry
        """
        print('-'*50)
        if len(self.datasetDict) == 0:
            print('No dataset file path in the object, skip HDF5 file writing.')
            return None

        # output directory
        output_dir = os.path.dirname(outputFile)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
            print(f'create directory: {output_dir}')

        maxDigit = max(len(i) for i in GEOMETRY_DSET_NAMES)
        length, width = self.get_size(box=box, xstep=xstep, ystep=ystep)

        self.outputFile = outputFile
        with h5py.File(self.outputFile, access_mode) as f:
            print(f'create HDF5 file {self.outputFile} with {access_mode} mode')

            ###############################
            for dsName in self.dsNames:
                # 3D datasets containing bperp
                if dsName == 'bperp':
                    self.dateList = list(self.datasetDict[dsName].keys())
                    dsDataType = np.float32
                    self.numDate = len(self.dateList)
                    dsShape = (self.numDate, length, width)
                    ds = f.create_dataset(dsName,
                                          shape=dsShape,
                                          maxshape=(None, dsShape[1], dsShape[2]),
                                          dtype=dsDataType,
                                          chunks=True,
                                          compression=compression)
                    print(('create dataset /{d:<{w}} of {t:<25} in size of {s}'
                           ' with compression = {c}').format(d=dsName,
                                                             w=maxDigit,
                                                             t=str(dsDataType),
                                                             s=dsShape,
                                                             c=str(compression)))

                    print('read coarse grid baseline files and linear interpolate into full resolution ...')
                    prog_bar = ptime.progressBar(maxValue=self.numDate)
                    for i, date_str in enumerate(self.dateList):
                        prog_bar.update(i+1, suffix=date_str)

                        # read and resize
                        fname = self.datasetDict[dsName][date_str]
                        data = read_isce_bperp_file(fname=fname,
                                                    full_shape=self.get_size(),
                                                    box=box,
                                                    xstep=xstep,
                                                    ystep=ystep)
                        # write
                        ds[i, :, :] = data

                    prog_bar.close()

                    # Write 1D dataset date accompnay the 3D bperp
                    dsName = 'date'
                    dsShape = (self.numDate,)
                    dsDataType = np.bytes_
                    print(('create dataset /{d:<{w}} of {t:<25}'
                           ' in size of {s}').format(d=dsName,
                                                     w=maxDigit,
                                                     t=str(dsDataType),
                                                     s=dsShape))
                    data = np.array(self.dateList, dtype=dsDataType)
                    ds = f.create_dataset(dsName, data=data)

                # 2D datasets containing height, latitude/longitude, range/azimuthCoord, incidenceAngle, shadowMask, etc.
                else:
                    dsDataType = np.float32
                    if dsName.lower().endswith('mask'):
                        dsDataType = np.bool_
                    dsShape = (length, width)
                    print(('create dataset /{d:<{w}} of {t:<25} in size of {s}'
                           ' with compression = {c}').format(d=dsName,
                                                             w=maxDigit,
                                                             t=str(dsDataType),
                                                             s=dsShape,
                                                             c=str(compression)))

                    # read
                    data = self.read(family=dsName, box=box, xstep=xstep, ystep=ystep)[0]

                    # water body: -1/True  for water and 0/False for land
                    # water mask:  0/False for water and 1/True  for land
                    fname = os.path.basename(self.datasetDict[dsName])
                    if fname.startswith('waterBody') or fname.endswith('.wbd'):
                        data = ~data
                        print('    input file "{}" is water body (True/False for water/land), '
                              'convert to water mask (False/True for water/land).'.format(fname))

                    elif dsName == 'waterMask':
                        # GMTSAR water/land mask: 1 for land, and nan for water / no data
                        if np.sum(np.isnan(data)) > 0:
                            print('    convert NaN value for waterMask to zero.')
                            data[np.isnan(data)] = 0

                    elif dsName == 'height':
                        noDataValueDEM = -32768
                        if np.any(data == noDataValueDEM):
                            data[data == noDataValueDEM] = np.nan
                            print(f'    convert no-data value for DEM {noDataValueDEM} to NaN.')

                    elif dsName == 'rangeCoord' and xstep != 1:
                        print(f'    scale value of {dsName:<15} by 1/{xstep} due to multilooking')
                        data /= xstep

                    elif dsName == 'azimuthCoord' and ystep != 1:
                        print(f'    scale value of {dsName:<15} by 1/{ystep} due to multilooking')
                        data /= ystep

                    elif dsName in ['incidenceAngle', 'azimuthAngle']:
                        # HyP3 (Gamma) angle of the line-of-sight vector (from ground to SAR platform)
                        # incidence angle 'theta' is measured from horizontal in radians
                        # azimuth   angle 'phi'   is measured from the east with anti-clockwise as positivve in radians
                        atr = readfile.read_attribute(self.file)
                        if atr.get('PROCESSOR', 'isce') == 'hyp3' and atr.get('UNIT', 'degrees').startswith('rad'):

                            if dsName == 'incidenceAngle':
                                msg = f'    convert {dsName:<15} from Gamma (from horizontal in radian) '
                                msg += ' to MintPy (from vertical in degree) convention.'
                                print(msg)
                                data[data == 0] = np.nan                        # convert no-data-value from 0 to nan
                                data = 90. - (data * 180. / np.pi)              # hyp3/gamma to mintpy/isce2 convention

                            elif dsName == 'azimuthAngle':
                                msg = f'    convert {dsName:<15} from Gamma (from east in radian) '
                                msg += ' to MintPy (from north in degree) convention.'
                                print(msg)
                                data[data == 0] = np.nan                        # convert no-data-value from 0 to nan
                                data = data * 180. / np.pi - 90.                # hyp3/gamma to mintpy/isce2 convention
                                data = ut.wrap(data, wrap_range=[-180, 180])    # rewrap within -180 to 180

                    # write
                    data = np.array(data, dtype=dsDataType)
                    ds = f.create_dataset(dsName,
                                          data=data,
                                          chunks=True,
                                          compression=compression)

            ###############################
            # Generate Dataset if it doesn't exist as a binary file: incidenceAngle, slantRangeDistance
            for dsName in [i for i in ['incidenceAngle', 'slantRangeDistance'] if i not in self.dsNames]:
                # Calculate data
                data = None
                if dsName == 'incidenceAngle':
                    data = self.get_incidence_angle(box=box, xstep=xstep, ystep=ystep)
                elif dsName == 'slantRangeDistance':
                    data = self.get_slant_range_distance(box=box, xstep=xstep, ystep=ystep)

                # Write dataset
                if data is not None:
                    dsShape = data.shape
                    dsDataType = np.float32
                    print(('create dataset /{d:<{w}} of {t:<25} in size of {s}'
                           ' with compression = {c}').format(d=dsName,
                                                             w=maxDigit,
                                                             t=str(dsDataType),
                                                             s=dsShape,
                                                             c=str(compression)))
                    ds = f.create_dataset(dsName,
                                          data=data,
                                          dtype=dsDataType,
                                          chunks=True,
                                          compression=compression)

            ###############################
            # Attributes
            self.get_metadata()
            if extra_metadata:
                self.metadata.update(extra_metadata)
                print(f'add extra metadata: {extra_metadata}')

            # update due to subset
            self.metadata = attr.update_attribute4subset(self.metadata, box)
            # update due to multilook
            if xstep * ystep > 1:
                self.metadata = attr.update_attribute4multilook(self.metadata, ystep, xstep)

            self.metadata['FILE_TYPE'] = self.name
            for key, value in self.metadata.items():
                f.attrs[key] = value

        print(f'Finished writing to {self.outputFile}')
        return self.outputFile


########################################################################################
def read_isce_bperp_file(fname, full_shape, box=None, xstep=1, ystep=1):
    """Read ISCE-2 coarse grid perpendicular baseline file, and project it to full size
    Parameters: fname      - str, bperp file name
                full_shape - tuple of 2 int, shape of file in full resolution
                box        - tuple of 4 int, subset range in (x0, y0, x1, y1) with respect to full resolution
                x/ystep    - int, number of pixels to pick/multilook for each output pixel
    Returns:    data       - 2D array of float32
    Example:    fname = '$PROJECT_DIR/merged/baselines/20160418/bperp'
                data = self.read_sice_bperp_file(fname, (3600,2200), box=(200,400,1000,1000))
    """
    # read original data
    data_c = readfile.read(fname)[0]

    # resize to full resolution
    data_min, data_max = np.nanmin(data_c), np.nanmax(data_c)
    if data_max != data_min:
        data_c = (data_c - data_min) / (data_max - data_min)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        data = resize(data_c, full_shape, order=1, mode='edge', preserve_range=True)
    if data_max != data_min:
        data = data * (data_max - data_min) + data_min

    # for debug
    debug_mode=False
    if debug_mode:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8,6));
        im = ax1.imshow(readfile.read(fname)[0]); fig.colorbar(im, ax=ax1)
        im = ax2.imshow(data);  fig.colorbar(im, ax=ax2)
        plt.show()

    # subset
    if box is not None:
        data = data[box[1]:box[3],
                    box[0]:box[2]]

    # multilook
    if xstep * ystep > 1:
        # output size if x/ystep > 1
        xsize = int(data.shape[1] / xstep)
        ysize = int(data.shape[0] / ystep)

        # sampling
        data = data[int(ystep/2)::ystep,
                    int(xstep/2)::xstep]
        data = data[:ysize, :xsize]

    return data


########################################################################################
class platformTrack:

    def __init__(self, name='platformTrack'):  # , pairDict = None):
        self.pairs = None

    def getPairs(self, pairDict, platTrack):
        pairs = pairDict.keys()
        self.pairs = {}
        for pair in pairs:
            if pairDict[pair].platform_track == platTrack:
                self.pairs[pair] = pairDict[pair]

    def getSize_geometry(self, dsName):
        pairs = self.pairs.keys()
        pairs2 = []
        width = []
        length = []
        files = []
        for pair in pairs:
            self.pairs[pair].get_metadata(dsName)
            if self.pairs[pair].length != 0 and self.pairs[pair].file not in files:
                files.append(self.pairs[pair].file)
                pairs2.append(pair)
                width.append(self.pairs[pair].width)
                length.append(self.pairs[pair].length)

        length = np.median(length)
        width = np.median(width)
        return pairs2, length, width

    def getSize(self):
        pairs = self.pairs.keys()
        self.numPairs = len(pairs)
        width = []
        length = []
        for pair in pairs:
            length.append(self.pairs[pair].length)
            width.append(self.pairs[pair].width)
        self.length = np.median(length)
        self.width = np.median(width)

    def getDatasetNames(self):
        # extract the name of the datasets which are actually the keys of
        # observations, quality and geometry dictionaries.

        pairs = [pair for pair in self.pairs.keys()]
        # Assuming all pairs of a given platform-track have the same observations
        # let's extract the keys of the observations of the first pair.

        if self.pairs[pairs[0]].observationsDict is not None:
            self.dsetObservationNames = [k for k in self.pairs[pairs[0]].observationsDict.keys()]
        else:
            self.dsetObservationNames = []

        # Assuming all pairs of a given platform-track have the same quality files
        # let's extract the keys of the quality dictionary of the first pair.
        if self.pairs[pairs[0]].qualityDict is not None:
            self.dsetQualityNames = [k for k in self.pairs[pairs[0]].qualityDict.keys()]
        else:
            self.dsetQualityNames = []

        ##################
        # Despite the observation and quality files, the geometry may not exist
        # for all pairs. Therefore we need to look at all pairs and get possible
        # dataset names.
        self.dsetGeometryNames = []
        for pair in pairs:
            if self.pairs[pair].geometryDict is not None:
                keys = [k for k in self.pairs[pair].geometryDict.keys()]
                self.dsetGeometryNames = list(set(self.dsetGeometryNames) | set(keys))
