############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Heresh Fattahi, Zhang Yunjun, Emre Havazli, 2019 #
############################################################


import datetime as dt
import os
import time

import h5py
import numpy as np

try:
    from osgeo import gdal
except ImportError:
    raise ImportError('Can not import gdal [version>=3.0]!')

from mintpy.multilook import multilook_data
from mintpy.objects import geometry, ifgramStack, sensor
from mintpy.subset import read_subset_template2box
from mintpy.utils import attribute as attr, ptime, utils as ut, writefile


####################################################################################
def run_or_skip(inps, ds_name_dict, out_file):
    flag = 'run'

    # check 1 - update mode status
    if not inps.updateMode:
        return flag

    # check 2 - output file existence
    if ut.run_or_skip(out_file, readable=True) == 'run':
        return flag

    # check 3 - output dataset info
    key = [i for i in ['unwrapPhase', 'height'] if i in ds_name_dict.keys()][0]
    ds_shape = ds_name_dict[key][1]
    in_shape = ds_shape[-2:]

    if 'unwrapPhase' in ds_name_dict.keys():
        # compare date12 and size
        ds = gdal.Open(inps.unwFile, gdal.GA_ReadOnly)
        in_date12_list = [ds.GetRasterBand(i+1).GetMetadata("unwrappedPhase")['Dates']
                          for i in range(ds_shape[0])]
        in_date12_list = ['_'.join(d.split('_')[::-1]) for d in in_date12_list]

        try:
            out_obj = ifgramStack(out_file)
            out_obj.open(print_msg=False)
            out_shape = (out_obj.length, out_obj.width)
            out_date12_list = out_obj.get_date12_list(dropIfgram=False)

            if out_shape == in_shape and set(in_date12_list).issubset(set(out_date12_list)):
                print('All date12   exists in file {} with same size as required,'
                      ' no need to re-load.'.format(os.path.basename(out_file)))
                flag = 'skip'
        except:
            pass

    elif 'height' in ds_name_dict.keys():
        # compare dataset names and size
        in_dsNames = list(ds_name_dict.keys())
        in_size = in_shape[0] * in_shape[1] * 4 * len(in_dsNames)

        out_obj = geometry(out_file)
        out_obj.open(print_msg=False)
        out_dsNames = out_obj.datasetNames
        out_shape = (out_obj.length, out_obj.width)
        out_size = os.path.getsize(out_file)

        if (set(in_dsNames).issubset(set(out_dsNames))
                and out_shape == in_shape
                and out_size > in_size * 0.3):
            print('All datasets exists in file {} with same size as required,'
                  ' no need to re-load.'.format(os.path.basename(out_file)))
            flag = 'skip'

    return flag


def read_subset_box(template_file, meta):
    """Read subset info from template file

    Parameters: template_file - str, path of template file
                meta          - dict, metadata
    Returns:    pix_box       - tuple of 4 int in (x0, y0, x1, y1)
                meta          - dict, metadata
    """

    if template_file and os.path.isfile(template_file):

        # read subset info from template file
        pix_box, geo_box = read_subset_template2box(template_file)

        # geo_box --> pix_box
        if geo_box is not None:
            coord = ut.coordinate(meta)
            pix_box = coord.bbox_geo2radar(geo_box)
            pix_box = coord.check_box_within_data_coverage(pix_box)
            print(f'input bounding box in lalo: {geo_box}')

    else:
        pix_box = None

    if pix_box is not None:
        # update metadata against the new bounding box
        print(f'input bounding box in yx: {pix_box}')
        meta = attr.update_attribute4subset(meta, pix_box)
    else:
        # translate box of None to tuple of 4 int
        length, width = int(meta['LENGTH']), int(meta['WIDTH'])
        pix_box = (0, 0, width, length)

    # ensure all index are in int16
    pix_box = tuple(int(i) for i in pix_box)

    return pix_box, meta


####################################################################################
def extract_metadata(stack):
    """Extract ARIA metadata for MintPy."""

    meta = {}
    ds = gdal.Open(stack, gdal.GA_ReadOnly)
    if not ds:
        raise RuntimeError(f'Failed to open file {stack} with GDAL.')

    # read metadata from unwrapStack.vrt file
    print(f'extract metadata from {stack}')
    metaUnw = ds.GetRasterBand(1).GetMetadata("unwrappedPhase")

    # copy over all metadata from unwrapStack
    for key, value in metaUnw.items():
        if key not in ["Dates", "perpendicularBaseline"]:
            meta[key] = value

    meta["ANTENNA_SIDE"] = -1
    meta["PROCESSOR"] = "isce"
    meta["FILE_LENGTH"] = ds.RasterYSize
    meta["LENGTH"] = ds.RasterYSize
    meta["ORBIT_DIRECTION"] = meta["orbitDirection"].upper()
    meta["PLATFORM"] = "Sen"
    meta["WAVELENGTH"] = float(meta["Wavelength (m)"])
    meta["WIDTH"] = ds.RasterXSize
    meta["NUMBER_OF_PAIRS"] = ds.RasterCount
    meta["STARTING_RANGE"] = float(meta["startRange"])

    # Note from YZ, 2019-07-25
    # convert isce azimuth angle to roipac orbit heading angle
    # This value is not consistent with band2 of los.rdr from ISCE/topsStack
    # need to check with ARIA-tools team.
    # use hardwired value for now
    #az_angle = float(meta["azimuthAngle"])
    #head_angle = -1 * (270 + az_angle)
    #head_angle -= np.round(head_angle / 360.) * 360.
    #meta['HEADING'] = head_angle
    if meta["ORBIT_DIRECTION"].startswith("D"):
        meta["HEADING"] = -168
    else:
        meta["HEADING"] = -12

    # ARIA standard products currently don't have number of range and
    # azimuth looks. They are however fixed to the following values
    meta["ALOOKS"] = 7
    meta["RLOOKS"] = 19
    meta["RANGE_PIXEL_SIZE"] = float(meta["slantRangeSpacing"]) * meta["RLOOKS"]

    # number of independent looks
    sen_dict = sensor.SENSOR_DICT['sen']
    rgfact = sen_dict['IW2']['range_resolution'] / sen_dict['range_pixel_size']
    azfact = sen_dict['IW2']['azimuth_resolution'] / sen_dict['azimuth_pixel_size']
    meta['NCORRLOOKS'] = meta['RLOOKS'] * meta['ALOOKS'] / (rgfact * azfact)

    # geo transformation
    transform = ds.GetGeoTransform()

    x_step = abs(transform[1])
    y_step = abs(transform[5]) * -1.

    W = transform[0]
    N = transform[3]
    E = W + x_step * ds.RasterXSize
    S = N + y_step * ds.RasterYSize

    meta["X_FIRST"] = f'{W:.9f}'
    meta["Y_FIRST"] = f'{N:.9f}'
    meta["X_STEP"] = f'{x_step:.9f}'
    meta["Y_STEP"] = f'{y_step:.9f}'
    meta["X_UNIT"] = "degrees"
    meta["Y_UNIT"] = "degrees"

    utc = meta["UTCTime (HH:MM:SS.ss)"]
    utc = time.strptime(utc, "%H:%M:%S.%f")
    meta["CENTER_LINE_UTC"] = utc.tm_hour*3600.0 + utc.tm_min*60.0 + utc.tm_sec

    # following values probably won't be used anywhere for the geocoded data
    # earth radius
    meta["EARTH_RADIUS"] = 6337286.638938101
    # nominal altitude of Sentinel1 orbit
    meta["HEIGHT"] = 693000.0

    if meta["ORBIT_DIRECTION"].startswith("ASC"):
        meta["LAT_REF1"] = str(S)
        meta["LAT_REF2"] = str(S)
        meta["LAT_REF3"] = str(N)
        meta["LAT_REF4"] = str(N)
        meta["LON_REF1"] = str(W)
        meta["LON_REF2"] = str(E)
        meta["LON_REF3"] = str(W)
        meta["LON_REF4"] = str(E)
    else:
        meta["LAT_REF1"] = str(N)
        meta["LAT_REF2"] = str(N)
        meta["LAT_REF3"] = str(S)
        meta["LAT_REF4"] = str(S)
        meta["LON_REF1"] = str(E)
        meta["LON_REF2"] = str(W)
        meta["LON_REF3"] = str(E)
        meta["LON_REF4"] = str(W)

    ds = None
    return meta


def write_geometry(outfile, demFile, incAngleFile, azAngleFile=None, waterMaskFile=None,
                   box=None, xstep=1, ystep=1):
    """Write geometry HDF5 file from list of VRT files."""

    print('-'*50)
    # box to gdal arguments
    # link: https://gdal.org/python/osgeo.gdal.Band-class.html#ReadAsArray
    if box is not None:
        kwargs = dict(
            xoff=box[0],
            yoff=box[1],
            win_xsize=box[2]-box[0],
            win_ysize=box[3]-box[1])
    else:
        kwargs = dict()

    print(f'writing data to HDF5 file {outfile} with a mode ...')
    with h5py.File(outfile, 'a') as f:

        # height
        ds = gdal.Open(demFile, gdal.GA_ReadOnly)
        bnd = ds.GetRasterBand(1)
        data = np.array(bnd.ReadAsArray(**kwargs), dtype=np.float32)
        data = multilook_data(data, ystep, xstep, method='nearest')
        data[data == bnd.GetNoDataValue()] = np.nan
        f['height'][:,:] = data

        # slantRangeDistance
        f['slantRangeDistance'][:,:] = float(f.attrs['STARTING_RANGE'])

        # incidenceAngle
        ds = gdal.Open(incAngleFile, gdal.GA_ReadOnly)
        bnd = ds.GetRasterBand(1)
        data = bnd.ReadAsArray(**kwargs)
        data = multilook_data(data, ystep, xstep, method='nearest')
        data[data == bnd.GetNoDataValue()] = np.nan
        f['incidenceAngle'][:,:] = data

        # azimuthAngle
        if azAngleFile is not None:
            ds = gdal.Open(azAngleFile, gdal.GA_ReadOnly)
            bnd = ds.GetRasterBand(1)
            data = bnd.ReadAsArray(**kwargs)
            data = multilook_data(data, ystep, xstep, method='nearest')
            data[data == bnd.GetNoDataValue()] = np.nan
            # azimuth angle of the line-of-sight vector:
            # ARIA: vector from target to sensor measured from the east  in counterclockwise direction
            # ISCE: vector from sensor to target measured from the north in counterclockwise direction
            # convert ARIA format to ISCE format, which is used in mintpy
            data -= 90
            f['azimuthAngle'][:,:] = data

        # waterMask
        if waterMaskFile is not None:
            # read
            ds = gdal.Open(waterMaskFile, gdal.GA_ReadOnly)
            bnd = ds.GetRasterBand(1)
            water_mask = bnd.ReadAsArray(**kwargs)
            water_mask = multilook_data(water_mask, ystep, xstep, method='nearest')
            water_mask[water_mask == bnd.GetNoDataValue()] = False

            # assign False to invalid pixels based on incAngle data
            ds = gdal.Open(incAngleFile, gdal.GA_ReadOnly)
            bnd = ds.GetRasterBand(1)
            data = bnd.ReadAsArray(**kwargs)
            data = multilook_data(data, ystep, xstep, method='nearest')
            water_mask[data == bnd.GetNoDataValue()] = False

            # write
            f['waterMask'][:,:] = water_mask

    print(f'finished writing to HD5 file: {outfile}\n')
    return outfile


def write_ifgram_stack(outfile, stackFiles, box=None, xstep=1, ystep=1, mli_method='nearest'):
    """Write stacks to HDF5 files from stack VRT files
    """

    print('-'*50)

    # remove None entries
    stackFiles = {key:val for key, val in stackFiles.items() if val is not None}

    # check all files exist
    for dsName, fname in stackFiles.items():
        if not os.path.exists(fname):
            raise Exception("%s does not exist" % fname)

    # determine field length for printing
    max_digit = max(len(os.path.basename(str(i))) for i in stackFiles.values())
    for stackFile in stackFiles.values():
        if stackFile is not None:
            print('open {f:<{w}} with gdal ...'.format(f=os.path.basename(stackFile), w=max_digit))

    # extract NoDataValue for each stack (from the last */date2_date1.vrt file for example)
    noDataValues = {}
    for dsName in stackFiles.keys():
        dsStack = gdal.Open(stackFiles[dsName], gdal.GA_ReadOnly)
        ds = gdal.Open(dsStack.GetFileList()[-1], gdal.GA_ReadOnly)
        noDataValues[dsName] = ds.GetRasterBand(1).GetNoDataValue()

        fileName = os.path.basename(stackFiles[dsName])
        print(f'grab NoDataValue for {fileName:<{max_digit}}: '
              f'{noDataValues[dsName]:<5} and convert to 0.')
        ds = None

    # sort the order of interferograms based on date1_date2 with date1 < date2
    nPairs = dsStack.RasterCount
    d12BandDict = {}
    for ii in range(nPairs):
        bnd = dsStack.GetRasterBand(ii+1)
        d12 = bnd.GetMetadata(bnd.GetMetadataDomainList()[0])["Dates"]
        d12 = sorted(d12.split("_"))
        d12 = f'{d12[0]}_{d12[1]}'
        d12BandDict[d12] = ii+1
    d12List = sorted(d12BandDict.keys())
    print(f'number of interferograms: {len(d12List)}')

    # box to gdal arguments
    # link: https://gdal.org/python/osgeo.gdal.Band-class.html#ReadAsArray
    if box is not None:
        kwargs = dict(
            xoff=box[0],
            yoff=box[1],
            win_xsize=box[2]-box[0],
            win_ysize=box[3]-box[1])
    else:
        kwargs = dict()

    # write to HDF5 file
    print(f'writing data to HDF5 file {outfile} with a mode ...')
    with h5py.File(outfile, "a") as f:
        prog_bar = ptime.progressBar(maxValue=nPairs)
        for ii in range(nPairs):
            d12 = d12List[ii]
            bndIdx = d12BandDict[d12]
            prog_bar.update(ii+1, suffix=f'{d12} {ii+1}/{nPairs}')

            f["date"][ii,0] = d12.split("_")[0].encode("utf-8")
            f["date"][ii,1] = d12.split("_")[1].encode("utf-8")
            f["dropIfgram"][ii] = True

            # loop through stacks
            print(stackFiles.keys())
            for dsName in stackFiles.keys():
                dsStack = gdal.Open(stackFiles[dsName], gdal.GA_ReadOnly)
                bnd = dsStack.GetRasterBand(bndIdx)
                data = bnd.ReadAsArray(**kwargs)
                if xstep * ystep > 1:
                    mli_method_spec = mli_method if dsName not in \
                        ['connCompStack'] else 'nearest'
                    print(f'apply {xstep} x {ystep} multilooking/downsampling via '
                          f'{mli_method_spec} to: {dsName}')
                    data = multilook_data(data, ystep, xstep, method=mli_method_spec)
                data[data == noDataValues[dsName]] = 0  #assign pixel with no-data to 0

                if dsName == 'unwrapPhase':
                    data *= -1  # date2_date1 -> date1_date2
                    f['unwrapPhase'][ii,:,:] = data

                    bperp = float(bnd.GetMetadata("unwrappedPhase")["perpendicularBaseline"])
                    bperp *= -1.0  # date2_date1 -> date1_date2
                    f["bperp"][ii] = bperp

                elif dsName == 'coherence':
                    f["coherence"][ii,:,:] = data

                elif dsName == 'connectComponent':
                    f["connectComponent"][ii,:,:] = data

                elif dsName == 'magnitude':
                    f["magnitude"][ii,:,:] = data

                elif dsName == 'ionosphere':
                    data *= -1.0  #date2_date1 -> date1_date2
                    f["unwrapPhase"][ii,:,:] = data

                    bperp = float(bnd.GetMetadata("ionosphere")["perpendicularBaseline"])
                    bperp *= -1.0  #date2_date1 -> date1_date2
                    f["bperp"][ii] = bperp

        prog_bar.close()

        # add MODIFICATION_TIME metadata to each 3D dataset
        for dsName in stackFiles.keys():
            dsName = 'unwrapPhase' if dsName == 'ionosphere' else dsName
            f[dsName].attrs['MODIFICATION_TIME'] = str(time.time())

    print(f'finished writing to HD5 file: {outfile}\n')
    dsUnw = None
    dsCoh = None
    dsComp = None
    dsAmp = None
    return outfile


# OPTIONAL - ARIA model-based corrections troposphereTotal, solidearthtides
def write_timeseries(outfile, corrStack, box=None,
                      xstep=1, ystep=1, mli_method='nearest'):
    """Write SET and TropsphericDelay corrections to HDF5 file from stack VRT files
       Correction layers are stored for each SAR acquisition date

    ARIA_GUNW_NC_PATH:
    troposhereTotal : models GMAO, HRRR, HRES, ERA5 '/science/grids/corrections/external/troposphere/'
    solidEarthTides '/science/grids/corrections/derived/solidearthtides/'
    """

    print('-'*50)

    # determine field length for printing
    max_digit = len(os.path.basename(str(corrStack)))

    if corrStack is not None:
        print('open {f:<{w}} with gdal ...'.format(f=os.path.basename(corrStack), w=max_digit))

        # check all files exist
        if not os.path.exists(corrStack):
            raise Exception("%s does not exist" % corrStack)

    # open raster
    dsCor = gdal.Open(corrStack, gdal.GA_ReadOnly)
    # extract NoDataValue (from the last date.vrt file for example)
    ds = gdal.Open(dsCor.GetFileList()[-1], gdal.GA_ReadOnly)
    noDataValue = ds.GetRasterBand(1).GetNoDataValue()
    ds = None

    # get the layer name (for tropo this will get the model name)
    layer = dsCor.GetRasterBand(1).GetMetadataDomainList()[0]

    # Get the wavelength. need to convert radians to meters
    wavelength = np.float64(dsCor.GetRasterBand(1).GetMetadata(layer)["Wavelength (m)"])
    phase2range = -wavelength / (4.*np.pi)

    # get model dates and time
    nDate = dsCor.RasterCount
    dateDict = {}
    sensingDict = {}
    for ii in range(nDate):
        bnd = dsCor.GetRasterBand(ii+1)
        date = bnd.GetMetadata(layer)["Dates"]
        utc = dt.datetime.strptime(date + ',' + \
                                   bnd.GetMetadata(layer)["UTCTime (HH:MM:SS.ss)"],
                                   "%Y%m%d,%H:%M:%S.%f")
        dateDict[date] = ii+1
        sensingDict[ii+1] = str(utc)
    dateList = sorted(dateDict.keys())
    print(f'number of {layer} datasets: {len(dateList)}')

    # box to gdal arguments
    # link: https://gdal.org/python/osgeo.gdal.Band-class.html#ReadAsArray
    if box is not None:
        kwargs = dict(
            xoff=box[0],
            yoff=box[1],
            win_xsize=box[2]-box[0],
            win_ysize=box[3]-box[1])
    else:
        kwargs = dict()

    if xstep * ystep > 1:
        msg = f'apply {xstep} x {ystep} multilooking/downsampling via {mli_method} to {layer}'
        print(msg)

    print(f'writing data to HDF5 file {outfile} with a mode ...')
    with h5py.File(outfile, "a") as f:
        prog_bar = ptime.progressBar(maxValue=nDate)
        for ii in range(nDate):
            date = dateList[ii]
            bndIdx = dateDict[date]
            utc = sensingDict[bndIdx]
            prog_bar.update(ii+1, suffix=f'{date} {ii+1}/{nDate}')

            f["date"][ii] = date.encode("utf-8")
            f["sensingMid"][ii] = utc.encode("utf-8")

            bnd = dsCor.GetRasterBand(bndIdx)
            data = bnd.ReadAsArray(**kwargs)
            data = multilook_data(data, ystep, xstep, method=mli_method)
            data[data == noDataValue] = 0         #assign pixel with no-data to 0
            data[np.isnan(data)] = 0              #assign nan pixel to 0
            f["timeseries"][ii,:,:] = data * phase2range

        prog_bar.close()

        # add MODIFICATION_TIME metadata to each 3D dataset
        for dsName in ['timeseries']:
            f[dsName].attrs['MODIFICATION_TIME'] = str(time.time())

    print(f'finished writing to HD5 file: {outfile}\n')
    dsCor = None

    return outfile


def get_number_of_epochs(vrtfile):
    ds = gdal.Open(vrtfile, gdal.GA_ReadOnly)

    return ds.RasterCount


def get_correction_layer(correction_filename):
    ds = gdal.Open(correction_filename, gdal.GA_ReadOnly)
    # get the layer name (for tropo this will get the model name)
    layer_name = ds.GetRasterBand(1).GetMetadataDomainList()[0]

    # Get type of correction
    if layer_name in ['GMAO', 'HRES', 'HRRR', "ERA5"]:
        layer_type = 'tropo'
    else:
        # ionosphere, solid earth tides
        layer_type = layer_name

    #close
    ds = None

    return layer_name, layer_type

####################################################################################
def load_aria(inps):
    """Prepare and load ARIA data and metadata into HDF5/MintPy format."""

    start_time = time.time()
    print(f'update mode: {inps.updateMode}')

    # extract metadata
    meta = extract_metadata(inps.unwFile)
    box, meta = read_subset_box(inps.template_file, meta)
    if inps.xstep * inps.ystep > 1:
        meta = attr.update_attribute4multilook(
            meta,
            lks_y=inps.ystep,
            lks_x=inps.xstep,
        )

    length = int(meta["LENGTH"])
    width = int(meta["WIDTH"])
    num_pair = int(meta["NUMBER_OF_PAIRS"])

    # prepare output directory
    out_dir = os.path.dirname(inps.outfile[0])
    os.makedirs(out_dir, exist_ok=True)

    ########## output file 1 - ifgramStack
    # define dataset structure for ifgramStack
    ds_name_dict = {
        "date"             : (np.dtype('S8'), (num_pair, 2)),
        "dropIfgram"       : (np.bool_,       (num_pair,)),
        "bperp"            : (np.float32,     (num_pair,)),
        "unwrapPhase"      : (np.float32,     (num_pair, length, width)),
        "coherence"        : (np.float32,     (num_pair, length, width)),
        "connectComponent" : (np.int16,       (num_pair, length, width)),
    }
    if inps.magFile is not None:
        ds_name_dict['magnitude'] = (np.float32, (num_pair, length, width))

    if run_or_skip(inps, ds_name_dict, out_file=inps.outfile[0]) == 'run':
        # initiate h5 file with defined structure
        meta['FILE_TYPE'] = 'ifgramStack'
        writefile.layout_hdf5(
            inps.outfile[0],
            ds_name_dict,
            metadata=meta,
            compression=inps.compression,
        )

        write_ifgram_stack(
            inps.outfile[0],
            stackFiles={'unwrapPhase': inps.unwFile,
                         'coherence': inps.corFile,
                         'connectComponent': inps.connCompFile,
                         'magnitude': inps.magFile},
            box=box,
            xstep=inps.xstep,
            ystep=inps.ystep,
            mli_method=inps.method,
        )

    ########## output file 2 - geometryGeo
    # define dataset structure for geometry
    ds_name_dict = {
        "height"             : (np.float32, (length, width)),
        "incidenceAngle"     : (np.float32, (length, width)),
        "slantRangeDistance" : (np.float32, (length, width)),
    }
    if inps.azAngleFile is not None:
        ds_name_dict["azimuthAngle"] = (np.float32, (length, width))
    if inps.waterMaskFile is not None:
        ds_name_dict["waterMask"] = (np.bool_, (length, width))

    if run_or_skip(inps, ds_name_dict, out_file=inps.outfile[1]) == 'run':
        # initiate h5 file with defined structure
        meta['FILE_TYPE'] = 'geometry'
        writefile.layout_hdf5(
            inps.outfile[1],
            ds_name_dict,
            metadata=meta,
            compression=inps.compression,
        )

        # write data to disk
        write_geometry(
            inps.outfile[1],
            demFile=inps.demFile,
            incAngleFile=inps.incAngleFile,
            azAngleFile=inps.azAngleFile,
            waterMaskFile=inps.waterMaskFile,
            box=box,
            xstep=inps.xstep,
            ystep=inps.ystep,
        )

    ########## output file 3 - correction layers

    # 3.1 - ionosphere
    if inps.ionoFile:
        # define correction dataset structure for ifgramStack
        ds_name_dict = {
            'date'             : (np.dtype('S8'), (num_pair, 2)),
            'dropIfgram'       : (np.bool_,       (num_pair,)),
            'bperp'            : (np.float32,     (num_pair,)),
            'unwrapPhase'      : (np.float32,     (num_pair, length, width)),
            "coherence"        : (np.float32,     (num_pair, length, width)),
        }
        meta['FILE_TYPE'] = 'ifgramStack'

        layer_name, _ = get_correction_layer(inps.ionoFile)

        if run_or_skip(inps, ds_name_dict, out_file=inps.outfile[0]) == 'run':
            outname = f'{out_dir}/ionStack.h5'

            writefile.layout_hdf5(
                outname,
                ds_name_dict,
                metadata=meta,
                compression=inps.compression,
                )

            # write data to disk
            write_ifgram_stack(
                outname,
                stackFiles={'ionosphere': inps.ionoFile,
                            'coherence': inps.corFile,},
                box=box,
                xstep=inps.xstep,
                ystep=inps.ystep,
                mli_method=inps.method,
            )

    # 3.2 - model based corrections: SolidEarthTides and Troposphere
    # Loop through other correction layers also provided as epochs
    # handle multiple tropo stacks (if specified)
    if inps.tropoFile is None:
        inps.tropoFile = [None]
    correction_layers = inps.tropoFile + [inps.setFile]
    for layer in correction_layers:
        if layer:
            # get name and type
            layer_name, _ = get_correction_layer(layer)
            num_dates = get_number_of_epochs(layer)

            meta['FILE_TYPE'] = 'timeseries'
            meta['UNIT'] = 'm'
            meta['DATA_TYPE'] = 'float32'
            for key in ['REF_Y', 'REF_X', 'REF_DATE']:
                if key in meta.keys():
                    meta.pop(key)

            # define correction dataset structure for timeseries
            ds_name_dict = {
            'date'           : (np.dtype('S8'),  (num_dates, )),
            'sensingMid'     : (np.dtype('S15'), (num_dates, )),
            'timeseries'     : (np.float32,      (num_dates, length, width)),
            }

            if run_or_skip(inps, ds_name_dict, out_file=inps.outfile[0]) == 'run':
                writefile.layout_hdf5(
                    f'{out_dir}/{layer_name}_ARIA.h5',
                    ds_name_dict,
                    metadata=meta,
                    compression=inps.compression,
                    )

                # write data to disk
                write_timeseries(
                    f'{out_dir}/{layer_name}_ARIA.h5',
                    corrStack=layer,
                    box=box,
                    xstep=inps.xstep,
                    ystep=inps.ystep,
                    )
    print('-'*50)

    # used time
    m, s = divmod(time.time() - start_time, 60)
    print(f'time used: {m:02.0f} mins {s:02.1f} secs.')
