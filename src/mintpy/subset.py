############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, 2013               #
############################################################


import os

import h5py
import numpy as np

from mintpy.utils import (
    attribute as attr,
    ptime,
    readfile,
    utils as ut,
    writefile,
)


################################################################
def get_coverage_box(atr):
    """Get Coverage Box of data in geo and pixel coordinates.

    Parameters: atr     - dict, meta data dictionary
    Returns:    pix_box - 4-tuple of int, defining in (UL_X, UL_Y, LR_X, LR_Y)
                geo_box - 4-tuple of float in lat/lon
    """

    length = int(atr['LENGTH'])
    width = int(atr['WIDTH'])

    # Get geo box
    if all(x in atr.keys() for x in ['Y_STEP', 'X_STEP', 'Y_FIRST', 'X_FIRST']):
        lat_step = float(atr['Y_STEP'])
        lon_step = float(atr['X_STEP'])
        ul_lat = float(atr['Y_FIRST'])
        ul_lon = float(atr['X_FIRST'])
        lr_lat = ul_lat + lat_step*length
        lr_lon = ul_lon + lon_step*width
        geo_box = (ul_lon, ul_lat, lr_lon, lr_lat)
    else:
        geo_box = None

    # Get pixel box
    if all(f'SUBSET_{x}' in atr.keys() for x in ['YMIN', 'XMIN', 'YMAX', 'XMAX']):
        pix_box = (
            int(atr['SUBSET_XMIN']),
            int(atr['SUBSET_YMIN']),
            int(atr['SUBSET_XMAX']),
            int(atr['SUBSET_YMAX']),
        )
    else:
        pix_box = None

    return pix_box, geo_box


def read_subset_template2box(template_file):
    """Read mintpy.subset.lalo/yx option from template file into box type.

    Parameters: template_file - str, path to the template file
    Returns     pix/geo_box   - tuple of 4 int or None
    """
    # initiate output
    pix_box, geo_box = None, None

    # read template file into dict
    tmpl = readfile.read_template(template_file)

    # dict: yx -> pix_box
    key = 'mintpy.subset.yx'
    if key in tmpl.keys():
        # ignore common typo: [ ]
        opt_str = tmpl[key].replace('[','').replace(']','')
        # convert : to ,
        opt_str = opt_str.replace(':',',')

        opt_str_list = [i.strip() for i in opt_str.split(',')]
        if len(opt_str_list) == 4:
            y0, y1, x0, x1 = (int(i) for i in opt_str_list)
            pix_box = (x0, y0, x1, y1)

    # dict: lalo -> geo_box
    key = 'mintpy.subset.lalo'
    if key in tmpl.keys():
        # ignore common typo: [ ]
        opt_str = tmpl[key].replace('[','').replace(']','')
        # convert : to ,
        opt_str = opt_str.replace(':',',')

        opt_str_list = [i.strip() for i in opt_str.split(',')]
        if len(opt_str_list) == 4:
            lat0, lat1, lon0, lon1 = (float(i) for i in opt_str_list)
            geo_box = (lon0, lat1, lon1, lat0)

    return pix_box, geo_box


def subset_box2inps(inps, pix_box, geo_box):
    """Update inps.subset_y/x/lat/lon from pixel_box and geo_box"""
    if geo_box:
        inps.subset_lon = [geo_box[0], geo_box[2]]
        inps.subset_lat = [geo_box[1], geo_box[3]]
    else:
        inps.subset_lon = None
        inps.subset_lat = None
    if pix_box:
        inps.subset_x = [pix_box[0], pix_box[2]]
        inps.subset_y = [pix_box[1], pix_box[3]]
    else:
        inps.subset_x = None
        inps.subset_y = None
    return inps


def get_box_overlap_index(box1, box2):
    """Get index box overlap area of two input boxes.

    Parameters: box1/2             - 4-tuple of int, indicating coverage of box1/2
                                     defining in (x0, y0, x1, y1)
    Returns:    overlap_idx_box1/2 - 4-tuple of int, indicating index of overlap area in box1/2
                                     defining in (idx_x0, idx_y0, idx_x1, idx_y1)
    """
    # Calculate the overlap of two input boxes
    # and output the index box of the overlap in each's coord.

    # Calculate Overlap Box
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])
    if x0 >= x1 or y0 >= y1:
        msg = 'No overlap between two input box range!\n'
        msg += f'box 1: {box1}\n'
        msg += f'box 2: {box2}\n'
        raise ValueError(msg)
    overlap_box = (x0, y0, x1, y1)

    # Overlap index for box1
    overlap_idx_box1 = (overlap_box[0] - box1[0],
                        overlap_box[1] - box1[1],
                        overlap_box[2] - box1[0],
                        overlap_box[3] - box1[1])
    # Overlap index for box2
    overlap_idx_box2 = (overlap_box[0] - box2[0],
                        overlap_box[1] - box2[1],
                        overlap_box[2] - box2[0],
                        overlap_box[3] - box2[1])

    return overlap_idx_box1, overlap_idx_box2


################################################################
def subset_input_dict2box(subset_dict, meta_dict):
    """Convert subset inputs dict into box in radar and/or geo coord.
    Parameters: subset_dict - dict, including the following 4 objects:
                              subset_x   : list of 2 int,   subset in x direction,   default=None
                              subset_y   : list of 2 int,   subset in y direction,   default=None
                              subset_lat : list of 2 float, subset in lat direction, default=None
                              subset_lon : list of 2 float, subset in lon direction, default=None
                meta_dict   - dict, including the following items:
                              'WIDTH'      : int
                              'LENGTH': int
                              'X_FIRST'    : float, optional
                              'Y_FIRST'    : float, optional
                              'X_STEP'     : float, optional
                              'Y_STEP'     : float, optional
    Returns:    pix_box     - 4-tuple of int, in pixel unit of 1, in (x0, y0, x1, y1)
                geo_box     - 4-tuple of float, in lat/lon unit (degree)
                              None if file is in radar coordinate.
    Examples:
        subset_dict = {'subset_x': None, 'subset_y': None, 'subset_lat': [30.5, 31.0], 'subset_lon': [130.0, 131.0]}
        subset_dict = {'subset_x': [100, 1100], 'subset_y': [2050, 2550], 'subset_lat': None, 'subset_lon': None}
        pix_box          = subset_input_dict2box(subset_dict, meta_dict)[0]
        pix_box, geo_box = subset_input_dict2box(subset_dict, meta_dict)
    """

    # Data Coverage
    width = int(float(meta_dict['WIDTH']))
    length = int(float(meta_dict['LENGTH']))

    # Use subset_lat/lon input if existed,  priority: lat/lon > y/x > len/wid
    coord = ut.coordinate(meta_dict)

    if 'Y_FIRST' in meta_dict.keys():
        geocoded_flag =True
    else:
        geocoded_flag =False

    if geocoded_flag:
        if subset_dict.get('subset_lat', None):
            sub_y = coord.lalo2yx(subset_dict['subset_lat'], None)[0]
        else:
            sub_y = [0, length]

        if subset_dict.get('subset_lon', None):
            sub_x = coord.lalo2yx(None, subset_dict['subset_lon'])[1]
        else:
            sub_x = [0, width]

        # Get subset box in y/x
        sub_x = sorted(sub_x)
        sub_y = sorted(sub_y)
        pix_box = (sub_x[0], sub_y[0], sub_x[1], sub_y[1])

        # Get subset box in lat/lon from subset box in y/x
        geo_box = coord.box_pixel2geo(pix_box)

    # for data in radar coordinates
    else:
        if subset_dict.get('subset_lat', None):
            geo_box = (subset_dict['subset_lon'][0], subset_dict['subset_lat'][1], subset_dict['subset_lon'][1], subset_dict['subset_lat'][0])
            pix_box = coord.bbox_geo2radar(geo_box, buf=0)
        else:
            if subset_dict['subset_y']:
                sub_y = subset_dict['subset_y']
            else:
                sub_y = [0, length]

            if subset_dict['subset_x']:
                sub_x = subset_dict['subset_x']
            else:
                sub_x = [0, width]
                 # Get subset box in y/x
            sub_x = sorted(sub_x)
            sub_y = sorted(sub_y)
            pix_box = (sub_x[0], sub_y[0], sub_x[1], sub_y[1])

            # Get subset box in lat/lon from subset box in y/x
            geo_box = coord.box_pixel2geo(pix_box)

    return pix_box, geo_box


################################################################
def subset_dataset(fname, dsName, pix_box, pix_box4data, pix_box4subset, fill_value=np.nan):

    # read data
    print(f'reading {dsName} in {pix_box4data} from {os.path.basename(fname)} ...')
    data = readfile.read(fname, datasetName=dsName, box=pix_box4data, print_msg=False)[0]
    ds_shape = data.shape
    ds_ndim = len(ds_shape)

    # subset 2D data
    if ds_ndim == 2:
        data_out = np.ones((pix_box[3] - pix_box[1],
                            pix_box[2] - pix_box[0]),
                           data.dtype) * fill_value
        data_out[pix_box4subset[1]:pix_box4subset[3],
                 pix_box4subset[0]:pix_box4subset[2]] = data

    # subset 3D data
    elif ds_ndim == 3:
        data_out = np.ones((ds_shape[0],
                            pix_box[3] - pix_box[1],
                            pix_box[2] - pix_box[0]),
                           data.dtype) * fill_value
        data_out[:,
                 pix_box4subset[1]:pix_box4subset[3],
                 pix_box4subset[0]:pix_box4subset[2]] = data

    return data_out


def subset_file(fname, subset_dict_input, out_file=None):
    """Subset file with
    Parameters: fname       - str, path/name of file
                subset_dict - dict, subsut parameter, including the following items:
                    subset_x   : list of 2 int,   subset in x direction,   default=None
                    subset_y   : list of 2 int,   subset in y direction,   default=None
                    subset_lat : list of 2 float, subset in lat direction, default=None
                    subset_lon : list of 2 float, subset in lon direction, default=None
                    tight      : bool, tight subset or not, for lookup table file, i.e. geomap*.trans
                    fill_value : float, optional. filled value for area outside of data coverage. default=None
                                 None/not-existed to subset within data coverage only.
                out_file    - str, path/name of output file
    Returns:    out_file    - str, path/name of output file
                              default: add prefix 'sub_',   if fname     in the current directory;
                                       keep the same fname, if fname not in the current directory.
    """

    # Input File Info
    atr = readfile.read_attribute(fname)
    width = int(atr['WIDTH'])
    length = int(atr['LENGTH'])
    print(f"subset {atr['FILE_TYPE']} file: {fname} ...")

    subset_dict = subset_dict_input.copy()
    # Read Subset Inputs into 4-tuple box in pixel and geo coord
    pix_box, geo_box = subset_input_dict2box(subset_dict, atr)

    coord = ut.coordinate(atr)
    # if fill_value exists and not None, subset data and fill assigned value for area out of its coverage.
    # otherwise, re-check subset to make sure it's within data coverage and initialize the matrix with np.nan
    outfill = False
    if 'fill_value' in subset_dict.keys() and subset_dict['fill_value']:
        outfill = True
    else:
        outfill = False
    if not outfill:
        pix_box = coord.check_box_within_data_coverage(pix_box)
        subset_dict['fill_value'] = np.nan

    geo_box = coord.box_pixel2geo(pix_box)
    data_box = (0, 0, width, length)
    print(f'data   range in (x0,y0,x1,y1): {data_box}')
    print(f'subset range in (x0,y0,x1,y1): {pix_box}')
    print(f'data   range in (W, N, E, S): {coord.box_pixel2geo(data_box)}')
    print(f'subset range in (W, N, E, S): {geo_box}')

    if pix_box == data_box:
        print('Subset range == data coverage, no need to subset. Skip.')
        return fname

    # Calculate Subset/Overlap Index
    pix_box4data, pix_box4subset = get_box_overlap_index(data_box, pix_box)

    ###########################  Data Read and Write  ######################
    # Output File Name
    if not out_file:
        if os.getcwd() == os.path.dirname(os.path.abspath(fname)):
            if 'tight' in subset_dict.keys() and subset_dict['tight']:
                fbase, fext = os.path.splitext(fname)
                out_file = f'{fbase}_tight{fext}'
            else:
                out_file = 'sub_'+os.path.basename(fname)
        else:
            out_file = os.path.basename(fname)
    print('writing >>> '+out_file)

    # update metadata
    atr = attr.update_attribute4subset(atr, pix_box)

    # subset datasets one by one
    dsNames = readfile.get_dataset_list(fname)

    in_ext = os.path.splitext(fname)[1]
    out_ext = os.path.splitext(out_file)[1]
    if in_ext in ['.h5', '.he5']:

        # initiate the output file
        if out_ext in ['.h5', '.he5']:
            writefile.layout_hdf5(out_file, metadata=atr, ref_file=fname)
        else:
            dsDict = dict()

        # subset dataset one-by-one
        for dsName in dsNames:
            with h5py.File(fname, 'r') as fi:
                ds = fi[dsName]
                ds_shape = ds.shape
                ds_ndim = ds.ndim
                print('cropping {d} in {b} from {f} ...'.format(
                    d=dsName,
                    b=pix_box4data,
                    f=os.path.basename(fname)))

                if ds_ndim == 2:
                    # read
                    data = ds[pix_box4data[1]:pix_box4data[3],
                              pix_box4data[0]:pix_box4data[2]]

                    # crop
                    data_out = np.ones((pix_box[3] - pix_box[1],
                                        pix_box[2] - pix_box[0]),
                                       data.dtype) * subset_dict['fill_value']
                    data_out[pix_box4subset[1]:pix_box4subset[3],
                             pix_box4subset[0]:pix_box4subset[2]] = data
                    data_out = np.array(data_out, dtype=data.dtype)

                    # write each dataset to HDF5 file
                    # OR save each dataset for binary file
                    if out_ext in ['.h5', '.he5']:
                        block = [0, int(atr['LENGTH']), 0, int(atr['WIDTH'])]
                        writefile.write_hdf5_block(
                            out_file,
                            data=data_out,
                            datasetName=dsName,
                            block=block,
                            print_msg=True)
                    else:
                        dsDict[dsName] = data_out

                elif ds_ndim == 3:
                    # 3D dataset is not supported in binary
                    if out_ext not in ['.h5', '.he5']:
                        raise ValueError(f'Writing 3D dataset {dsName} into binary file is NOT supported!')

                    prog_bar = ptime.progressBar(maxValue=ds_shape[0])
                    for i in range(ds_shape[0]):
                        # read
                        data = ds[i,
                                  pix_box4data[1]:pix_box4data[3],
                                  pix_box4data[0]:pix_box4data[2]]

                        # crop
                        data_out = np.ones((1,
                                            pix_box[3] - pix_box[1],
                                            pix_box[2] - pix_box[0]),
                                           data.dtype) * subset_dict['fill_value']
                        data_out[:,
                                 pix_box4subset[1]:pix_box4subset[3],
                                 pix_box4subset[0]:pix_box4subset[2]] = data

                        # write
                        block = [i, i+1, 0, int(atr['LENGTH']), 0, int(atr['WIDTH'])]
                        writefile.write_hdf5_block(
                            out_file,
                            data=data_out,
                            datasetName=dsName,
                            block=block,
                            print_msg=False)

                        prog_bar.update(i+1, suffix=f'{i+1}/{ds_shape[0]}')
                    prog_bar.close()
                    print(f'finished writing to file: {out_file}')

        # write to binary file
        if out_ext not in ['.h5', '.he5']:
            writefile.write(dsDict, out_file=out_file, metadata=atr)

    else:
        # binary --> binary / hdf5 file
        dsDict = dict()
        for dsName in dsNames:
            dsDict[dsName] = subset_dataset(
                fname,
                dsName,
                pix_box,
                pix_box4data,
                pix_box4subset,
                fill_value=subset_dict['fill_value'],
            )

        atr['BANDS'] = len(dsDict.keys())
        writefile.write(dsDict, out_file=out_file, metadata=atr, ref_file=fname)

        # write extra metadata files for ISCE2 binary data files
        if out_ext not in ['.h5', '.he5'] and (os.path.isfile(fname+'.xml') or os.path.isfile(fname+'.aux.xml')):
            writefile.write_isce_xml(atr, out_file)

    return out_file


def read_aux_subset2inps(inps):
    # Convert All Inputs into subset_y/x/lat/lon
    # Input Priority: subset_y/x/lat/lon > reference > template > tight
    if all(not i for i in [inps.subset_x,
                           inps.subset_y,
                           inps.subset_lat,
                           inps.subset_lon]):
        # 1. Read subset info from Reference File
        if inps.reference:
            ref_atr = readfile.read_attribute(inps.reference)
            pix_box, geo_box = get_coverage_box(ref_atr)
            print('using subset info from '+inps.reference)

        # 2. Read subset info from template options
        elif inps.template_file:
            pix_box, geo_box = read_subset_template2box(inps.template_file)
            print('using subset info from '+inps.template_file)

        # 3. Use subset from tight info
        elif inps.tight:
            inps.lookup_file = ut.get_lookup_file(inps.lookup_file)
            if not inps.lookup_file:
                raise Exception('No lookup file found! Can not use --tight option without it.')

            atr_lut = readfile.read_attribute(inps.lookup_file)
            coord = ut.coordinate(atr_lut)
            if 'Y_FIRST' in atr_lut.keys():
                rg_lut = readfile.read(inps.lookup_file, datasetName='range')[0]
                rg_unique, rg_pos = np.unique(rg_lut, return_inverse=True)
                idx_row, idx_col = np.where(rg_lut != rg_unique[np.bincount(rg_pos).argmax()])
                pix_box = (np.min(idx_col) - 10, np.min(idx_row) - 10,
                           np.max(idx_col) + 10, np.max(idx_row) + 10)
                geo_box = coord.box_pixel2geo(pix_box)
                del rg_lut

            else:
                lat = readfile.read(inps.lookup_file, datasetName='latitude')[0]
                lon = readfile.read(inps.lookup_file, datasetName='longitude')[0]
                geo_box = (np.nanmin(lon), np.nanmax(lat),
                           np.nanmax(lon), np.nanmin(lat))
                pix_box = None
                del lat, lon
        else:
            raise Exception('No subset inputs found!')

        # Update subset_y/x/lat/lon
        inps = subset_box2inps(inps, pix_box, geo_box)

    return inps
