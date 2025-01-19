############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, 2013               #
############################################################


import os
import shutil
import time

import numpy as np
from skimage.transform import resize

from mintpy.objects import (
    IFGRAM_DSET_NAMES,
    cluster,
    giantTimeseries,
    ifgramStack,
    timeseries,
)
from mintpy.utils import ptime, readfile, time_func, writefile


#####################################################################################
def check_reference(atr1, atr2):
    """Check reference date and point
    Parameters: atr1/2   - dict, metadata of file1/2
    Returns:    ref_date - str, None for re-referencing in time  is NOT needed
                ref_y/x  - int, None for re-referencing in space is NOT needed
    """
    # 1. reference date
    # if same, do nothing
    # if different, use the 1st one as the reference
    ref_date1 = atr1.get('REF_DATE', None)
    ref_date2 = atr2.get('REF_DATE', None)
    if ref_date1 == ref_date2:
        ref_date = None
    else:
        ref_date = ref_date1

    # 2. reference point
    # if same, do nothing
    # if different, use the 1st one as the reference
    ref_yx1 = [atr1.get('REF_Y', None), atr1.get('REF_X', None)]
    ref_yx2 = [atr2.get('REF_Y', None), atr2.get('REF_X', None)]
    if ref_yx1 == ref_yx2:
        ref_y, ref_x = None, None
    else:
        ref_y, ref_x = ref_yx1

    # ensure ref_y/x are integer
    ref_y = int(ref_y) if ref_y is not None else None
    ref_x = int(ref_x) if ref_x is not None else None

    return ref_date, ref_y, ref_x


def diff_timeseries(file1, file2, out_file, force_diff=False, max_num_pixel=2e8):
    """Calculate the difference between two time-series files.
    file1.shape and file2.shape are different.

    Parameters: file1         - str, path of file1
                file2         - str, path of file2
                out_file      - str, path of output file
                force_diff    - bool, overwrite existing output file
                max_num_pixel - float, maximum number of pixels for each block
    Returns:    out_file      - str, path of output file
    """

    # basic info
    atr1 = readfile.read_attribute(file1)
    atr2 = readfile.read_attribute(file2)
    k1 = atr1['FILE_TYPE']
    k2 = atr2['FILE_TYPE']
    length1, width1 = int(atr1['LENGTH']), int(atr1['WIDTH'])
    length2, width2 = int(atr2['LENGTH']), int(atr2['WIDTH'])
    date_list1 = timeseries(file1).get_date_list()
    if k2 == 'timeseries':
        date_list2 = timeseries(file2).get_date_list()
        unit_fac = 1.
    elif k2 == 'giantTimeseries':
        date_list2 = giantTimeseries(file2).get_date_list()
        unit_fac = 0.001

    # check file size
    different_size = False
    if length1 != length2 or width1 != width2:
        different_size = True
        kwargs = dict(
            output_shape=(length1, width1),
            order=1,
            mode='constant',
            anti_aliasing=True,
            preserve_range=True,
        )
        print('WARNING: file 1/2 have different sizes:')
        print(f'    file 1: ({atr1["LENGTH"]}, {atr1["WIDTH"]})')
        print(f'    file 2: ({atr2["LENGTH"]}, {atr2["WIDTH"]})')
    if different_size and not force_diff:
        raise Exception('To enforce the differencing anyway, use --force option.')

    # check reference date / point
    ref_date, ref_y, ref_x = check_reference(atr1, atr2)
    if ref_date:
        ref_data = readfile.read(file2, datasetName=ref_date, resize2shape=(length1, width1))[0]
        if different_size:
            ref_data = resize(ref_data, **kwargs)

    # check dates shared by two timeseries files
    date_list_shared = [i for i in date_list1 if i in date_list2]
    date_flag_shared = np.ones((len(date_list1)), dtype=np.bool_)
    if date_list_shared != date_list1:
        print(f'WARNING: {file2} does not contain all dates in {file1}')
        if force_diff:
            date_list_ex = list(set(date_list1) - set(date_list_shared))
            print('Continue and enforce the differencing for their shared dates only.')
            print(f'\twith following dates are ignored for differencing:\n{date_list_ex}')
            date_flag_shared[np.array([date_list1.index(i) for i in date_list_ex])] = 0
        else:
            raise Exception('To enforce the differencing anyway, use --force option.')

    # get reference matrix
    if ref_y and ref_x:
        ref_box = (ref_x, ref_y, ref_x + 1, ref_y + 1)
        ref_val = []
        for date in date_list_shared:
            ref_val.append(readfile.read(file2, datasetName=date, box=ref_box,resize2shape=(length1, width1))[0])
        ref_val = np.array(ref_val)* unit_fac
    else:
        ref_val = None

    # resample data2
    data2_resample = []
    for date in date_list_shared:
        data2_resample.append(readfile.read(file2, datasetName=date, resize2shape=(length1, width1))[0])
    data2_resample = np.array(data2_resample)* unit_fac

    # instantiate the output file
    writefile.layout_hdf5(out_file, ref_file=file1)

    # block-by-block IO
    length, width = int(atr1['LENGTH']), int(atr1['WIDTH'])
    num_box = int(np.ceil(len(date_list1) * length * width / max_num_pixel))
    box_list, num_box = cluster.split_box2sub_boxes(
        box=(0, 0, width, length),
        num_split=num_box,
        dimension='y',
        print_msg=True,
    )

    for i, box in enumerate(box_list):
        if num_box > 1:
            print(f'\n------- processing patch {i+1} out of {num_box} --------------')
            print(f'box: {box}')

        # read data2 (consider different reference_date/pixel)
        print(f'read from file: {file2}')
        data2 = data2_resample[:,box[1]:box[3],box[0]:box[2]]

        if ref_val is not None:
            print(f'* referencing data from {os.path.basename(file2)} to y/x: {ref_y}/{ref_x}')
            data2 -= np.tile(ref_val.reshape(-1, 1, 1), (1, data2.shape[1], data2.shape[2]))

        if ref_date:
            print(f'* referencing data from {os.path.basename(file2)} to date: {ref_date}')
            ref_ind = date_list_shared.index(ref_date)
            data2 -= np.tile(data2[ref_ind, :, :], (data2.shape[0], 1, 1))

        # read data1
        print(f'read from file: {file1}')
        data = readfile.read(file1, box=box)[0]

        # apply differencing
        mask = data == 0.
        data[date_flag_shared] -= data2
        data[mask] = 0.                   # Do not change zero phase value
        del data2

        # write the block
        block = [0, data.shape[0], box[1], box[3], box[0], box[2]]
        writefile.write_hdf5_block(out_file, data=data, datasetName=k1, block=block)

    return out_file


def diff_timeseries_and_velocity(file1, file2, out_file, max_num_pixel=2e8):
    """Calculate the difference between a time-series file and a velocity file.

    Parameters: file1         - str, path of file1 (time series)
                file2         - str, path of file2 (velocity)
                out_file      - str, path of output file
                max_num_pixel - float, maximum number of pixels for each block
    Returns:    out_file      - str, path of output file
    """

    # basic info
    atr1 = readfile.read_attribute(file1)
    atr2 = readfile.read_attribute(file2)
    date_list = timeseries(file1).get_date_list()
    num_date = len(date_list)

    # check reference point
    _, ref_y, ref_x = check_reference(atr1, atr2)

    if ref_y and ref_x:
        ref_box = (ref_x, ref_y, ref_x + 1, ref_y + 1)
        ref_val = readfile.read(file2, datasetName='velocity', box=ref_box)[0]
    else:
        ref_val = None

    # check dataset names in the time-func file
    ds_names = readfile.get_dataset_list(file2)
    ds_names = [x for x in ds_names if not x.endswith('Std')]
    if 'velocity' not in ds_names:
        raise ValueError(f'No velocity dataset found in file2: {file2}!')
    if ds_names != ['velocity']:
        print('WARNING: ONLY velocity is supported, ignore the following datasets and continue:')
        print([x for x in ds_names if x != 'velocity'])

    # instantiate the output file
    writefile.layout_hdf5(out_file, ref_file=file1)

    # block-by-block IO
    length, width = int(atr1['LENGTH']), int(atr1['WIDTH'])
    num_box = int(np.ceil(len(date_list) * length * width / max_num_pixel))
    box_list, num_box = cluster.split_box2sub_boxes(
        box=(0, 0, width, length),
        num_split=num_box,
        dimension='y',
        print_msg=True,
    )

    for i, box in enumerate(box_list):
        box_wid = box[2] - box[0]
        box_len = box[3] - box[1]
        num_pixel = box_len * box_wid
        if num_box > 1:
            print(f'\n------- processing patch {i+1} out of {num_box} --------------')
            print(f'box: {box}')

        ## Re-construct the time series from the time-func file #########
        #   here is a crude option, m to be only the linear function
        #   To-do: need a proper new function to get m = timeseries2velocity.hdf5_dataset2model()

        # read file2 (consider different reference pixel)
        print(f'read velocity from file2: {file2}')
        velo = readfile.read(file2, datasetName='velocity', box=box)[0]

        if ref_val is not None:
            print(f'* referencing velocity to y/x: {ref_y}/{ref_x} with value of {ref_val*100:.2f} cm/year')
            velo -= ref_val

        # calculate design matrix from the time-func file
        model = {'polynomial' : 1}
        G_fit = time_func.get_design_matrix4time_func(date_list, model=model)

        print(f'* reconstructing time-series from {os.path.basename(file2)} with model {model}')
        m = np.vstack([np.zeros(num_pixel), velo.flatten()])
        ts_fit = np.matmul(G_fit, m)
        data2 = ts_fit.reshape(-1, box_len, box_wid)

        ###################################################################

        if 'REF_DATE' in atr1.keys():
            print(f'* referencing time-series from file2: {os.path.basename(file2)} to date: {atr1["REF_DATE"]}')
            ref_ind = date_list.index(atr1["REF_DATE"])
            data2 -= np.tile(data2[ref_ind, :, :], (num_date, 1, 1))

        # read data1
        print(f'read time-series from file1: {file1}')
        data = readfile.read(file1, box=box)[0]

        # apply differencing
        mask = data == 0.
        data -= data2
        data[mask] = 0.               # Do not change zero phase value
        del data2

        # write the block
        block = [0, num_date, box[1], box[3], box[0], box[2]]
        writefile.write_hdf5_block(out_file, data=data, datasetName='timeseries', block=block)

    return out_file


def diff_ifgram_stack(file1, file2, out_file):
    """Calculate the difference between two ifgramStack files.

    Parameters: file1    - str, path of file1
                file2    - str, path of file2
                out_file - str, path of output file
    Returns:    out_file - str, path of output file
    """

    obj1 = ifgramStack(file1)
    obj1.open()
    obj2 = ifgramStack(file2)
    obj2.open()
    ds_names = list(set(obj1.datasetNames) & set(obj2.datasetNames))
    if len(ds_names) == 0:
        raise ValueError('no common dataset between two files!')
    ds_name = [i for i in IFGRAM_DSET_NAMES if i in ds_names][0]

    # read data
    print(f'reading {ds_name} from file {file1} ...')
    data1 = readfile.read(file1, datasetName=ds_name)[0]
    print(f'reading {ds_name} from file {file2} ...')
    data2 = readfile.read(file2, datasetName=ds_name)[0]

    # consider reference pixel
    if 'unwrapphase' in ds_name.lower():
        print(f'referencing to pixel ({obj1.refY},{obj1.refX}) ...')
        ref1 = data1[:, obj1.refY, obj1.refX]
        ref2 = data2[:, obj2.refY, obj2.refX]
        for i in range(data1.shape[0]):
            data1[i,:][data1[i, :] != 0.] -= ref1[i]
            data2[i,:][data2[i, :] != 0.] -= ref2[i]

    # operation and ignore zero values
    data1[data1 == 0] = np.nan
    data2[data2 == 0] = np.nan
    data = data1 - data2
    del data1, data2
    data[np.isnan(data)] = 0.

    # write to file
    ds_dict = {ds_name : data}
    writefile.write(ds_dict, out_file=out_file, ref_file=file1)

    return out_file


def diff_ifgram_and_timeseries(unw_file, ts_file, cor_file):
    """Calculate the difference between two unwrapped interferogram files.

    Parameters: unw_file - str, path of the interferogram file
                ts_file  - str, path of the time-series file, e.g. ERA5.h5, SET.h5
                cor_file - str, path of output corrected interferogram file
    Returns:    cor_file - str, path of output corrected interferogram file
    """

    atr = readfile.read_attribute(unw_file)
    dunit = atr.get('UNIT', 'radian')
    dname = 'phase' if dunit.startswith('rad') else ''

    # read data
    print(f'read {dname} from {unw_file}')
    data, atr = readfile.read(unw_file, datasetName=dname)
    date1, date2 = ptime.yyyymmdd(atr['DATE12'].split('-'))

    # read the correction
    print(f'calc {dname} for {date1}-{date2} from {ts_file}')
    delay  = readfile.read(ts_file, datasetName=date2)[0]
    delay -= readfile.read(ts_file, datasetName=date1)[0]
    if dunit.startswith('rad'):
        print(f'convert {dname} from radian to meter')
        delay *= -4. * np.pi / float(atr['WAVELENGTH'])

    # apply the correction (and re-referencing)
    data -= delay
    if 'REF_Y' in atr.keys():
        ref_y, ref_x = int(atr['REF_Y']), int(atr['REF_X'])
        data -= data[ref_y, ref_x]
        print(f're-referencing to pixel ({ref_y}, {ref_x})')

    if atr['FILE_TYPE'] == '.unw':
        print(f'read magnitude from {unw_file}')
        mag = readfile.read(unw_file, datasetName='magnitude')[0]
        ds_dict = {'magnitude': mag, 'phase': data}
    else:
        ds_dict = data

    print(f'write corrected data to {cor_file}')
    writefile.write(ds_dict, cor_file, atr)

    # prepare ISCE metadata file by
    # 1. copy and rename metadata files
    # 2. update file path inside files
    for ext in ['xml', 'vrt']:
        unw_meta_file = f'{unw_file}.{ext}'
        cor_meta_file = f'{cor_file}.{ext}'

        if os.path.isfile(unw_meta_file):
            # copy
            shutil.copy2(unw_meta_file, cor_meta_file)
            print(f'copy {unw_meta_file} to {cor_meta_file} and update the filename')

            # update file path
            meta_file = f'{cor_file}.{ext}'
            with open(meta_file) as f:
                s = f.read()
            s = s.replace(os.path.basename(unw_file),
                          os.path.basename(cor_file))
            with open(meta_file, 'w') as f:
                f.write(s)

    return cor_file


def diff_file(file1, file2, out_file, force_diff=False, max_num_pixel=2e8):
    """calculate/write file1 - file2

    Parameters: file1         - str, path of file1
                file2         - list(str), path of file2(s)
                out_file      - str, path of output file
                force_diff    - bool, overwrite existing output file
                max_num_pixel - float, maximum number of pixels for each block
    """
    start_time = time.time()
    print(f'{file1} - {file2} --> {out_file}')

    # Read basic info
    atr1 = readfile.read_attribute(file1)
    atr2 = readfile.read_attribute(file2[0])
    k1 = atr1['FILE_TYPE']
    k2 = atr2['FILE_TYPE']
    print(f'the 1st input file is: {k1}')

    if k1 == 'timeseries':
        if k2 not in ['timeseries', 'giantTimeseries', 'velocity']:
            print('If the first file is timeseries, the following file must be either timeseries or velocity.')
            raise Exception('Input multiple dataset files are not the same file type!')
        if k2 in ['timeseries', 'giantTimeseries']:
            diff_timeseries(file1, file2[0], out_file, force_diff, max_num_pixel)
        elif k2 == 'velocity':
            diff_timeseries_and_velocity(file1, file2[0], out_file, max_num_pixel)

    elif all(i == 'ifgramStack' for i in [k1, k2]):
        diff_ifgram_stack(file1, file2[0], out_file)

    elif k1 in ['.unw', 'displacement', '.off'] and k2 == 'timeseries':
        diff_ifgram_and_timeseries(unw_file=file1, ts_file=file2[0], cor_file=out_file)

    else:
        # get common dataset list
        ds_names_list = [readfile.get_dataset_list(x) for x in [file1] + file2]
        ds_names = list(set.intersection(*map(set, ds_names_list)))
        # if all files have one dataset, ignore dataset name variation and take the 1st one as reference
        if all(len(x) == 1 for x in ds_names_list):
            ds_names = ds_names_list[0]
        print('List of common datasets across files: ', ds_names)
        if len(ds_names) < 1:
            raise ValueError(f'No common datasets found among files:\n{[file1] + file2}')

        # loop over each dataset
        dsDict = {}
        for ds_name in ds_names:
            print(f'differencing {ds_name} ...')
            data = readfile.read(file1, datasetName=ds_name)[0]
            dtype = data.dtype

            # loop over each file2
            for i, fname in enumerate(file2):
                # ignore ds_name if input file has single dataset
                ds_name2read = None if len(ds_names_list[i+1]) == 1 else ds_name
                # read
                data2 = readfile.read(fname, datasetName=ds_name2read)[0]
                # do the referencing for velocity files
                if ds_name == 'velocity':
                    ref_y, ref_x = check_reference(atr1, atr2)[1:]
                    if ref_y and ref_x:
                        print(f'* referencing data from {os.path.basename(file2[0])} to y/x: {ref_y}/{ref_x}')
                        data2 -= data2[ref_y, ref_x]
                # convert to float32 to apply the operation because some types, e.g. bool, do not support it.
                # then convert back to the original data type
                data = np.array(data, dtype=np.float32) - np.array(data2, dtype=np.float32)

            # save data in the same type as the 1st file
            dsDict[ds_name] = np.array(data, dtype=dtype)

        # output
        print(f'use metadata from the 1st file: {file1}')
        writefile.write(dsDict, out_file=out_file, metadata=atr1, ref_file=file1)

    # used time
    m, s = divmod(time.time()-start_time, 60)
    print(f'time used: {m:02.0f} mins {s:02.1f} secs')

    return out_file
