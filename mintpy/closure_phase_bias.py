#!/usr/bin/env python
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Yujie Zheng, Feb 2022                            #
############################################################


import os
import sys
import time
import argparse
import numpy as np
import glob
from datetime import datetime as dt

from mintpy.objects import ifgramStack, cluster
from mintpy.utils import arg_group, ptime, readfile, writefile, isce_utils
from mintpy import ifgram_inversion as ifginv



################################################################################
REFERENCE = """reference:
  Y. Zheng, H. Fattahi, P. Agram, M. Simons and P. Rosen, (2022). On Closure Phase
    and Systematic Bias in Multi-looked SAR Interferometry, in IEEE Trans. Geosci.
    Remote Sens., doi:10.1109/TGRS.2022.3167648.
"""

EXAMPLE = """example:
  # create mask for areas suseptible to biases
  closure_phase_bias.py -i inputs/ifgramStack.h5 --nl 20 -a mask
  closure_phase_bias.py -i inputs/ifgramStack.h5 --nl 20 -a mask --num-sigma 2.5

  # estimate and correct for biases
  closure_phase_bias.py -i inputs/ifgramStack.h5 --nl 20 --bw 10 -a quick_estimate
  closure_phase_bias.py -i inputs/ifgramStack.h5 --nl 20 --bw 10 -a estimate --noupdate -c local
"""

def create_parser():
    parser = argparse.ArgumentParser(description = 'Phase non-closure related biases correction',
                                    formatter_class = argparse.RawTextHelpFormatter,
                                    epilog=REFERENCE+'\n'+EXAMPLE)

    parser.add_argument('-i','--ifgramstack', type=str, dest='stack_file',
                        help='interferogram stack file that contains the unwrapped phases')
    parser.add_argument('--nl','--conn-level', dest='nl', type=int, default=20,
                        help='connection level that we are correcting to (or consider as no bias)\n'
                             '(default: %(default)s)')
    parser.add_argument('--bw', dest='bw', type=int, default=10,
                        help='bandwidth of time-series analysis that you want to correct')
    parser.add_argument('-a','--action', dest='action', type=str, default='mask',
                        choices={'mask', 'quick_estimate', 'estimate'},
                        help='action to take (default: %(default)s):\n'
                             'mask           - create a mask of areas susceptible to closure phase errors\n'
                             'quick_estimate - quick and approximate estimation on how bias decays with time\n'
                             '                 output sequential closure phase files\n'
                             'estimate       - estimate how bias decays with time\n'
                             '                 processed for each pixel on a pixel by pixel basis [slow]')

    # mask configuration
    mask = parser.add_argument_group('Mask', 'Configuration for closure phase bias mask')
    mask.add_argument('--num-sigma', dest='num_sigma', type=float, default=3,
                      help='Threashold for phase, in number of sigmas (default: %(default)s).\n'
                           'Assuming a Gaussian distribution for the cumulative closure phase'
                           ' with sigma = pi / sqrt(3*num_cp)')
    mask.add_argument('--eps','--epsilon', dest='epsilon', type=float, default=0.3,
                      help='Threashold for the normalized amplitude in [0-1] (default: %(default)s).')

    # compute
    parser = arg_group.add_parallel_argument(parser)
    parser = arg_group.add_memory_argument(parser)

    # output
    parser.add_argument('--noupdate', dest='update_closure_phase', action='store_false',
                        help='Use when no need to compute closure phases')
    parser.add_argument('-o', dest='outdir', type=str, default='./', help='output file directory')

    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    return inps


#############################  Closure Phase  ##################################
def seq_closure_phase(stack_obj, box, conn_level):
    """Computes wrapped sequential closure phases for a given conneciton level.

    For conn_level=5, closure_phase = p12 + p23 + p34 + p45 + p56 - p16.

    Parameters: stack_obj  - ifgramStack object
                box        - tuple of 4 int, bounding box in (x0, y0, x1, y1)
                conn_level - int, connection level of the closure phase
                normalize  - bool, normalize the output complex magnitude by num_cp
    Returns:    cp_w       - 3D np.ndarray in float32 in size of (num_cp, box_len, box_wid)
                             wrapped sequential closure phases for the given connection level.
    """
    # basic info
    num_date = len(stack_obj.get_date_list(dropIfgram=True))
    box_wid = box[2] - box[0]
    box_len = box[3] - box[1]

    ## get the closure index
    cp_idx = stack_obj.get_closure_phase_index(conn_level=conn_level, dropIfgram=True)
    num_cp = cp_idx.shape[0]
    print(f'Number of closure measurements expected: {num_date - conn_level}')
    print(f'Number of closure measurements found   : {num_cp}')
    if num_cp < num_date-n:
        msg = f'num_cp ({num_cp}) < num_date-conn_level ({num_date-conn_level})'
        msg += ' --> some interferograms are missing!'
        raise Exception(msg)

    ## read data
    phase = readfile.read(stack_obj.file, box=box, print_msg=False)[0]
    ref_phase = stack_obj.get_reference_phase(dropIfgram=False)
    for i in range(phase.shape[0]):
        mask = phase[i] != 0.
        phase[i][mask] -= ref_phase[i]

    ## calculate cum seq closure phase
    cp_w = np.zeros((num_cp, box_len, box_wid), dtype=np.float32)
    for i in range(num_cp):

        # calculate closure phase - cp0_w
        idx_plus, idx_minor = cp_idx[i, :-1], cp_idx[i, -1]
        cp0_w = np.sum(phase[idx_plus])
        cp0_w -= phase[idx_minor]

        # get the wrapped closure phase
        cp_w[i] = np.angle(np.exp(1j * cp0_w))

    return cp_w


def sum_seq_closure_phase(stack_obj, box, conn_level, normalize=False):
    """Computes the sum of complex sequential closure phase for a given connection level.

    For conn_level=5, closure_phase = p12 + p23 + p34 + p45 + p56 - p16.

    Parameters: stack_obj  - ifgramStack object
                box        - tuple of 4 int, bounding box in (x0, y0, x1, y1)
                conn_level - int, connection level of the closure phase
                normalize  - bool, normalize the output complex magnitude by num_cp
    Returns:    cum_cp     - 2D np.ndarray in complex64 in (box_len, box_wid)
                             sum of sequential closure phase for the given connection level
                num_cp     - integer, number of closure phases used in the sum
    """

    # basic info
    num_date = len(stack_obj.get_date_list(dropIfgram=True))
    box_wid = box[2] - box[0]
    box_len = box[3] - box[1]

    ## get the closure index
    cp_idx = stack_obj.get_closure_phase_index(conn_level=conn_level, dropIfgram=True)
    num_cp = cp_idx.shape[0]
    print(f'Number of closure measurements expected: {num_date - conn_level}')
    print(f'Number of closure measurements found   : {num_cp}')
    if num_cp < 1:
        raise Exception(f"No triplets found at connection level: {conn_level}!")

    ## read data
    phase = readfile.read(stack_obj.file, box=box, print_msg=False)[0]
    ref_phase = stack_obj.get_reference_phase(dropIfgram=False)
    for i in range(phase.shape[0]):
        mask = phase[i] != 0.
        phase[i][mask] -= ref_phase[i]

    ## calculate cum seq closure phase
    cum_cp = np.zeros((box_len, box_wid), dtype=np.complex64)
    for i in range(num_cp):

        # calculate closure phase - cp0_w
        idx_plus, idx_minor = cp_idx[i, :-1], cp_idx[i, -1]
        cp0_w = np.sum(phase[idx_plus])
        cp0_w -= phase[idx_minor]

        # cumulative
        cum_cp = cum_cp + np.exp(1j * cp0_w)

    # normalize
    if normalize:
        cum_cp /= num_cp

    return cum_cp, num_cp



#################################  Mask  #######################################
def calc_closure_phase_mask(stack_file, nl, num_sigma=3, threshold_amp=0.3, outdir='./', max_memory=4.0):
    """Calculate a mask for areas most suseptible to biases.

    Parameters: stack_file        - str, path for ifgramStack.h5 file
                nl                - int, connection level at which we assume is bias-free
                num_sigma         - float, number of sigmas for computing phase threshold
                threshold_amp     - float, threshold of ampliutde of the cumulative sequential closure phase
                outdir            - str, directory of output files
                max_mermory       - float, maximum memory in GB for each patch processed
    Returns:    mask              - 2D np.ndarray of size (length, width) in boolean, 0 for areas suseptible to biases.
                                    Saved to file: maskClosurePhase.h5
                avg_closure_phase - 2D np.ndarray of size (length, width) in complex64, average cum. seq. closure phase
                                    Saved to file: avgCpxClosurePhase.h5
    """

    # basic info
    stack_obj = ifgramStack(stack_file)
    stack_obj.open()
    length, width = stack_obj.length, stack_obj.width
    date_list = stack_obj.get_date_list(dropIfgram=True)
    print(f'number of valid acquisitions: {len(date_list)}')
    print(f'start / end date: {date_list[0]} / {date_list[-1]}')
    print(f'length / width: {length} / {width}')

    # calculate the average complex closure phase
    # process block-by-block to save memory
    box_list, num_box = ifginv.split2boxes(stack_file, max_memory)
    avg_closure_phase = np.zeros([length,width], dtype=np.complex64)
    for i, box in enumerate(box_list):
        if num_box > 1:
            print('\n------- processing patch {} out of {} --------------'.format(i+1, num_box))
            print(f'box: {box}')
            print('box width:  {}'.format(box[2] - box[0]))
            print('box length: {}'.format(box[3] - box[1]))

        (avg_closure_phase[box[1]:box[3], box[0]:box[2]],
         num_cp) = sum_seq_closure_phase(
            stack_obj=stack_obj,
            box=box,
            conn_level=nl,
            normalize=True)

    # create mask
    mask = np.ones([length,width], dtype=bool)

    ## What is a good thredshold?
    # Assume that it's pure noise so that the phase is uniform distributed from -pi to pi.
    # The standard deviation of phase in each loop is:
    #     pi/sqrt(3)
    # (technically should be smaller because when forming loops there should be a reduction in phase variance)
    # The standard deviation of phase in cumulative wrapped closure phase is:
    #     pi/sqrt(3)/sqrt(num_cp) -- again another simplification assuming no correlation.
    # We use 3\delta as threshold -- 99.7% confidence
    threshold_pha = np.pi / np.sqrt(3 * num_cp) * num_sigma

    # mask areas with potential bias
    mask[np.abs(np.angle(avg_closure_phase)) > threshold_pha] = 0

    # unmask areas with low correlation
    # where it's hard to know wheter there is bias or not
    mask[np.abs(np.abs(avg_closure_phase) < threshold_amp)] = 1

    # write file 1 - mask
    out_file = os.path.join(outdir, 'maskClosurePhase.h5')
    meta = dict(stack_obj.metadata)
    meta['FILE_TYPE'] = 'mask'
    meta['DATA_TYPE'] = 'bool'
    ds_name_dict = {'mask': [np.bool_, (length, width), mask],}
    writefile.layout_hdf5(out_file, ds_name_dict, meta)

    # write file 2 - average closure phase
    out_file = os.path.join(outdir, 'avgCpxClosurePhase.h5')
    meta['DATA_TYPE'] = 'float32'
    ds_name_dict2 = {
        'phase'     : [np.float32, (length, width), np.angle(avg_closure_phase)],
        'amplitude' : [np.float32, (length, width), np.abs(avg_closure_phase)],
    }
    writefile.layout_hdf5(out_file, ds_name_dict2, meta)

    return mask, avg_closure_phase


################################################################################
def cum_seq_unw_closure_phase(conn_level, conn_dir, date_list, meta):
    '''Outpu cumulative con-n sequential closure phase in time-series format.

    Reference: Eq. 25 in Zheng et al., 2022, but divided by conn_level.

    Parameters: conn_level  - int, connection level of closure phases
                conn_dir    - str, path of sequential closure phases file for connection - n
                date_list   - list of str, SLC dates
                meta        - dict, metadata of ifgramStack.h5
    Returns:    bias_ts     - 3D np.ndarray in size of (num_ifgram, length, width) in float32,
                              cumulative sequential closure phase time series,
                              saved to file: conn{conn_level}_cumSeqClosurePhase.h5
                common_mask - 2D np.ndarray in size of (length, width) in bool,
                              mask based on common connected components,
                              saved to file: conn{conn_level}_maskConnComp.h5
    '''

    # output file
    cum_cp_file = os.path.join(conn_dir, f'conn{conn_level}_cumSeqClosurePhase.h5')
    mask_file = os.path.join(conn_dir, f'conn{conn_level}_maskConnComp.h5')

    if os.path.isfile(cum_cp_file) and os.path.isfile(mask_file):
        msg = 'Cumulative seq closure phase time series file (& its mask file) exist as below, skip re-generating.'
        msg += f'\n{cum_cp_file}\n{mask_file}'
        print(msg)
        return

    # basic info
    length, width = int(meta['LENGTH']), int(meta['WIDTH'])
    ref_y, ref_x = meta['REF_Y'], meta['REF_X']

    unw_files = sorted(glob.glob(os.path.join(conn_dir, '*.unw')))
    num_file = len(unw_files)

    print('calculate the cumulative seq closure phase time series ...')
    cp_phase = np.zeros((num_file, length, width), dtype=np.float32)
    mask = np.zeros((num_file, length, width), dtype=np.float32)

    prog_bar = ptime.progressBar(maxValue=num_file)
    for i, unw_file in enumerate(unw_files):
        prog_bar.update(i+1, suffix=os.path.basename(unw_file))

        unw = readfile.read(unw_file, datasetName='phase')[0]
        unw -= unw[ref_y, ref_x]
        cp_phase[i] = unw

        conn_comp_file = unw_file + '.conncomp'
        conn_comp = readfile.read(conn_comp_file)[0]
        mask[i] = np.where(conn_comp >= 1, 1, np.nan)

    prog_bar.close()

    # compute sequential closure phase
    num_date = len(date_list)
    bias_ts = np.zeros((num_date, length, width), dtype=np.float32)
    bias_ts[1:num_date-conn_level+1, :, :] = np.cumsum(cp_phase, 0)
    for i in range(num_date-conn_level+1, num_date):
        bias_ts[i] = (i - num_date + conn_level) * cp_phase[-1] + bias_ts[num_date - conn_level]
    bias_ts /= conn_level

    # write bias time series to HDF5 file
    ds_dict = {
        'timeseries' : [np.float32,     (num_date, length, width), bias_ts],
        'date'       : [np.dtype('S8'), (num_date), np.array(date_list, np.string_)],
    }
    meta['FILE_TYPE'] = 'timeseries'
    writefile.layout_hdf5(cum_cp_file, ds_dict, metadata=meta)

    # write mask to HDF5 file
    common_mask = np.where(np.isnan(np.sum(mask,0)), False, True)
    meta['FILE_TYPE'] = 'mask'
    writefile.write(common_mask, out_file=mask_file, metadata=meta)

    return bias_ts, common_mask


def compute_unwrap_closure_phase(stack_file, conn_level, max_memory, outdir):
    '''Output the following phase time-sseries of connection-conn:

    + wrapped
    + unwrapped sequential closure phases
    + cumulative closure phase
    Output directory: outdir/closurePhase/conn{conn_level}_cp

    Parameters: stack_file  - str, path for ifgramStack.h5
                conn_level  - int, connection level
                max_mermory - float, maximum memory in GB for each patch processed
                outdir      - str, path for output files
    Returns:    various wrapped, unwrapped and cumulative closure phase time-series

    '''
    # basic info
    stack_obj = ifgramStack(stack_file)
    stack_obj.open()
    length, width = stack_obj.length, stack_obj.width
    meta = dict(stack_obj.metadata)
    print(f'scene length x width: {length} x {width}')

    date_list = stack_obj.get_date_list(dropIfgram=False)
    num_date = len(date_list)
    print(f'number of acquisitions found: {num_date}')
    print(f'start / end date: {date_list[0]} / {date_list[-1]}')
    # number of expected closure phase
    num_cp = num_date - conn_level

    # process block-by-block
    # split igram_file into blocks to save memory
    box_list, num_box = ifginv.split2boxes(stack_file,max_memory)
    closure_phase = np.zeros([num_cp, length, width],np.float32)
    for i, box in enumerate(box_list):
        print(box)
        if num_box > 1:
            print('\n------- processing patch {} out of {} --------------'.format(i+1, num_box))
            print('box length: {}'.format(box[3] - box[1]))
            print('box width : {}'.format(box[2] - box[0]))

        closure_phase[:, box[1]:box[3], box[0]:box[2]] = seq_closure_phase(
            stack_obj,
            box=box,
            conn_level=conn_level)


    ## filter the closure phase and re-unwrap

    # output directory
    conn_dir = os.path.join(cpdir, f'closurePhase/conn{conn_level}_cp')
    os.makedirs(conn_dir, exist_ok=True)

    kernel = isce_utils.gaussian_kernel(5, 5, 1, 1)
    for i in range(num_cp):
        # some day we will need to make this 4 digits.
        int_file = os.path.join(conn_dir, f'conn{conn_level}_filt_{i:03}.int')
        cor_file = os.path.join(conn_dir, f'conn{conn_level}_filt_{i:03}.cor')
        unw_file = os.path.join(conn_dir, f'conn{conn_level}_filt_{i:03}.unw')

        if not os.path.isfile(int_file):
            # filter the closure phase interferogram
            closure_phase_filt = isce_utils.convolve(
                data=np.exp(1j*closure_phase[i]),
                kernel=kernel)

            # write to binary file in isce2 format
            with open(int_file, mode='wb') as fid:
                closure_phase_filt.tofile(fid)

            # write metadata in isce2 format
            meta['FILE_TYPE']='.int'
            meta['INTERLEAVE']='BIP'
            meta['DATA_TYPE']='complex64'
            meta['BANDS']=1
            writefile.write_isce_xml(meta, int_file)

        if not os.path.isfile(cor_file):
            isce_utils.estimate_coherence(int_file, cor_file)

        if not os.path.isfile(unw_file):
            isce.unwrap_snaphu(int_file, cor_file, unw_file, meta)


    ## output accumulated unwrapped closure phase time-series
    cum_seq_unw_closure_phase(conn_level, conn_dir, date_list, meta)

    return


################################################################################
def seq2cum_closure_phase(conn, outdir, box):
    '''Read cumulative sequential closure phase from individual closure phase directory.

    Reference: Eq. (25) in Zheng et al. (2022).

    Parameters: conn    - integer, connection level of sequential closure phases
                outdir  - string, directory of conn_seqcumclosurephase.h5
                box     - list in size of (4,) in integer, coordinates of bounding box
    Returns:    biasts  - 3D array in size of (num_date, box_lengh, box_width) in float,
                          cumulative sequential closure phases
    '''
    dname = f'conn{conn}_cp'
    fname = f'conn{conn}_seqcumclosurephase.h5'
    seq_cp_file = os.path.join(outdir, 'closurePhase', dname, fname)
    bias_ts = readfile.read(seq_cp_file, box=box, print_msg=False)[0]
    return bias_ts


def estimate_ratioX(tbase, n, nl, wvl, box, outdir, mask=False):
    ''' This script estimates w(n\delta_t)/w(delta_t), Eq.(29) in Zheng et al., 2022

    Parameters: tbase           - list in size of (num_date,) in float, time in accumulated years
                n               - integer, connection-level
                nl              - integer, minimum connection-level that we think is bias-free
                wvl             - float, wavelength
                box             - list in size of (4,) in integer, coordinates of bounding box
                outdir          - string, the working directory
                mask            - bool, whether to mask out areas with average bias velocity less than 1 mm/year
    Returns:    wratio          - 2D array of size (box_length, box_width) in float, w(n\delta_t)/w(delta_t), Eq.(29) in Zheng et al., 2022
                wratio_velocity - 2D array of size (box_length, box_width) in float, bias-velocity at n*delta_t temporal baseline
    '''
    box_width  = box[2] - box[0]
    box_length = box[3] - box[1]
    cum_bias_conn_1 = seq2cum_closure_phase(nl, outdir, box)[-1,:,:]
    coef = -4*np.pi/wvl
    delta_T = tbase[-1]-tbase[0]
    vel_bias_conn1 = cum_bias_conn_1/coef/delta_T

    if n==1:
        wratio = np.ones([box_length, box_width])
        wratio_velocity = np.multiply(wratio,vel_bias_conn1)
        if mask:
            # if average velocity smaller than 1 mm/year (hardcoded here), mask out for better visual
            # this option is only turned on while outputint Wratio.h5 file.
            wratio[abs(vel_bias_conn1)<0.1]=np.nan
    else:
        cum_bias_conn_n =  seq2cum_closure_phase(n, outdir,box)[-1,:,:]
        wratio = np.divide(cum_bias_conn_n,cum_bias_conn_1)
        wratio = 1-wratio
        wratio[wratio>1]=1
        wratio[wratio<0]=0
        wratio_velocity = np.multiply(wratio,vel_bias_conn1)
        if mask:
            # if average velocity smaller than 1 mm/year (hardcoded here), mask out for better visual
            wratio[abs(vel_bias_conn1) < 0.1] = np.nan

    return wratio,wratio_velocity # wratio is a length by width 2D matrix


def estimate_ratioX_all(bw,nl,outdir,box):
    ''' Estimate w(n\delta_t)/w(delta_t) for n=1:bw

    Parameters: nl     - integer, minimum connection-level that we think is bias-free
                bw     - integer, bandwidth of given time-sereis analysis
                box    - list in size of (4,) in integer, coordinates of bounding box
                outdir - string, the working directory
    Returns:    wratio - 3D array in size of (bw+1, length, width) in float, 
                         the first slice (w[0,:,:]) is a padding 
                         to ensure that wratio[n,:,:]=w(n\delta_t)/w(delta_t).
    '''
    box_width  = box[2] - box[0]
    box_length = box[3] - box[1]
    cum_bias_conn_1 = seq2cum_closure_phase(nl, outdir, box)[-1,:,:]
    wratio = np.zeros([bw+1,box_length, box_width], dtype = np.float32)
    for n in np.arange(2,bw+1):
        cum_bias_conn_n =  seq2cum_closure_phase(n, outdir, box)[-1,:,:]
        wratio[n,:,:] = np.divide(cum_bias_conn_n,cum_bias_conn_1)

    wratio = 1-wratio
    wratio[wratio>1]=1
    wratio[wratio<0]=0
    return wratio


def get_design_matrix_W(M, A, bw, box, tbase, nl, outdir):
    ''' computes the matrix W (Eq. 16 in Zheng et al., 2022) for a bounding box.

    Parameters: M      - integer, number of interferograms
                A      - 2D array in size of (M, num_date) in integer, design matrix specifying SAR acquisitions used
                bw     - integer, bandwidth of time-series analysis
                tbase  - list in size of (num_date,) in float, time in accumulated years
                nl     - integer, minimum connection-level that we think is bias-free
                box    - list in size of (4,) in integer, coordinates of bounding box
                outdir - string, the working directory
    Returns:    W      - 2D array in size of (numpix, M) in float, 
                         each row stores the diagnal component of W (Eq. 16 in Zheng et al., 2022) for one pixel.
    '''
    box_width  = box[2] - box[0]
    box_length = box[3] - box[1]
    numpix = box_width * box_length
    # intial output value
    W = np.zeros((numpix,M),dtype = np.float32)
    wratioall = estimate_ratioX_all(bw, nl, outdir, box)
    for i in range(M):
        Aline = list(A[i,:])
        idx1 = Aline.index(-1)
        idx2 = Aline.index(1)
        conn = idx2 - idx1
        if conn > bw:
            print('Interferograms with maximum connection level larger than input bandwidth exists in ifgramStack.h5, '
                  'use modify_network.py to adjust the maximum connection level')
        wratio = wratioall[conn,:,:]
        wratio = wratio.reshape(-1)
        W[:,i] = wratio

    return W


def average_temporal_span(date_ordinal,bw):
    '''compute average temporal span (days) for interferogram subsets chosen for limited bandwidth analysis

    Parameters:     date_ordinal        - list of size (num_date,) in integer, time in days
                    bw                  - integer, bandwidth of time-series analysis
    Return：        avgtime             - float, average time-span in days for interferograms subsets of bandwith bw.
    '''
    avgtime = 0
    numigram = 0
    for level in range(1, bw+1):
        slcdate_firstn = date_ordinal[0:level]
        slcdate_lastn = date_ordinal[-level:]
        for i in range(level):
            avgtime = avgtime + slcdate_lastn[i] - slcdate_firstn[i]
        numigram = numigram + len(date_ordinal)-level

    avgtime = avgtime/numigram

    return avgtime


def average_connN_igrams(date_ordinal,conn):
    ''' compute average temporal span (days) for connection-n interferograms

    Parameters:     date_ordinal        - list of size (num_date,) in integer, time in days
                    conn                - integer, connection level of interferograms
    Return：        avgtime             - float, average time-span in days for connnection conn interferograms .
    '''
    slcdate_firstn = date_ordinal[0:conn]
    slcdate_lastn = date_ordinal[-conn:]
    avgtime = 0
    for i in range(conn):
        avgtime = avgtime + slcdate_lastn[i] - slcdate_firstn[i]

    numigram = len(date_ordinal)-conn
    avgtime = avgtime/numigram

    return avgtime


def estimate_tsbias_approx(nl, bw, tbase, date_ordinal, wvl, box, outdir):
    ''' This script gives a quick approximate estimate of bias of a time-series of a certain bandwidth (bw) for a bounding box
        This estimate is not exact, but often close enough.
        It is good for a quick estimate to see how big the biases are.

    Parameters: nl              - integer, connection level that we assume bias-free
                bw              - integer, bandwidth of the given time-series analysis
                tbase           - list in size of (num_date,) in float, time in accumulated years
                date_ordinal    - list of size (num_date,) in integer, time in days
                wvl             - float, wavelength of the SAR system
                box             - list in size of (4,) in integer, coordinates of bounding box
                outdir          - string, directory for outputing files
    Returns:    biasts          - 3D array in size of (num_date, box_length, box_width) in float, bias timeseries
    '''
    # average temporal span for ifgrams of connection-1 to connection-bw
    deltat_n = [average_connN_igrams(date_ordinal,n) for n in range(1,bw+1)]
    avgtimespan = average_temporal_span(date_ordinal,bw)
    # the bias in a bandwidth-bw analysis is similar to bias in connectoin-p interferograms
    p = (np.abs(np.asarray(deltat_n) - avgtimespan)).argmin()+1
    print('p = ',p)
    coef = -4*np.pi/wvl
    m1 = 2
    m2 = nl
    wratio_p = estimate_ratioX(tbase, p, nl, wvl, box, outdir)[0]
    wratio_m1 = estimate_ratioX(tbase, m1, nl, wvl, box, outdir)[0]
    wratio_p[np.isnan(wratio_p)] = 0
    wratio_m1[np.isnan(wratio_m1)] = 0
    wratio_m1[abs(wratio_m1-1)<0.1] = np.nan
    ratio1 = np.divide(wratio_p,(1-wratio_m1))
    biasts1 = seq2cum_closure_phase(m1, outdir, box)
    biasts2 = seq2cum_closure_phase(m2, outdir, box)
    for i in range(biasts1.shape[0]):
        biasts1[i,:,:] = np.multiply(biasts1[i,:,:]/coef,ratio1)
        biasts2[i,:,:] = np.multiply(biasts2[i,:,:]/coef,wratio_p)
    biasts = biasts1
    biasts[np.isnan(biasts)]=biasts2[np.isnan(biasts1)]
    return biasts


def quick_bias_correction(stack_file, nl, bw, max_memory, outdir):
    '''Output Wr (eq.20 in Zheng et al., 2022) and a quick approximate solution to bias time-series

    Parameters: stack_file                - string, path for ifgramStack.h5
                nl                          - integer, connection level at which we assume is bias-free
                bw                          - integer, bandwidth of the given time-series.
                wvl                         - float, wavelength of the SAR System
                max_mermory                 - float, maximum memory in GB for each patch processed
                outdir                      - string, directory for output files
    Returns:    Wratio.h5                   - output hdf5 file storing two 3D array of size (bw, length, width) of float, wratios and bias_velocity
                bias_timeseries_approx.h5   - output hdf5 file storing a 3D array of size (num_date, length, width) of float, approximate bias time-series.
    '''
    stack_obj = ifgramStack(stack_file)
    stack_obj.open()
    length, width = stack_obj.length, stack_obj.width
    date12_list = stack_obj.get_date12_list(dropIfgram=True)

    date1s = [i.split('_')[0] for i in date12_list]
    date2s = [i.split('_')[1] for i in date12_list]
    date_list = sorted(list(set(date1s + date2s)))
    # tbase in the unit of years
    date_format = ptime.get_date_str_format(date_list[0])
    dates = np.array([dt.strptime(i, date_format) for i in date_list])
    tbase = [i.days + i.seconds / (24 * 60 * 60) for i in (dates - dates[0])]
    tbase = np.array(tbase, dtype=np.float32) / 365.25
    date_ordinal = []
    for date_str in date_list:
        format_str = '%Y%m%d'
        datetime_obj = dt.strptime(date_str, format_str)
        date_ordinal.append(datetime_obj.toordinal())

    meta = dict(stack_obj.metadata)
    wvl = float(meta['WAVELENGTH']) *100
    date_list = np.array(date_list, np.string_)
    connlist = list(np.arange(1,bw+1))
    connlist.append(nl)
    Wr_filedir = os.path.join(outdir, 'Wratio.h5')
    meta['FILE_TYPE'] = None
    ds_name_dict = {
        'wratio'        : [np.float32,     (len(connlist)-1, length, width), None],
        'bias_velocity' : [np.float32,     (len(connlist)-1, length, width), None],
        'date'          : [np.dtype('S8'), np.shape(date_list),              date_list],}
    writefile.layout_hdf5(Wr_filedir, ds_name_dict, meta)

    # split igram_file into blocks to save memory
    box_list, num_box = ifginv.split2boxes(stack_file, max_memory)

    #process block-by-block
    for i, box in enumerate(box_list):
            box_width  = box[2] - box[0]
            box_length = box[3] - box[1]
            print(box)
            if num_box > 1:
                print('\n------- processing patch {} out of {} --------------'.format(i+1, num_box))
                print('box width:  {}'.format(box_width))
                print('box length: {}'.format(box_length))

            w_ratios = np.zeros([len(connlist)-1,box_length,box_width])
            w_ratios_velocity = np.zeros([len(connlist)-1,box_length, box_width])
            for idx in range(len(connlist)-1):
                conn = connlist[idx]
                w,wv = estimate_ratioX(tbase, conn, nl, wvl, box, outdir,mask=True)
                w_ratios[idx,:,:] = w
                w_ratios_velocity[idx,:,:] = wv

            # write the block to disk
            block = [0, len(connlist)-1,box[1], box[3], box[0], box[2]]

            writefile.write_hdf5_block(Wr_filedir,
                                       data=w_ratios,
                                       datasetName='wratio',
                                       block=block)

            writefile.write_hdf5_block(Wr_filedir,
                                       data=w_ratios_velocity,
                                       datasetName='bias_velocity',
                                       block=block)

    # a quick/approximate estimate for bias time-series
    biasfile = os.path.join(outdir, 'bias_timeseries_approx.h5')
    meta = dict(stack_obj.metadata)
    ds_name_dict = {
        'timeseries' : [np.float32,     (len(date_list), length, width), None],
        'date'       : [np.dtype('S8'), np.shape(date_list),             date_list],
    }
    writefile.layout_hdf5(biasfile, ds_name_dict, meta)
    for i, box in enumerate(box_list):
        tsbias = estimate_tsbias_approx(nl, bw, tbase, date_ordinal, wvl, box, outdir)
        block = [0, len(date_list),box[1], box[3], box[0], box[2]]
        writefile.write_hdf5_block(biasfile,
                                   data=tsbias/100,
                                   datasetName='timeseries',
                                   block=block)

    return


def estimate_bias(stack_file, nl, bw, wvl, box, outdir):
    '''Output bias time-series of a certain bandwidth (bw) for a bounding box using the algorithm provided in Zheng et al., 2022

    Parameters: stack_file - string, path for ifgramStack.h5
                nl           - integer, connection level at which we assume is bias-free
                bw           - integer, bandwidth of the given time-series.
                wvl          - float, wavelength of the SAR System
                box          - list in size of (4,) in integer, coordinates of bounding box
                outdir       - string, directory for output files
    Returns:    biasts_bwn   - 3D array of size (bw, box_length, box_width) of float, estimated bias time-series
                box          - list in size of (4,) in integer, coordinates of bounding box, output for parallel computing
    '''
    coef = -4*np.pi/wvl
    box_width  = box[2] - box[0]
    box_length = box[3] - box[1]
    numpix = box_width * box_length
    stack_obj = ifgramStack(stack_file)
    stack_obj.open()
    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    A,B = stack_obj.get_design_matrix4timeseries(date12_list = date12_list, refDate = 'no')[0:2]
    B = B[:,:-1]

    # We first need to have the bias time-series for bw-1 analysis
    biasts_bw1_rough = seq2cum_closure_phase(nl, outdir, box)
    m = 2
    biasts_bw1_fine  = seq2cum_closure_phase(m, outdir, box)
    date1s = [i.split('_')[0] for i in date12_list]
    date2s = [i.split('_')[1] for i in date12_list]
    date_list = sorted(list(set(date1s + date2s)))
    num_date = len(date_list)
    # tbase in the unit of years
    date_format = ptime.get_date_str_format(date_list[0])
    dates = np.array([dt.strptime(i, date_format) for i in date_list])
    tbase = [i.days + i.seconds / (24 * 60 * 60) for i in (dates - dates[0])]
    tbase = np.array(tbase, dtype=np.float32) / 365.25
    tbase_diff = np.diff(tbase).reshape(-1, 1)
    delta_T = tbase[-1]-tbase[0]
    velocity_m = biasts_bw1_fine[-1,:,:]/coef/delta_T
    mask = np.where(np.abs(velocity_m)<0.1, 0,1)

    for i in range(num_date):
        biasts_bw1_fine[i,:,:] = np.multiply(np.divide(biasts_bw1_rough[-1,:,:],
                                                       biasts_bw1_fine[-1,:,:]),
                                             biasts_bw1_fine[i,:,:])

    biasts_bw1_rough = biasts_bw1_rough.reshape(num_date,-1)
    biasts_bw1_fine = biasts_bw1_fine.reshape(num_date,-1)
    mask = mask.reshape(-1)

    # Then We construct ifgram_bias (W A \Phi^X, or Wr A w(\delta_t)\Phi^X
    # Eq.(19) in Zheng et al., 2022), same structure with stack_file
    biasts_bwn = np.zeros((num_date, numpix),dtype = np.float32)
    num_ifgram = np.shape(A)[0]
    if num_ifgram != int(bw*(num_date*2-bw-1)/2): # check the dimensions
        print('Number of interferograms expected: ',int(bw*(num_date*2-bw-1)/2))
        print('Number of interferograms found: ', num_ifgram)
        raise Exception("Modify maximum connection in ifgramStack.h5 to be consistent with input bandwidth!")

    # this matrix is a numpix by num_ifgram matrix, each row stores the diagnal component of the Wr matrix for that pixel
    W = get_design_matrix_W(num_ifgram, A, bw, box, tbase, nl, outdir)
    prog_bar = ptime.progressBar(maxValue=numpix)
    for i in range(numpix):
        Wr = np.diag(W[i,:])
        WrA = np.matmul(Wr,A)
        Dphi_rough = biasts_bw1_rough[:,i]
        Dphi_fine  = biasts_bw1_fine [:,i]
        if mask[i] == 0 :
            Dphi_bias = np.matmul(WrA,Dphi_rough)
        else:
            Dphi_bias  = np.matmul(WrA,Dphi_fine)

        # here we perform phase velocity inversion as per the original SBAS paper rather doing direct phase inversion.
        B_inv  = np.linalg.pinv(B)
        biasvel = np.matmul(B_inv,Dphi_bias)
        biasts = np.cumsum(biasvel.reshape(-1)*tbase_diff.reshape(-1))
        biasts_bwn[1:,i] = biasts/coef
        prog_bar.update(i+1, every=200, suffix='{}/{} pixels'.format(i+1, numpix))
    prog_bar.close()
    biasts_bwn = biasts_bwn.reshape(num_date, box_length, box_width)

    return biasts_bwn,box


def bias_correction(stack_file, nl, bw, max_memory, outdir, parallel):
    '''Output a solution to bias time-series

    Parameters: stack_file       - string, path for ifgramStack.h5
                nl                 - integer, connection level at which we assume is bias-free
                bw                 - integer, bandwidth of the given time-series.
                max_memory         - float, maximum memory in GB for each patch processed
                outdir             - string, directory for output files
                parallel           - dictonary containing settings of parallel computing. To turn off, set parallel['clustertype']=''
    Returns:    bias_timeseries.h5 - output hdf5 file storing a 3D array of size (num_date, length, width) of float, estimated bias time-series.
    '''
    stack_obj = ifgramStack(stack_file)
    stack_obj.open()
    length, width = stack_obj.length, stack_obj.width
    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    date1s = [i.split('_')[0] for i in date12_list]
    date2s = [i.split('_')[1] for i in date12_list]
    date_list = sorted(list(set(date1s + date2s)))
    # split igram_file into blocks to save memory
    box_list, num_box = ifginv.split2boxes(stack_file, max_memory)

    # estimate for bias time-series
    biasfile = os.path.join(outdir, 'bias_timeseries.h5')
    meta = dict(stack_obj.metadata)
    wvl = float(meta['WAVELENGTH']) *100 # convert to cm
    date_list = np.array(date_list, np.string_)
    num_date = len(date_list)
    ds_name_dict = {
        'timeseries' : [np.float32,     (len(date_list), length, width), None],
        'date'       : [np.dtype('S8'), np.shape(date_list),             date_list],}
    writefile.layout_hdf5(biasfile, ds_name_dict, meta)

    data_kwargs = {
        "stack_file" : stack_file,
        "nl"         : nl,
        "bw"         : bw,
        "wvl"        : wvl,
        "outdir"     : outdir,
    }
    num_threads_dict = cluster.set_num_threads("1")
    start_time = time.time()
    for i, box in enumerate(box_list):
        box_width  = box[2] - box[0]
        box_length = box[3] - box[1]
        print(box)
        if num_box > 1:
            print('\n------- processing patch {} out of {} --------------'.format(i+1, num_box))
            print('box width:  {}'.format(box_width))
            print('box length: {}'.format(box_length))
        #update box argument in the input data
        data_kwargs['box'] = box
        if not parallel['clustertype']:
            # non-parallel
            tsbias = estimate_bias(stack_file, nl, bw, wvl, box, outdir)[:-1]
        else:
            # parallel
            print('\n\n------- start parallel processing using Dask -------')
            # initiate the output data
            tsbias = np.zeros((num_date, box_length, box_width), np.float32)
            # initiate dask cluster and client
            cluster_obj = cluster.DaskCluster(
                cluster_type=parallel['clustertype'],
                num_worker=parallel['numWorker'],
                config_name=parallel['config_name'],
            )
            cluster_obj.open()
            # run dask
            tsbias, box = cluster_obj.run(
                func=estimate_bias,
                func_data=data_kwargs,
                results=[tsbias, box],
            )
            # close dask cluster and client
            cluster_obj.close()
            print('------- finished parallel processing -------\n\n')

        block = [0, len(date_list),box[1], box[3], box[0], box[2]]
        writefile.write_hdf5_block(biasfile,
                                   data=tsbias/100,
                                   datasetName='timeseries',
                                   block=block)

    # roll back to the original number of threads
    cluster.roll_back_num_threads(num_threads_dict)
    m, s = divmod(time.time() - start_time, 60)
    print('time used: {:02.0f} mins {:02.1f} secs.\n'.format(m, s))
    return






################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)

    if inps.action == 'mask':
        calc_closure_phase_mask(
            stack_file=inps.stack_file,
            nl=inps.nl,
            num_sigma=inps.num_sigma,
            threshold_amp=inps.epsilon,
            outdir=inps.outdir,
            max_memory=inps.maxMemory)

    elif inps.action.endswith ('estimate'):
        # to make sure we have con-2 closure phase processed
        max_conn_level = np.maximum(2, inps.bw)
        if inps.update_closure_phase:
            for conn_level in np.arange(2, max_conn_level+1):
                compute_unwrap_closure_phase(inps.stack_file, conn_level, inps.maxMemory, inps.outdir)
            compute_unwrap_closure_phase(inps.stack_file, inps.nl, inps.maxMemory, inps.outdir)

        if inps.action == 'quick_estimate':
            # a quick solution to bias-correction
            # output diagonal component of Wr (how fast the bias-inducing signal decays with temporal baseline)
            quick_bias_correction(inps.stack_file, inps.nl, inps.bw, inps.maxMemory, inps.outdir)

        elif inps.action == 'estimate':
            # bias correction
            parallel={
                "clustertype" : inps.cluster,
                "numWorker"   : inps.numWorker,
                "config_name" : inps.config,
            }
            bias_correction(inps.stack_file, inps.nl, inps.bw, inps.maxMemory, inps.outdir, parallel)

    return


################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
