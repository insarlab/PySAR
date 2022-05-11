#!/usr/bin/env python
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Yujie Zheng, Feb 2022                            #
############################################################
# Compute average con-nl closure phase and output mask identifying areas suseptible to closure phase errors.

import os
import sys
import argparse
import numpy as np
import cv2
import glob
from datetime import datetime as dt

from mintpy.objects import ifgramStack
from mintpy.utils import readfile, writefile, ptime
from mintpy import ifgram_inversion as ifginv
import isce
import isceobj

################################################################################
REFERENCE = """reference:
  Y. Zheng, H. Fattahi, P. Agram, M. Simons and P. Rosen, "On Closure Phase and Systematic Bias in Multi-looked SAR Interferometry," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2022.3167648.
"""
EXAMPLE = """example:
    closure_phase_bias.py -i inputs/ifgramStack.h5 --nl 20 --action create_mask
    closure_phase_bias.py -i inputs/ifgramStack.h5 --nl 20 --numsigma 2.5 --action create_mask
    closure_phase_bias.py -i inputs/ifgramStack.h5 --nl 20 --bw 10 -a quick_biasEstimate
    closure_phase_bias.py -i inputs/ifgramStack.h5 --nl 20 --bw 10 -a biasEstimate
"""

def create_parser():
    parser = argparse.ArgumentParser(description = 'This script deals with phase non-closure related bias, either in terms of masking, estimation or correction.')
    parser.add_argument('-i','--ifgramstack',type = str, dest = 'ifgram_stack',help = 'interferogram stack file that contains the unwrapped phases')
    parser.add_argument('--nl', dest = 'nl', type = int, default = 20, help = 'connection level that we are correcting to (or consider as no bias)')
    parser.add_argument('--bw', dest = 'bw', type = int, default = 10, help = 'bandwidth of time-series analysis that you want to correct')
    parser.add_argument('--wvl',dest = 'wvl', type = float, default = 5.6, help = 'wavelength in cm of the SAR satellite, default 5.6 cm' )
    parser.add_argument('--numsigma',dest = 'numsigma', type = float, default = 3, help = 'Threashold for phase (number of sigmas,0-infty), default to be 3 sigma of a Gaussian distribution (assumed distribution for the cumulative closure phase) with sigma = pi/sqrt(3*num_cp)')
    parser.add_argument('--epi',dest = 'episilon', type = float, default = 0.3, help = 'Threashold for amplitude (0-1), default 0.3')
    parser.add_argument('--maxMemory', dest = 'max_memory', type = float, default = 8, help = 'max memory to use in GB')
    parser.add_argument('-o', dest = 'outdir', type = str, default = '.', help = 'output file directory')
    parser.add_argument('-a','--action', dest='action', type=str, default='create_mask',
                        choices={'create_mask', 'quick_biasEstimate', 'biasEstimate'},
                        help='action to take (default: %(default)s):\n'+
                             'create_mask -  create a mask of areas susceptible to closure phase errors\n'+
                             'quick_biasEstimate - estimate how bias decays with time, will output sequential closure phase files, and gives a quick and appximate bias estimation\n'
                             'biasEstimate - estimate how bias decays with time, processed for each pixel on a pixel by pixel basis, not parallized yet')
    return parser

def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args = iargs)
    return inps

def seq_closurePhase(SLC_list, date12_list_all, ifgram_stack, ref_phase, n, box):
    """
    Input parameters:
        SLC_list : list of SLC dates
        date12_list_all: date12 of all the interferograms stored in the ifgramstack file
        ifgram_stack: stack file
        refphase : reference phase
        n        : connection level of the closure phase
        box      : bounding box for the patch
    Output: cp_w : stack of wrapped sequential closure phases of connection n
    """
    cp_idx = []
    NSLC = len(SLC_list)
    for i in range(NSLC-n):
        ifgram = []
        flag = True
        for j in range(n):
            ifgram.append('{}_{}'.format(SLC_list[i+j],SLC_list[i+j+1]))
        ifgram.append('{}_{}'.format(SLC_list[i],SLC_list[i+n]))
        for ifgram_name in ifgram:
            if ifgram_name not in date12_list_all:
                flag = False # if missing an interferogram, we won't make the corresponding closure phase
        if flag:
            cp_idx.append([date12_list_all.index(ifgram[j]) for j in range(n+1)])

    cp_idx = np.array(cp_idx, np.int16)
    cp_idx = np.unique(cp_idx, axis = 0)

    num_cp = len(cp_idx)
    print('Number of closure measurements expected, ', len(SLC_list)-n)
    print('Number of closure measurements found, ', num_cp)

    if num_cp < len(SLC_list)-n:
        print('Missing interferograms, abort')
        raise Exception("Some interferograms are missing")

    box_width  = box[2] - box[0]
    box_length = box[3] - box[1]
    phase = readfile.read(ifgram_stack, box=box,print_msg=False)[0]
    cp_w = np.zeros((num_cp, box_length, box_width), np.float32)
    for i in range(num_cp):
        cp0_w = np.zeros ((box_length, box_width), np.float32)
        for j in range(n):
                    idx = cp_idx[i,j]
                    cp0_w = cp0_w + phase[idx,:,:] - ref_phase[idx]
        idx = cp_idx[i,n]
        cp0_w = cp0_w - (phase[idx,:,:]-ref_phase[idx])
        cp_w[i,:,:] = np.angle(np.exp(1j*cp0_w))

    return cp_w


def sum_seq_closurePhase(SLC_list, date12_list_all, ifgram_stack, ref_phase, n, box):
    """
    Input parameters:
        SLC_list : list of SLC dates
        date12_list_all: date12 of all the interferograms stored in the ifgramstack file
        ifgram_stack: stack file
        refphase : reference phase
        n        : connection level of the closure phase
        box      : bounding box for the patch
    Output parameters:
        cum_cp   : sum of consecutive complex sequential closure phase of connection n
        num_cp   : number of closure phases in the sum
    """
    cp_idx = []
    NSLC = len(SLC_list)
    for i in range(NSLC-n):
        ifgram = []
        flag = True
        for j in range(n):
            ifgram.append('{}_{}'.format(SLC_list[i+j],SLC_list[i+j+1]))
        ifgram.append('{}_{}'.format(SLC_list[i],SLC_list[i+n]))
        for ifgram_name in ifgram:
            if ifgram_name not in date12_list_all:
                flag = False # if missing an interferogram, we won't make the corresponding closure phase
        if flag:
            cp_idx.append([date12_list_all.index(ifgram[j]) for j in range(n+1)])

    cp_idx = np.array(cp_idx, np.int16)
    cp_idx = np.unique(cp_idx, axis = 0)

    num_cp = len(cp_idx)
    print('Number of closure measurements expected, ', len(SLC_list)-n)
    print('Number of closure measurements found, ', num_cp)

    if num_cp <1:
        print('No closure phase measurements found, abort')
        raise Exception("No triplets found!")

    box_width  = box[2] - box[0]
    box_length = box[3] - box[1]
    phase = readfile.read(ifgram_stack, box=box,print_msg=False)[0]
    cum_cp = np.zeros((box_length, box_width), np.complex64)
    for i in range(num_cp):
        cp0_w = np.zeros ((box_length, box_width), np.float32)
        for j in range(n):
                    idx = cp_idx[i,j]
                    cp0_w = cp0_w + phase[idx,:,:] - ref_phase[idx]
        idx = cp_idx[i,n]
        cp0_w = cp0_w - (phase[idx,:,:]-ref_phase[idx])
        cum_cp = cum_cp + (np.exp(1j*cp0_w))

    return cum_cp, num_cp

def gaussian_kernel(Sx, Sy, sig_x, sig_y):
    if np.mod(Sx,2) == 0:
        Sx = Sx + 1

    if np.mod(Sy,2) ==0:
            Sy = Sy + 1

    x,y = np.meshgrid(np.arange(Sx),np.arange(Sy))
    x = x + 1
    y = y + 1
    x0 = (Sx+1)/2
    y0 = (Sy+1)/2
    fx = ((x-x0)**2.)/(2.*sig_x**2.)
    fy = ((y-y0)**2.)/(2.*sig_y**2.)
    k = np.exp(-1.0*(fx+fy))
    a = 1./np.sum(k)
    k = a*k
    return k

def convolve(data, kernel):

    R = cv2.filter2D(data.real,-1,kernel)
    Im = cv2.filter2D(data.imag,-1,kernel)

    return R + 1J*Im

def write_xml(filename,width,length,bands,dataType,scheme):
     img=isceobj.createImage()
     img.setFilename(filename)
     img.setWidth(width)
     img.setLength(length)
     img.setAccessMode('Read')
     img.bands=bands
     img.dataType=dataType
     img.scheme = scheme
     img.renderHdr()
     img.renderVRT()
     return None

def estCoherence(outfile, corfile):
    from mroipac.icu.Icu import Icu

    #Create phase sigma correlation file here
    filtImage = isceobj.createIntImage()
    filtImage.load( outfile + '.xml')
    filtImage.setAccessMode('read')
    filtImage.createImage()

    phsigImage = isceobj.createImage()
    phsigImage.dataType='FLOAT'
    phsigImage.bands = 1
    phsigImage.setWidth(filtImage.getWidth())
    phsigImage.setFilename(corfile)
    phsigImage.setAccessMode('write')
    phsigImage.createImage()


    icuObj = Icu(name='sentinel_filter_icu')
    icuObj.configure()
    icuObj.unwrappingFlag = False
    icuObj.useAmplitudeFlag = False
    #icuObj.correlationType = 'NOSLOPE'

    icuObj.icu(intImage = filtImage,  phsigImage=phsigImage)
    phsigImage.renderHdr()

    filtImage.finalizeImage()
    phsigImage.finalizeImage()

def unwrap_snaphu(intfile,corfile,unwfile,length, width):
    from contrib.Snaphu.Snaphu import Snaphu

    altitude = 800000.0
    earthRadius = 6371000.0
    wavelength = 0.056

    snp = Snaphu()
    snp.setInitOnly(False)
    snp.setInput(intfile)
    snp.setOutput(unwfile)
    snp.setWidth(width)
    snp.setCostMode('DEFO')
    snp.setEarthRadius(earthRadius)
    snp.setWavelength(wavelength)
    snp.setAltitude(altitude)
    snp.setCorrfile(corfile)
    snp.setInitMethod('MST')
   # snp.setCorrLooks(corrLooks)
    snp.setMaxComponents(100)
    snp.setDefoMaxCycles(2.0)
    snp.setRangeLooks(80)
    snp.setAzimuthLooks(20)
    snp.setCorFileFormat('FLOAT_DATA')
    snp.prepare()
    snp.unwrap()

    write_xml(unwfile, width, length, 2 , "FLOAT",'BIL')

def cum_seq_unwclosurePhase(n,filepath,length, width, refY, refX, SLC_list, meta):
    '''output cumulative con-n sequential closure phase in time-series format (Eq. 25 in Zheng et al., 2022, but divided by n)
    Input parameters: n -  connection level of closure phases
                      filepath -- filepath of sequential closure phases of connection - n
                      width, length -- width and length of the interferograms
                      refY, refX -- reference point coordinates
                      SLC_list: list of SLC
                      meta: metadata of ifgramStack.h5
    '''
    outfiledir = os.path.join(filepath, 'con'+str(n)+'_seqcumclosurephase.h5')
    outmaskdir = os.path.join(filepath, 'con'+str(n)+'_seqcumclosurephase_maskconcp.h5')
    print('Creating '+outfiledir)
    if not os.path.isfile(outfiledir) or not os.path.isfile(outmaskdir):
        filelist = glob.glob(os.path.join(filepath, '*.unw'))
        numfile = len(filelist)
        filelist_st = sorted(filelist)
        cp_phase_all = np.zeros((numfile, length, width),np.float32)
        mask_all = np.zeros((numfile, length, width),np.float32)

        for i in range(numfile):
            file = filelist_st[i]
            concpfile = file.replace('.unw','.unw.conncomp')
            corfile = file.replace('.unw','.cor')
            cp =np.fromfile(file, dtype='float32')
            cp = cp.reshape([length,width*2])
            cor = np.fromfile(corfile, dtype = 'float32')
            concp = np.fromfile(concpfile, dtype = 'byte')
            cor = cor.reshape([length, width])
            concp = concp.reshape([length, width])
            cp_phase = cp[:,width:]
            cp_phase = cp_phase - cp_phase[refY,refX]
            mask_concp = np.where(concp >= 1, 1, np.nan)
            cp_phase_all[i,:,:] = cp_phase
            mask_all[i,:,:] = mask_concp

        # compute sequential closure phase
        N = len(SLC_list)
        biasts = np.zeros([N,length, width], np.float32)
        biasts[0,:,:] = 0
        biasts[1:N-n+1,:,:]= np.cumsum(cp_phase_all,0)
        for i in range(N-n+1,N):
            biasts[i,:,:] = (i-N+n)*cp_phase_all[-1,:,:]+biasts[N-n,:,:]

        SLC_list = np.array(SLC_list, np.string_)
        dsDict = dict()
        dsDict = {'seqcum_closurephase': [np.float32, (N, length, width), biasts/n],
                          'date': [np.dtype('S8'),np.shape(SLC_list), SLC_list],}

        meta['FILE_TYPE'] = 'timeseries'
        writefile.layout_hdf5(outfiledir, dsDict, meta)

        mask = np.where(np.isnan(np.sum(mask_all,0)), 0, 1)
        dsDict = dict()
        dsDict = {'mask_concp': [np.float32, (length, width),mask ],
                      'date': [np.dtype('S8'),np.shape(SLC_list), SLC_list],}
        meta['FILE_TYPE'] = 'mask'
        writefile.layout_hdf5(outmaskdir, dsDict, meta)

def seq2cum_closurePhase(conn, outdir, box):
    '''
    this script read in cumulative sequential closure phase from individual closure phase directory (Eq. 25) in Zheng et al., 2022
    output should be a 3D matrix of size NSLC by box_lengh by box_width
    '''
    filepath = 'con'+str(conn)+'_cp'
    filename = 'con'+str(conn)+'_seqcumclosurephase.h5'
    seqcpfile = os.path.join(outdir, 'ClosurePhase', filepath, filename)
    biasts = readfile.read(seqcpfile, box=box,print_msg=False)[0]
    return biasts

def estimate_ratioX(tbase, n, nl, wvl, box, outdir):
    '''
    # This script estimates w(n\delta_t)/w(delta_t), Eq.(29) in Zheng et al., 2022
    # input: tbase - time in accumulated years
    # input: n - connection-level
    # input: nl - minimum connection-level that we think is bias-free
    # input: wvl - wavelength
    # input: box - the patch that is being processed
    # input: outdir - the working directory
    # output: wratio - Eq.(29)
    # output: wratio_velocity - bias-velocity at n*delta_t temporal baseline
    '''
    box_width  = box[2] - box[0]
    box_length = box[3] - box[1]
    cum_bias_conn_1 = seq2cum_closurePhase(nl, outdir, box)[-1,:,:]
    coef = -4*np.pi/wvl
    delta_T = tbase[-1]-tbase[0]
    vel_bias_conn1 = cum_bias_conn_1/coef/delta_T

    if n==1:
        wratio = np.ones([box_length, box_width])
        wratio[abs(vel_bias_conn1)<0.1]=0 # if average velocity smaller than 1 mm/year (hardcoded here), regard as no bias
        wratio_velocity = np.multiply(wratio,vel_bias_conn1)

    else:
        cum_bias_conn_n =  seq2cum_closurePhase(n, outdir,box)[-1,:,:]
        wratio = np.divide(cum_bias_conn_n,cum_bias_conn_1)
        wratio = 1-wratio
        wratio[abs(vel_bias_conn1)<0.1]=0 # if average velocity smaller than 1 mm/year (hardcoded here), regard as no bias
        wratio[wratio>1]=1
        wratio[wratio<0]=0
        wratio_velocity = np.multiply(wratio,vel_bias_conn1)

    return wratio,wratio_velocity # wratio is a length by width 2D matrix

def estimate_ratioX_all(bw,nl,outdir,box):
    '''
    Estimate wratio for connection-1 through connection bw
    '''
    box_width  = box[2] - box[0]
    box_length = box[3] - box[1]
    cum_bias_conn_1 = seq2cum_closurePhase(nl, outdir, box)[-1,:,:]
    wratio = np.zeros([bw+1,box_length, box_width], dtype = np.float32)
    for n in np.arange(2,bw+1):
        cum_bias_conn_n =  seq2cum_closurePhase(n, outdir, box)[-1,:,:]
        wratio[n,:,:] = np.divide(cum_bias_conn_n,cum_bias_conn_1)

    wratio = 1-wratio
    wratio[wratio>1]=1
    wratio[wratio<0]=0
    return wratio # wratio is a bw+1 by length by width 3D matrix, the first layer is a padding

def get_design_matrix_W(M, A, bw, box, tbase, nl, outdir):
    '''
    # output: W, numpix by numifgram matrix, each row stores the diagnal component of W (Eq. 16 in Zheng et al., 2022) for one pixel.
    # input: M - num_ifgram
    # input: A - M by N design matrix specifying SAR acquisitions used
    # input: tbase - time in accumulated years
    # input: nl - minimum connection-level that we think is bias-free
    # input: box - the patch that is being processed
    # input: outdir - the working directory
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
        wratio = wratioall[conn,:,:]
        wratio = wratio.reshape(-1)
        W[:,i] = wratio

    return W

def averagetemporalspan(date_ordinal,conn):
    '''
    compute average temporal span (days) for interferogram subsets chosen for bw-n analysis
    '''
    avgtime = 0
    numigram = 0
    for level in range(1, conn+1):
        slcdate_firstn = date_ordinal[0:level]
        slcdate_lastn = date_ordinal[-level:]
        for i in range(level):
            avgtime = avgtime + slcdate_lastn[i] - slcdate_firstn[i]
        numigram = numigram + len(date_ordinal)-level

    avgtime = avgtime/numigram

    return avgtime

def averageconnNigrams(date_ordinal,conn):
    '''
    compute average temporal span (days) for n-connection interferograms
    '''
    slcdate_firstn = date_ordinal[0:conn]
    slcdate_lastn = date_ordinal[-conn:]
    avgtime = 0
    for i in range(conn):
        avgtime = avgtime + slcdate_lastn[i] - slcdate_firstn[i]

    numigram = len(date_ordinal)-conn
    avgtime = avgtime/numigram

    return avgtime

def estimatetsbias_approx(nl, bw, tbase, date_ordinal, wvl, box, outdir):
    '''
    # This script gives a quick approximate estimate of bias of a time-series of a certain bandwidth (bw)
    # This estimate is not exact, but often close enough.
    # It is good for a quick estimate to see how big the biases are.
    Input parameters: nl - connection level that we assume bias-free
                      bw - bandwidth of the given time-series analysis
                      wvl - wavelength of the SAR system
                      box - patch that we are processing
                      outdir - directory for outputing files
    Output parameters: biasts - bias timeseries
    '''
    deltat_n = [averageconnNigrams(date_ordinal,n) for n in range(1,bw+1)] # average temporal span for ifgrams of connection-1 to connection-bw
    avgtimespan = averagetemporalspan(date_ordinal,bw)
    p = (np.abs(np.asarray(deltat_n) - avgtimespan)).argmin()+1 # the bias in a bandwidth-bw analysis is similar to bias in connectoin-p interferograms
    print('p = ',p)
    coef = -4*np.pi/wvl
    m1 = 2
    m2 = nl
    wratio_p = estimate_ratioX(tbase, p, nl, wvl, box, outdir)[0]
    wratio_m1 = estimate_ratioX(tbase, m1, nl, wvl, box, outdir)[0]
    wratio_m1[abs(wratio_m1-1)<0.1] = np.nan
    ratio1 = np.divide(wratio_p,(1-wratio_m1))
    biasts1 = seq2cum_closurePhase(m1, outdir, box)
    biasts2 = seq2cum_closurePhase(m2, outdir, box)
    for i in range(biasts1.shape[0]):
        biasts1[i,:,:] = np.multiply(biasts1[i,:,:]/coef,ratio1)
        biasts2[i,:,:] = np.multiply(biasts2[i,:,:]/coef,wratio_p)
    biasts = biasts1
    biasts[np.isnan(biasts)]=biasts2[np.isnan(biasts1)]
    return biasts

def quickbiascorrection(ifgram_stack, nl, bw, wvl, max_memory, outdir):
    '''
    Output Wr (eq.20 in Zheng et al., 2022) and a quick approximate solution to bias time-series
    Input parameters:
            ifgram_stack : ifgramStack object
            nl: connection level at which we assume is bias-free
            bw: bandwidth of the given time-series.
            wvl: wavelength of the SAR System
            max_mermory: maximum memory for each patch processed
            outdir: directory for output files
    '''
    stack_obj = ifgramStack(ifgram_stack)
    stack_obj.open()
    length, width = stack_obj.length, stack_obj.width
    date12_list = stack_obj.get_date12_list(dropIfgram=True)

    date1s = [i.split('_')[0] for i in date12_list]
    date2s = [i.split('_')[1] for i in date12_list]
    SLC_list = sorted(list(set(date1s + date2s)))
    # tbase in the unit of years
    date_format = ptime.get_date_str_format(SLC_list[0])
    dates = np.array([dt.strptime(i, date_format) for i in SLC_list])
    tbase = [i.days + i.seconds / (24 * 60 * 60) for i in (dates - dates[0])]
    tbase = np.array(tbase, dtype=np.float32) / 365.25
    date_ordinal = []
    for date_str in SLC_list:
        format_str = '%Y%m%d'
        datetime_obj = dt.strptime(date_str, format_str)
        date_ordinal.append(datetime_obj.toordinal())

    meta = dict(stack_obj.metadata)
    SLC_list = np.array(SLC_list, np.string_)
    connlist = list(np.arange(1,bw+1))
    connlist.append(nl)
    Wr_filedir = os.path.join(outdir, 'Wratio.h5')
    meta['FILE_TYPE'] = None
    ds_name_dict = {'wratio': [np.float32, (len(connlist)-1, length, width), None],
                'bias_velocity': [np.float32, (len(connlist)-1, length, width), None],
                'date': [np.dtype('S8'),np.shape(SLC_list), SLC_list],}
    writefile.layout_hdf5(Wr_filedir, ds_name_dict, meta)

    # split igram_file into blocks to save memory
    box_list, num_box = ifginv.split2boxes(ifgram_stack, max_memory)

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
                w,wv = estimate_ratioX(tbase, conn, nl, wvl, box, outdir)
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
    ds_name_dict = {'timeseries': [np.float32, (len(SLC_list), length, width), None],
            'date': [np.dtype('S8'),np.shape(SLC_list), SLC_list],}
    writefile.layout_hdf5(biasfile, ds_name_dict, meta)
    for i, box in enumerate(box_list):
        tsbias = estimatetsbias_approx(nl, bw, tbase, date_ordinal, wvl, box, outdir)
        block = [0, len(SLC_list),box[1], box[3], box[0], box[2]]
        writefile.write_hdf5_block(biasfile,
                                   data=tsbias/100,
                                   datasetName='timeseries',
                                   block=block)

    return

def estimate_bias(ifgram_stack, nl, bw, wvl, box, outdir):
    '''
    input: ifgram_stack -- the ifgramstack file that you did time-series analysis with
    input: nl -- the connection level that we assume bias-free
    input: bw -- the bandwidth of the time-series analysis, should be consistent with the network stored in ifgram_stack
    input: wvl -- wavelength of the SAR satellite
    input: box -- the patch that is processed
    input: outdir -- directory for output files
    outut: biasts_bwn : estimated bias timeseries of the given patch
    '''
    coef = -4*np.pi/wvl
    box_width  = box[2] - box[0]
    box_length = box[3] - box[1]
    numpix = box_width * box_length
    stack_obj = ifgramStack(ifgram_stack)
    stack_obj.open()
    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    A,B = stack_obj.get_design_matrix4timeseries(date12_list = date12_list, refDate = 'no')[0:2]
    B = B[:,:-1]

    # We first need to have the bias time-series for bw-1 analysis
    biasts_bw1_rough = seq2cum_closurePhase(nl, outdir, box)
    m = 2
    biasts_bw1_fine  = seq2cum_closurePhase(m, outdir, box)
    date1s = [i.split('_')[0] for i in date12_list]
    date2s = [i.split('_')[1] for i in date12_list]
    SLC_list = sorted(list(set(date1s + date2s)))
    NSLC = len(SLC_list)
    # tbase in the unit of years
    date_format = ptime.get_date_str_format(SLC_list[0])
    dates = np.array([dt.strptime(i, date_format) for i in SLC_list])
    tbase = [i.days + i.seconds / (24 * 60 * 60) for i in (dates - dates[0])]
    tbase = np.array(tbase, dtype=np.float32) / 365.25
    tbase_diff = np.diff(tbase).reshape(-1, 1)
    delta_T = tbase[-1]-tbase[0]
    velocity_m = biasts_bw1_fine[-1,:,:]/coef/delta_T
    mask = np.where(np.abs(velocity_m)<0.1, 0,1)

    for i in range(NSLC):
        biasts_bw1_fine [i,:,:]  = np.multiply(np.divide(biasts_bw1_rough[-1,:,:],biasts_bw1_fine[-1,:,:]),biasts_bw1_fine[i,:,:])

    biasts_bw1_rough = biasts_bw1_rough.reshape(NSLC,-1)
    biasts_bw1_fine = biasts_bw1_fine.reshape(NSLC,-1)
    mask = mask.reshape(-1)

    # Then We construct ifgram_bias (W A \Phi^X, or Wr A w(\delta_t)\Phi^X in Eq.(19) in Zheng et al., 2022) , same structure with ifgram_stack
    biasts_bwn = np.zeros((NSLC, numpix),dtype = np.float32)
    num_ifgram = np.shape(A)[0]
    W = get_design_matrix_W(num_ifgram, A, bw, box, tbase, nl, outdir) # this matrix is a numpix by num_ifgram matrix, each row stores the diagnal component of the Wr matrix for that pixel
    for i in range(numpix):
        if i%2000==0:
            print(i, 'out of ', numpix, 'pixels processed')
        Wr = np.diag(W[i,:])
        WrA = np.matmul(Wr,A)
        Dphi_rough = biasts_bw1_rough[:,i]
        Dphi_fine  = biasts_bw1_fine [:,i]
        if mask[i] == 0 :
            Dphi_bias = np.matmul(WrA,Dphi_rough)
        else:
            Dphi_bias  = np.matmul(WrA,Dphi_fine)
        B_inv  = np.linalg.pinv(B) # here we perform phase velocity inversion as per the original SBAS paper rather doing direct phase inversion.
        biasvel = np.matmul(B_inv,Dphi_bias)
        biasts = np.cumsum(biasvel.reshape(-1)*tbase_diff.reshape(-1))
        biasts_bwn[1:,i] = biasts/coef
    biasts_bwn = biasts_bwn.reshape(NSLC, box_length, box_width)

    return biasts_bwn

def biascorrection(ifgram_stack, nl, bw, wvl, max_memory, outdir):
    '''
    input: ifgram_stack -- the ifgramstack file that you did time-series analysis with
    input: nl -- the connection level that we assume bias-free
    input: bw -- the bandwidth of the time-series analysis, should be consistent with the network stored in ifgram_stack
    input: wvl -- wavelength of the SAR satellite
    input: max_memory -- maximum memory of each patch
    input: outdir -- directory for output files
    '''
    stack_obj = ifgramStack(ifgram_stack)
    stack_obj.open()
    length, width = stack_obj.length, stack_obj.width
    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    date1s = [i.split('_')[0] for i in date12_list]
    date2s = [i.split('_')[1] for i in date12_list]
    SLC_list = sorted(list(set(date1s + date2s)))
    # split igram_file into blocks to save memory
    box_list, num_box = ifginv.split2boxes(ifgram_stack, max_memory)

    # estimate for bias time-series
    biasfile = os.path.join(outdir, 'bias_timeseries.h5')
    meta = dict(stack_obj.metadata)
    SLC_list = np.array(SLC_list, np.string_)
    ds_name_dict = {'timeseries': [np.float32, (len(SLC_list), length, width), None],
            'date': [np.dtype('S8'),np.shape(SLC_list), SLC_list],}
    writefile.layout_hdf5(biasfile, ds_name_dict, meta)
    for i, box in enumerate(box_list):
        box_width  = box[2] - box[0]
        box_length = box[3] - box[1]
        print(box)
        if num_box > 1:
            print('\n------- processing patch {} out of {} --------------'.format(i+1, num_box))
            print('box width:  {}'.format(box_width))
            print('box length: {}'.format(box_length))
        tsbias = estimate_bias(ifgram_stack, nl, bw, wvl, box, outdir)
        block = [0, len(SLC_list),box[1], box[3], box[0], box[2]]
        writefile.write_hdf5_block(biasfile,
                                   data=tsbias/100,
                                   datasetName='timeseries',
                                   block=block)
    return


def creat_cp_mask(ifgram_stack, nl, max_memory, numsigma, threshold_amp, outdir):
    """
    Input parameters:
        ifgram_stack: stack file
        nl        : maximum connection level that assumed to be bias free
        max_memory : maxum memory for each bounding box
        threshold_pha, threshold_amp: threshold of phase and ampliutde of the cumulative sequential closure phase
    """
    stack_obj = ifgramStack(ifgram_stack)
    stack_obj.open()
    length, width = stack_obj.length, stack_obj.width
    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    date12_list_all = stack_obj.get_date12_list(dropIfgram=False)
    print('scene length, width', length, width)
    ref_phase = stack_obj.get_reference_phase(unwDatasetName = 'unwrapPhase')

    # retrieve the list of SLC dates from ifgramStack.h5
    ifgram0 = date12_list[0]
    date1, date2 = ifgram0.split('_')
    SLC_list = [date1, date2]
    for ifgram in date12_list:
        date1, date2 = ifgram.split('_')
        if date1 not in SLC_list:
            SLC_list.append(date1)
        if date2 not in SLC_list:
            SLC_list.append(date2)
    SLC_list.sort()
    print('number of SLC found : ', len(SLC_list))
    print('first SLC: ', SLC_list[0])
    print('last  SLC: ', SLC_list[-1])

    # split igram_file into blocks to save memory
    box_list, num_box = ifginv.split2boxes(ifgram_stack,max_memory)
    closurephase =  np.zeros([length,width],np.complex64)
    #process block-by-block
    for i, box in enumerate(box_list):
            box_width  = box[2] - box[0]
            box_length = box[3] - box[1]
            print(box)
            if num_box > 1:
                print('\n------- processing patch {} out of {} --------------'.format(i+1, num_box))
                print('box width:  {}'.format(box_width))
                print('box length: {}'.format(box_length))

            closurephase[box[1]:box[3],box[0]:box[2]], numcp = sum_seq_closurePhase(SLC_list, date12_list_all, ifgram_stack, ref_phase,nl,box)
    # What is a good thredshold?
    # Assume that it's pure noise so that the phase is uniform distributed from -pi to pi.
    # The standard deviation of phase in each loop is pi/sqrt(3) (technically should be smaller because when forming loops there should be a reduction in phase variance)
    # The standard deviation of phase in cumulative wrapped closure phase is pi/sqrt(3)/sqrt(numcp) -- again another simplification assuming no correlation.
    # We use 3\delta as threshold -- 99.7% confidence

    threshold_pha = np.pi/np.sqrt(3)/np.sqrt(numcp)*numsigma

    mask = np.ones([length,width],np.float32)
    mask[np.abs(np.angle(closurephase))>threshold_pha] = 0 # this masks areas with potential bias
    mask[np.abs(np.abs(closurephase)/numcp < threshold_amp)] = 1 # this unmasks areas with low correlation (where it's hard to know wheter there is bias either)

    # save mask
    meta = dict(stack_obj.metadata)
    meta['FILE_TYPE'] = 'mask'
    ds_name_dict = {'cpmask': [np.float32, (length, width), mask],}
    writefile.layout_hdf5(os.path.join(outdir,'cpmask.h5'), ds_name_dict, meta)

    # also save the average closure phase
    ds_name_dict2 = {'phase': [np.float32, (length, width), np.angle(closurephase)],
                    'amplitude':[np.float32,(length,width),np.abs(closurephase)/numcp],}
    writefile.layout_hdf5(os.path.join(outdir,'avgwcp.h5'), ds_name_dict2, meta)

    return

# ouput wrapped, and unwrapped sequential closure phases, and cumulative closure phase time-series of connection-conn
def compute_unwrap_closurephase(ifgram_stack, conn, max_memory, outdir):
    stack_obj = ifgramStack(ifgram_stack)
    stack_obj.open()
    length, width = stack_obj.length, stack_obj.width
    meta = dict(stack_obj.metadata)
    date12_list = stack_obj.get_date12_list(dropIfgram=True)
    date12_list_all = stack_obj.get_date12_list(dropIfgram=False)
    print('scene length, width', length, width)
    ref_phase = stack_obj.get_reference_phase(unwDatasetName = 'unwrapPhase')
    refX = stack_obj.refX
    refY = stack_obj.refY
    # retrieve the list of SLC dates from ifgramStack.h5
    ifgram0 = date12_list[0]
    date1, date2 = ifgram0.split('_')
    SLC_list = [date1, date2]
    for ifgram in date12_list:
        date1, date2 = ifgram.split('_')
        if date1 not in SLC_list:
            SLC_list.append(date1)
        if date2 not in SLC_list:
            SLC_list.append(date2)
    SLC_list.sort()
    print('number of SLC found : ', len(SLC_list))
    print('first SLC: ', SLC_list[0])
    print('last  SLC: ', SLC_list[-1])

    # split igram_file into blocks to save memory
    box_list, num_box = ifginv.split2boxes(ifgram_stack,max_memory)

    closurephase =  np.zeros([len(SLC_list)-conn, length,width],np.float32)
    #process block-by-block
    for i, box in enumerate(box_list):
            box_width  = box[2] - box[0]
            box_length = box[3] - box[1]
            print(box)
            if num_box > 1:
                print('\n------- processing patch {} out of {} --------------'.format(i+1, num_box))
                print('box width:  {}'.format(box_width))
                print('box length: {}'.format(box_length))

            closurephase[:,box[1]:box[3],box[0]:box[2]] = seq_closurePhase(SLC_list, date12_list_all, ifgram_stack, ref_phase, conn, box)

    # directory
    cpdir = os.path.join(outdir, 'ClosurePhase')
    if not os.path.isdir(cpdir):
        os.mkdir(cpdir)

    cpdir_conn = os.path.join(cpdir,'con'+str(conn)+'_cp')
    if not os.path.isdir(cpdir_conn):
        os.mkdir(cpdir_conn)

    # filter and output
    for i in range(len(SLC_list)-conn):
        concpname = 'conn'+str(conn)+'_filt_'+'{:03}'.format(i)+'.int'
        concpdir = os.path.join(cpdir_conn,concpname)
        if not os.path.isfile(concpdir):
            kernel = gaussian_kernel(5,5,1,1)
            closurephase_filt = convolve(np.exp(1j*closurephase[i,:,:]), kernel)
            fid = open(concpdir,mode = 'wb')
            closurephase_filt.tofile(fid)
            fid.close()
            write_xml(concpdir, width, length, 1, 'CFLOAT','BIP')

    # compute phase sigma and output
    for i in range(len(SLC_list)-conn):
        concpname = 'conn'+str(conn)+'_filt_'+'{:03}'.format(i)+'.int'
        concpdir = os.path.join(cpdir_conn,concpname)
        concpcorname = 'conn'+str(conn)+'_filt_'+'{:03}'.format(i)+'.cor'
        concpcordir = os.path.join(cpdir_conn, concpcorname)
        if not os.path.isfile(concpcordir):
            estCoherence(concpdir, concpcordir)

  #  unwrap
    for i in range(len(SLC_list)-conn):
        concpname = 'conn'+str(conn)+'_filt_'+'{:03}'.format(i)+'.int'
        concpdir = os.path.join(cpdir_conn, concpname)
        concpcorname = 'conn'+str(conn)+'_filt_'+'{:03}'.format(i)+'.cor'
        concpcordir = os.path.join(cpdir_conn, concpcorname)
        concpunwname = 'conn'+str(conn)+'_filt_'+'{:03}'.format(i)+'.unw'
        concpunwdir = os.path.join(cpdir_conn, concpunwname)
        if not os.path.isfile(concpunwdir):
            unwrap_snaphu(concpdir,concpcordir,concpunwdir,length, width)

  # output accumulated unwrapped closure phase time-series
    cum_seq_unwclosurePhase(conn,cpdir_conn,length,width,refY,refX, SLC_list, meta)


def main(iargs = None):
    inps = cmd_line_parse(iargs)
    if inps.numsigma:
        numsigma = inps.numsigma
    else:
        numsigma = 3
    if inps.action == 'create_mask':
        creat_cp_mask(inps.ifgram_stack, inps.nl, inps.max_memory, numsigma, inps.episilon, inps.outdir)

    if inps.action == 'quick_biasEstimate':
        for conn in np.arange(2,inps.bw+2): # to make sure we have con-2 closure phase processed
            compute_unwrap_closurephase(inps.ifgram_stack, conn, inps.max_memory, inps.outdir)
        compute_unwrap_closurephase(inps.ifgram_stack, inps.nl, inps.max_memory, inps.outdir)
        # a quick solution to bias-correction and output diagonal component of Wr (how fast the bias-inducing signal decays with temporal baseline)
        quickbiascorrection(inps.ifgram_stack, inps.nl, inps.bw, inps.wvl, inps.max_memory, inps.outdir)

    if inps.action == 'biasEstimate':
        for conn in np.arange(2,inps.bw+2): # to make sure we have con-2 closure phase processed
            compute_unwrap_closurephase(inps.ifgram_stack, conn, inps.max_memory, inps.outdir)
        compute_unwrap_closurephase(inps.ifgram_stack, inps.nl, inps.max_memory, inps.outdir)
        # bias correction
        biascorrection(inps.ifgram_stack, inps.nl, inps.bw, inps.wvl, inps.max_memory, inps.outdir)

if __name__ == '__main__':
    main(sys.argv[1:])
