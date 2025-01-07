############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Yang Qingyue, Hu Changyang, 2024                 #
############################################################


import os

import numpy as np
import scipy
from scipy.interpolate import UnivariateSpline, griddata

from mintpy.objects import timeseries
from mintpy.utils import readfile, writefile

############################################################################

def read_topographic_data(geom_file):
    print('read height from file: '+geom_file)
    dem = readfile.read(geom_file, datasetName='height', print_msg=False)[0]
    Na, Nr = dem.shape
    dataLine = dem.flatten()

    index = np.where(~np.isnan(dataLine))[0]
    x = np.ceil(index / Na).astype(int)
    y = np.mod(index, Na).astype(int)

    xi = np.arange(0, Nr)
    yi = np.arange(0, Na)

    dataInp = griddata((x, y), dataLine[index], (xi[None, :], yi[:, None]), method='cubic')
    dem[np.isnan(dem)] = dataInp[np.isnan(dem)]
    return dem

def estimate_local_slope(dem, ts_data, inps, n_ref, meta):
    """Estimate local slope based on texture correlation for each acquisition of timeseries
    Parameters: dem     : 2D array in size of (          length, width)
                ts_data : 3D array in size of (num_date, length, width)
                inps    : Namespace
                n_ref   : Index of reference SAR image in timeseries
                meta    : Metadata of timeseries
    Returns:    k_htc   : 3D array in size of (num_date, length, width), length and width depend on the window size and overlap ratio
    """

    # Filtering parameters for obtaining high-frequency texture
    w1 = 9  # slope filtering parameters used in gaussian filter
    w2 = 13 # texture correlation filtering parameters used in gaussian filter
    res_step = 0.5 # step size for mask updating
    Iteration = 10 # iteration times for mask updating
    rg = 40 # range for slope searching
    step = 0.0001 # step size for slope searching
    # w2 = 2 * int(truncate * sigma + 0.5) + 1
    truncate = ((w2 - 1)/2 - 0.5)/w1 # truncation factor for gaussian filter

    lamda = float(meta['WAVELENGTH'])
    ref_y = int(meta['REF_Y'])
    ref_x = int(meta['REF_X'])

    N, Na, Nr = ts_data.shape
    W = inps.windowsize
    w = (W-1)/2 # half window size
    overlap = round(inps.overlapratio*(2*w+1))

    print('reading mask from file: '+inps.mask_file)
    mask_coh = readfile.read(inps.mask_file, datasetName='temporalCoherence')[0]
    mask_coh = np.where(mask_coh < 0.87, np.nan, 1)
    mask = mask_coh

    Na_C = np.arange(w + 1, Na + 1, 2 * w - overlap)
    Nr_C = np.arange(w + 1, Nr + 1, 2 * w - overlap)

    k_LLF = np.zeros((N, len(Na_C), len(Nr_C)))  # LLF: slope values in spatially discrete distributions
    d_LLF = np.zeros((N, len(Na_C), len(Nr_C)))  # LLF: intercept values in spatially discrete distributions
    k_htc = np.zeros((N, len(Na_C), len(Nr_C)))  # HTC: slope values in spatially discrete distributions

    ts_data = 4 * np.pi / lamda * ts_data[:N, :, :]
    reference_value = ts_data[:, ref_y-1, ref_x-1]
    ts_data = ts_data - reference_value[:, np.newaxis, np.newaxis]
    ts_data[np.isnan(ts_data)] = 0

    for na, nac in enumerate(Na_C):
        for nr, nrc in enumerate(Nr_C):
            # patch
            u = nac - w
            U = np.where(u < 1, 1, u)
            d = nac + w
            D = np.where(d > Na, Na, d)
            l = nrc - w
            L = np.where(l < 1, 1, l)
            r = nrc + w
            R = np.where(r > Nr, Nr, r)

            if na == len(Na_C) - 1:
                D = Na
            if nr == len(Nr_C) - 1:
                R = Nr
            U = int(U)
            D = int(D)
            L = int(L)
            R = int(R)
            tmp = np.full((Na, Nr), np.nan)
            tmp[U-1:D, L-1:R] = 1
            mask_process = mask.copy()
            mask_process[np.isnan(tmp)] = np.nan

            # solve
            result_compare = np.zeros((N, 4))  # line0-3: k_LLF, k_ILLF(iteration), k_htc, d_LLF
            for n in range(N):
                if n == n_ref:
                    continue
                # ----------- initial value ----------- #
                # mask
                mask_std = np.ones((Na, Nr)) * mask_process

                # iterative linear fitting
                # res+step = 0.5 # step [rad]
                # Iteration = 10
                for i in range(Iteration):
                    # linear fitting
                    phase_tmp = ts_data[n, :, :] * mask_std
                    dem_tmp = dem * mask_std
                    valid_idx = ~np.isnan(phase_tmp) & ~np.isnan(dem_tmp)
                    if np.nansum(valid_idx) == 0:
                        continue
                    coe = np.polyfit(dem[valid_idx], phase_tmp[valid_idx], 1)
                    cor_tmp = phase_tmp - coe[0]*dem_tmp - coe[1]

                    # result recording
                    if i == 0:
                        result_compare[n, 0] = coe[0]
                        result_compare[n, 3] = coe[1]

                    # mask uploading
                    max_tmp = np.nanmax(np.abs(cor_tmp)) #TODO
                    max_tmp = np.where(max_tmp > res_step, max_tmp - res_step, max_tmp)
                    mask_std[np.abs(cor_tmp) > max_tmp] = np.nan

                    if np.nansum(~np.isnan(mask_std)) < np.nansum(~np.isnan(mask_process)) / 10: #TODO
                        break

                # ----------- texture correlation ----------- #
                mask_tmp = mask[U-1:D, L-1:R]

                A = dem[U-1:D, L-1:R]
                A_LP = scipy.ndimage.gaussian_filter(A, sigma=w1, truncate=truncate, mode='nearest')
                A = A - A_LP
                A_line = A[~np.isnan(mask_tmp)]
                A_line = A_line / np.linalg.norm(A_line)

                # range = 40
                # step = 0.0001
                # left
                k_left = -rg * step + coe[0]
                phase_ts_Scor_tmp = ts_data[n, :, :] - k_left * dem
                C = phase_ts_Scor_tmp[U-1:D, L-1:R]
                C[np.isnan(C)] = 0
                C_LP = scipy.ndimage.gaussian_filter(C, sigma=w1, truncate=truncate, mode='nearest')
                C = C - C_LP
                C_line = C[~np.isnan(mask_tmp)]
                C_line = C_line / np.linalg.norm(C_line)
                conv_AC = np.nansum(A_line * C_line)
                record_left = np.abs(conv_AC)
                # right
                k_right = rg * step + coe[0]
                phase_ts_Scor_tmp = ts_data[n, :, :] - k_right * dem
                C = phase_ts_Scor_tmp[U-1:D, L-1:R]
                C[np.isnan(C)] = 0
                C_LP = scipy.ndimage.gaussian_filter(C, sigma=w1, truncate=truncate, mode='nearest')
                C = C - C_LP
                C_line = C[~np.isnan(mask_tmp)]
                C_line = C_line / np.linalg.norm(C_line)
                conv_AC = np.nansum(A_line * C_line)
                record_right = np.abs(conv_AC)

                if record_left >= record_right:
                    k = np.arange(0, rg + 1) * step + coe[0]
                    record = np.zeros(len(k))
                else:
                    k = np.arange(-rg, 1) * step + coe[0]
                    record = np.zeros(len(k))

                for i in enumerate(k):
                    phase_ts_Scor_tmp = ts_data[n, :, :] - k[i] * dem
                    C = phase_ts_Scor_tmp[U-1:D, L-1:R]
                    C[np.isnan(C)] = 0
                    C_LP = scipy.ndimage.gaussian_filter(C, sigma=w1, truncate=truncate, mode='nearest')
                    C = C - C_LP
                    C_line = C[~np.isnan(mask_tmp)]
                    C_line = C_line / np.linalg.norm(C_line)

                    conv_AC = np.nansum(A_line * C_line)
                    record[i] = np.abs(conv_AC)

                # slope
                index = np.argmin(record)
                result_compare[n, 1] = coe[0]
                result_compare[n, 2] = k[index]

            # result recording
            k_LLF[:, na, nr] = result_compare[:, 0]
            d_LLF[:, na, nr] = result_compare[:, 3]
            k_htc[:, na, nr] = result_compare[:, 2]
    return k_htc

def slope_interpolation(ts_data, inps, k_htc):
    """Estimated local slope interpolation to obtain full-scale slope
    Parameters: ts_data : 3D array in size of (num_date, length, width)
                inps    : Namespace
                k_htc   : 3D array in size of (num_date, length, width), length and width depend on the window size and overlap ratio
    Returns:    k_htc_interp   : 3D array in size of (num_date, length, width)
    """

    # Filtering parameters for obtaining high-frequency texture correlation
    sigma_slope = 7 # standard deviation for gaussian filter
    w_slope = 7 # window size for gaussian filter
    truncate_slope = ((w_slope - 1)/2 - 0.5)/sigma_slope # truncation factor for gaussian filter
    N, Na, Nr = ts_data.shape
    W = inps.windowsize
    w = (W-1)/2
    overlap = round(inps.overlapratio*(2*w+1))
    Na_C = np.arange(w + 1, Na + 1, 2 * w - overlap)
    Nr_C = np.arange(w + 1, Na + 1, 2 * w - overlap)
    # K_htc
    k_htc_interp = np.zeros((N, Na, Nr))
    for n in range(0, N):
        # filtering
        k_htc_filt = scipy.ndimage.gaussian_filter(k_htc[n, :, :], sigma=sigma_slope, truncate=truncate_slope, mode='nearest')
        # interpolation
        result = np.zeros((Na, Nr))
        coords_y = Na_C - 1
        coords_x = Nr_C - 1
        # Row interpolation
        for i in range(k_htc_filt.shape[0]):
            spline_x = UnivariateSpline(coords_x, k_htc_filt[i, :], k=3, s=0)
            result[i, :] = spline_x(np.arange(0, Nr))
        # Column interpolation
        for j in range(Nr):
            spline_y = UnivariateSpline(coords_y, result[:k_htc_filt.shape[0], j], k=3, s=0)
            result[:, j] = spline_y(np.arange(0, Na))

        k_htc_interp[n] = result
    return k_htc_interp

def intercept_filtering(dem, ts_data, inps, k_htc_interp, meta):
    """Estimate and correct tropospheric delay using intercept filtering
    Parameters: dem     : 2D array in size of (          length, width)
                ts_data : 3D array in size of (num_date, length, width)
                inps    : Namespace
                k_htc_interp : 3D array in size of (num_date, length, width)
                meta    : Metadata of timeseries
    Returns:    phase_ts_htc_low   : 3D array in size of (num_date, length, width)
    """

    # Filtering parameters for obtaining high-frequency texture correlation
    sigma_intercept = 251 # standard deviation for gaussian filter
    w_intercept = 251 # window size for gaussian filter
    truncate_intercept = ((w_intercept - 1)/2 - 0.5)/sigma_intercept # truncation factor for gaussian filter
    ref_y = int(meta['REF_Y'])
    ref_x = int(meta['REF_X'])
    lamda = float(meta['WAVELENGTH'])
    N = ts_data.shape[0]

    ts_data = 4 * np.pi / lamda * ts_data[:N, :, :]
    reference_value = ts_data[:, ref_y-1, ref_x-1]
    ts_data = ts_data - reference_value[:, np.newaxis, np.newaxis]
    ts_data[np.isnan(ts_data)] = 0

    phase_ts_htc_low = ts_data.copy()
    intercept = np.zeros(phase_ts_htc_low.shape)
    for n in range(0, N):
        tmp = ts_data[n, :, :] - k_htc_interp[n, :, :] * dem
        tmp_filt = scipy.ndimage.gaussian_filter(tmp, sigma=w_intercept, truncate=truncate_intercept, mode='nearest')
        tmp = tmp - tmp_filt
        phase_ts_htc_low[n, :, :] = tmp
        intercept[n, :, :] = tmp_filt
    reference_phase = phase_ts_htc_low[:, ref_y - 1, ref_x - 1]
    phase_ts_htc_low = phase_ts_htc_low - reference_phase[:, np.newaxis, np.newaxis] #TODO

    return phase_ts_htc_low


############################################################################

def run_tropo_local_texture(inps):

    # read time-series data
    ts_obj = timeseries(inps.timeseries_file)
    ts_obj.open()
    ts_data = ts_obj.read()
    inps.date_list = list(ts_obj.dateList)
    n_ref = inps.date_list.index(ts_obj.metadata['REF_DATE'])

    # read topographic data (DEM)
    dem = read_topographic_data(inps.geom_file)

    # slope estimation
    k_htc = estimate_local_slope(dem, ts_data, inps, n_ref, ts_obj.metadata)

    # slope interpolation
    k_htc_interp = slope_interpolation(ts_data, inps, k_htc)

    # intercept filtering
    ts_htc_low = intercept_filtering(dem, ts_data, inps, k_htc_interp, ts_obj.metadata)
    lamda = float(ts_obj.metadata['WAVELENGTH'])
    ts_htc_data = lamda / 4 /np.pi * ts_htc_low

    # write corrected time-series file
    meta = dict(ts_obj.metadata)
    if not inps.outfile:
        fbase = os.path.splitext(inps.timeseries_file)[0]
        inps.outfile = f'{fbase}_tropolocaltexture.h5'

    writefile.write(
        ts_htc_data,
        out_file=inps.outfile,
        metadata=meta,
        ref_file=inps.timeseries_file
    )

    return
