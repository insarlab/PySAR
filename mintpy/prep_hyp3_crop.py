#!/usr/bin/env python3
############################################################
# Program is part of VEG-tools                             #
# Copyright (c) 2021,         VEG                          #
# Author: Xie Lei, Wang Jiageng,  Jun 2022                 #
# A special thank to Benjy Marks                           #
# The environment needs to install hyp3lib                 #
############################################################
import argparse
import os
import shutil
import glob
from zipfile import ZipFile
import numpy as np
import pathlib
from osgeo import gdal
from osgeo import osr
from hyp3lib.cutGeotiffs import cutFiles
import mintpy.prep_hyp3


def gdal_clip(source_path, target_extent, output_path=None, target_format='GTiff', flag_prompt=False):
    """
    Generate and run gdal command for clip img
    :param source_path: source path need to be clipped
    :param target_extent: e_min north_min e_max north_max
    :param output_path: output path, can be None
    :param target_format: Default GTiff
    :param flag_prompt: Default False
    :return:
    """
    # gdalwarp cmd
    # e.g., gdalwarp -te 78.25 30.22 78.67 30.64 -of GTiff input output

    if output_path is None:
        p = pathlib.Path(os.path.abspath(source_path))
        output_path = os.path.join(p.parent, '%s_clip%s' % (p.stem, p.suffix))

    cmd = 'gdalwarp -te %s %s %s %s -of %s %s %s' % (
        target_extent[0], target_extent[1], target_extent[2], target_extent[3],
        target_format, source_path, output_path)
    if flag_prompt:
        print(':: GDAL cmd: %s' % cmd)
        print('-' * 50)

    os.system(cmd)
    return True


def read_img(filename):
    dataset = gdal.Open(filename)

    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize

    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)

    minx = im_geotrans[0]
    maxy = im_geotrans[3]
    maxx = minx + im_geotrans[1] * dataset.RasterXSize
    miny = maxy + im_geotrans[5] * dataset.RasterYSize

    im_extent = [minx, maxx, miny, maxy]
    del dataset
    return im_proj, im_geotrans, im_data, im_extent


def hyp3_reprojection(utm_path, latlon_path):
    gdal.AllRegister()
    src_data = gdal.Open(utm_path)
    src_srs_wkt = src_data.GetProjection()
    src_srs = osr.SpatialReference()
    src_srs.ImportFromWkt(src_srs_wkt)
    src_width = src_data.RasterXSize
    src_height = src_data.RasterYSize
    src_trans = src_data.GetGeoTransform()
    origin_lx_src = src_trans[0]
    origin_ty_src = src_trans[3]
    pixl_w_src = src_trans[1]
    pixl_h_src = src_trans[5]
    origin_rx_src = origin_lx_src + pixl_w_src * src_width
    origin_by_src = origin_ty_src + pixl_h_src * src_height
    driver = gdal.GetDriverByName("GTiff")
    driver.Register()
    dst_data = driver.Create(latlon_path, src_width, src_height, 1, gdal.GDT_Float32)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(4326)
    dst_srs.ImportFromProj4("proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")
    ct = osr.CoordinateTransformation(src_srs, dst_srs)
    origin_lx_dst, origin_ty_dst, temp = ct.TransformPoint(origin_lx_src, origin_ty_src)
    origin_rx_dst, origin_by_dst, temp1 = ct.TransformPoint(origin_rx_src, origin_by_src)
    pixl_w_dst = (origin_rx_dst - origin_lx_dst) / src_width
    pixl_h_dst = (origin_by_dst - origin_ty_dst) / src_height
    dst_trans = [origin_lx_dst, pixl_w_dst, 0, origin_ty_dst, 0, pixl_h_dst]

    dst_srs_wkt = dst_srs.ExportToWkt()
    dst_data.SetGeoTransform(dst_trans)
    dst_data.SetProjection(dst_srs_wkt)
    data = src_data.GetRasterBand(1).ReadAsArray()
    dst_data.WriteArray(data)
    dst_data.FlushCache()


def unzip_crop(workdir: str, geo_box, target_dir):
    #
    # lower left
    lonlat_ll = [geo_box[2], geo_box[1]]
    # up right
    lonlat_ur = [geo_box[3], geo_box[0]]

    os.chdir(workdir)
    file_list = glob.glob('*_*')
    file_list.sort()
    if os.path.isdir(target_dir) and target_dir != '':
        shutil.rmtree(target_dir)
    ################
    else:
        target_dir = os.path.join(pathlib.Path(os.path.abspath(
            workdir)).parent, pathlib.Path(os.path.abspath(workdir)).stem + '_clip')

    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    target_extent = (lonlat_ll[0], lonlat_ll[1], lonlat_ur[0], lonlat_ur[1])

    print(" Wait")
    # conduct the clip operation
    for i in range(len(file_list)):
        tmp_tif_file_list = glob.glob('%s/*.tif' % file_list[i])
        tmp_target_dir = os.path.join(target_dir, file_list[i])
        if os.path.isdir(tmp_target_dir):
            shutil.rmtree(tmp_target_dir)
        os.mkdir(tmp_target_dir)
        for tmp_file in tmp_tif_file_list:
            p = pathlib.Path(os.path.abspath(tmp_file))
            tmp_output_path = os.path.join(
                tmp_target_dir, '%s_clip%s' % (p.stem, p.suffix))
            # Transfer the reference from WGS84 UTM into WGS84
            p_out = os.path.abspath(tmp_output_path)[:-9] + '_latlon.tif'
            hyp3_reprojection(os.path.abspath(p), p_out)
            _, _, im_data, im_extent = read_img(p_out)
            im_extent = crop_region_banpick(target_extent, im_extent)
            if target_extent == (-181, -91, 181, 91):
                p_rename = os.path.abspath(p_out)[:-4] + '.tif'
                os.rename(p_out, p_rename)
                # leave the generate of *_clip.tif below
            else:
                # transfer *.clip.tif  into *_clip.tif
                # to meet the requirement of prep_hyp3
                p_rename = os.path.abspath(p_out)[:-4] + '_clip.tif'
                gdal_clip(p_out, im_extent, output_path=p_rename)
                os.remove(p_out)
        # copy the txt file into
        txt_name = file_list[i] + '/' + file_list[i] + '.txt'
        txt_name1 = file_list[i] + '.txt'
        txt_path0 = os.path.join(workdir, txt_name)
        txt_path1 = os.path.join(tmp_target_dir, txt_name1)
        shutil.copy(txt_path0, txt_path1)
        print(':: [%s] Transferd and Clipped files in: %s' % (i + 1, os.path.abspath(file_list[i])))

    # do the clip for transformed but clipped via hyp3lib.cutFiles
    files = []
    if target_dir == '':
        wk1_dir = workdir + '_clip'
    else:
        wk1_dir = target_dir

    print("current path:  " + workdir)
    if os.path.exists(wk1_dir):
        filetypes = ['unw_phase', 'corr', 'dem', 'lv_theta', 'water_mask']
        for root, _, file_list in os.walk(wk1_dir):
            for file in file_list:
                for f in filetypes:
                    if f + '_latlon.tif' in file and f + '_latlon.tif.xml' not in file:
                        full_path = os.path.join(root, file)  # Get the full path to the file
                        files.append(full_path)
        print("There are " + str(len(files)) + " satisfied files")
        # Generate
        cutFiles(files)
        print('The crop process has finished')
    # generate _clip.tif.rsc
    for i in files:
        if target_extent == (-181, -91, 181, 91):
            mintpy.prep_hyp3.main(files)
        else:
            clip_files = ['', i[:-4] + '_clip.tif']
            mintpy.prep_hyp3.main(clip_files)


def unzip_files(work_dir, target_postfix_list):
    # return the path that contains unziped files

    os.chdir(work_dir)
    file_list = glob.glob('*.zip')
    file_list.sort()

    primary_date_list = [file_list[i][5:13] for i in range(len(file_list))]
    secondary_date_list = [file_list[i][21:29] for i in range(len(file_list))]
    extract_name_list = ['%s_%s' % (primary_date_list[i], secondary_date_list[i])
                         for i in range(len(file_list))]

    target_dir = os.path.join(os.path.dirname(pathlib.Path(work_dir)), 'unzip')
    if os.path.isdir(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(target_dir)
    for i in range(len(file_list)):
        print(':: [%s] Unpack file: %s' % (i + 1, file_list[i]))
        # make a new directory for the target unzip files
        current_target_path = os.path.join(target_dir, extract_name_list[i])
        if os.path.isdir(current_target_path):
            shutil.rmtree(current_target_path)
        os.mkdir(current_target_path)
        # unzip target files
        with ZipFile(file_list[i], 'r') as zipObject:
            # post_idx = 0  # index for postfix
            zip_file_list = zipObject.namelist()
            for tmp_file in zip_file_list:
                if tmp_file.endswith(tuple(target_postfix_list)):
                    # get tmp_post name
                    # get index -> get name
                    tmp_post = np.array([tmp_file.find(tmp)
                                         for tmp in target_postfix_list])
                    tmp_post = tmp_post[tmp_post != -1][0]
                    tmp_post = tmp_file[tmp_post:]
                    # unzip step
                    zipObject.extract(tmp_file, target_dir)
                    # copy and rename target file
                    copy_path = current_target_path
                    copy_path = os.path.join(copy_path, '%s_%s' % (
                        extract_name_list[i], tmp_post))
                    source_path = glob.glob('%s/S1*/*%s' %
                                            (target_dir, tmp_post))[0]
                    shutil.copy(source_path, copy_path)

                # add a judgement of .txt and not .md.txt
                # copy the renamed.txt into a correct area
                if tmp_file.endswith('.txt') and not tmp_file.endswith('README.md.txt'):
                    print("Have not done yet")

            # delete tmp file
            tmp_unzip_path = glob.glob('%s/S1*' % target_dir)[0]
            shutil.rmtree(tmp_unzip_path)
    return target_dir


def crop_region_banpick(geobox, imgbox):
    region = [0.0, 0.0, 0.0, 0.0]
    if geobox[0] == -181:
        region[0] = imgbox[2]
    if geobox[1] == -91:
        region[1] = imgbox[0]
    if geobox[2] == 181:
        region[2] = imgbox[3]
    if geobox[3] == 91:
        region[3] = imgbox[1]
    return region


def asf_crop(work_dir, geobox, crop_state, out_path):
    # geobox represent the n,s,w,e
    # e.g., wrapped_phase.tif, inc_map.tif, water_mask.tif,
    if geobox[:] == [91, -91, -181, 181] and crop_state == 1:
        print("the crop state code is crop but the region has not been input")
        exit()
    elif crop_state == 0:
        geobox = [91, -91, -181, 181]
    # adjusted geobox before, the bounding should not need to be adjusted after.
    target_postfix_list = ['corr.tif', 'unw_phase.tif', 'dem.tif', 'lv_theta.tif', 'water_mask.tif']
    target_path = out_path

    unzip_crop(workdir=work_dir, geo_box=geobox, target_dir=target_path)

    return 0


def main():
    str_h = """Example usage: 
    asf_crop.py --i /Examples/asf_hyp3_unzip/ --o /Examples/asf_utm_hyp3/
    asf_crop.py --i /Examples/asf_hyp3_unzip/ --o /Examples/asf_utm_hyp3/ --c 1 --e -91.5  --n 1.5 
    asf_crop.py --i /Examples/asf_hyp3_unzip/ --o /Examples/asf_utm_hyp3/ --c 1 --e -91.5 --w -90 --n 1.5 --s -0.75  
    
    # rely on asf-hyp3-lib, please install it at here 
    
    # we plan to add a new script to 
    # add a filter for --sd start-date
    # add a filter for --ed end-date
    # add a filter for area co-coverage, although this may be covered by frame and path.
    # add a if-else function for invalid area recognise.
    # to avoid some edge set by 0.0, we use impossible value for N,S,W,E. This may be improved soon.
    
    # need to use the unzipped folder to run the script.
    
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter, description=str_h)
    parser.add_argument('--i', type=str, help="The input folder path that contains all unzipped datas")
    parser.add_argument('--o', type=str, default='', help="The output folder path that contains all the cropped files")
    parser.add_argument('--c', type=int, default=0, help="0 = do not crop, 1 = do the crop")
    parser.add_argument('--n', type=float, default=91, help="north latitude")
    parser.add_argument('--s', type=float, default=-91, help="south latitude")
    parser.add_argument('--w', type=float, default=-181, help="west latitude")
    parser.add_argument('--e', type=float, default=181, help="east latitude")
    args = parser.parse_args()

    # edge = n,s,w,e

    try:
        asf_crop(work_dir=args.i, geobox=[args.n, args.s, args.w, args.e], crop_state=args.c, out_path=args.o)
    except Exception as e:
        print("Something wrong")
        print(e)
    finally:
        print("Crop process has finished")
    return 0


if __name__ == '__main__':
    main()
