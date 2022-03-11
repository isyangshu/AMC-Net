# -*- coding: utf-8 -*-
from __future__ import print_function

import cv2
import time
import os
import operator
import numpy as np
import argparse
from PIL import Image

__author__ = 'zj'

image_formats = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']


def is_pgm_file(in_path):
    if not os.path.isfile(in_path):
        return False
    if in_path is not str and not in_path.endswith('.pgm'):
        return False
    return True


def convert_pgm_by_PIL(in_path, out_path):
    if not is_pgm_file(in_path):
        raise Exception("%s 不是一个PGM文件" % in_path)
    # 读取文件
    im = Image.open(in_path)
    im.save(out_path)


def convert_pgm_P5(in_path, out_path):
    """
    将pgm文件转换成其它图像格式
    读取二进制文件，先读取幻数，再读取宽和高，以及最大值
    :param in_path: 输入pgm文件路径
    :param out_path: 输出文件路径
    """
    if not is_pgm_file(in_path):
        raise Exception("%s 不是一个PGM文件" % in_path)
    with open(in_path, 'rb') as f:
        # 读取两个字节 - 幻数，并解码成字符串
        magic_number = f.readline().strip().decode('utf-8')
        if not operator.eq(magic_number, "P5"):
            raise Exception("该图像有误")
        # 读取高和宽
        width, height = f.readline().strip().decode('utf-8').split(' ')
        width = int(width)
        height = int(height)
        # 读取最大值
        maxval = f.readline().strip()
        # 每次读取灰度值的字节数
        if int(maxval) < 256:
            one_reading = 1
        else:
            one_reading = 2
        # 创建空白图像，大小为(行，列)=(height, width)
        img = np.zeros((height, width))
        img[:, :] = [[ord(f.read(one_reading)) for j in range(width)] for i in range(height)]
        cv2.imwrite(out_path, img)
        print('%s save ok' % out_path)


def convert_pgm_P5_batch(in_dir, out_dir, res_format):
    """
    批量转换PGM文件
    :param in_dir: pgm文件夹路径
    :param out_dir: 输出文件夹路径
    :param res_format: 结果图像格式
    """
    if not os.path.isdir(in_dir):
        raise Exception('%s 不是路径' % in_dir)
    if not os.path.isdir(out_dir):
        raise Exception('%s 不是路径' % out_dir)
    if not res_format in image_formats:
        raise Exception('%s 暂不支持' % res_format)
    file_list = os.listdir(in_dir)
    for file_name in file_list:
        file_path = os.path.join(in_dir, file_name)
        # 若为pgm文件路径，那么将其进行格式转换
        if is_pgm_file(file_path):
            file_out_path = os.path.join(out_dir, os.path.splitext(file_name)[0] + '.' + res_format)
            convert_pgm_P5(file_path, file_out_path)
        # 若为目录，则新建结果文件目录，递归处理
        elif os.path.isdir(file_path):
            file_out_dir = os.path.join(out_dir, file_name)
            if not os.path.exists(file_out_dir):
                os.mkdir(file_out_dir)
            convert_pgm_P5_batch(file_path, file_out_dir, res_format)
        else:
            pass
    print('batch operation over')


if __name__ == '__main__':
    script_start_time = time.time()

    parser = argparse.ArgumentParser(description='Format Converter - PGM')

    ### Positional arguments

    ### Optional arguments

    parser.add_argument('-i', '--input', type=str, help='Path to the pgm file')
    parser.add_argument('-o', '--output', type=str, help='Path to the result file')
    parser.add_argument('--input_dir', type=str, help='Dir to the pgm files')
    parser.add_argument('--output_dir', type=str, help='Dir to the result files')
    parser.add_argument('-f', '--format', default='png', type=str, help='result image format')
    parser.add_argument('-b', '--batch', action="store_true", default=True, help='Batch processing')

    args = vars(parser.parse_args())
    # print(args)
    in_path = args['input']
    out_path = args['output']

    isbatch = args['batch']
    in_dir = args['input_dir']
    out_dir = args['output_dir']
    res_format = args['format']

    if in_path is not None and out_path is not None:
        # 转换单个pgm文件格式
        convert_pgm_P5(in_path, out_path)
        # convert_pgm_by_PIL(in_path, out_path)
    elif isbatch:
        # 批量转换
        convert_pgm_P5_batch(in_dir, out_dir, res_format)
    else:
        print('请输入相应参数')

    print('Script took %s seconds.' % (time.time() - script_start_time,))
