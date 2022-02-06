"""
This is a tool to get a quick visual result.
For official visual comparison, please use professional softwares like ENVI, ArcMap, etc.
"""
import gdal
from PIL import Image
import numpy as np
import argparse
from numba import jit
import mmcv

import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(description='convert .TIF to visible .png')
    parser.add_argument('-s', '--src_file', required=True, help='source file name')
    parser.add_argument('-n', '--num', type=int, default=1, help='total image id to convert')
    parser.add_argument('-d', '--dst_dir', default='examples', help='saved directory')
    return parser.parse_args()


@jit(nopython=True)
def linear(data):
    img_new = np.zeros(data.shape)
    sum_ = data.shape[1] * data.shape[2]
    for i in range(0, data.shape[0]):
        num = np.zeros(5000)
        prob = np.zeros(5000)
        for j in range(0, data.shape[1]):
            for k in range(0, data.shape[2]):
                num[data[i, j, k]] = num[data[i, j, k]] + 1
        for tmp in range(0, 5000):
            prob[tmp] = num[tmp] / sum_
        min_val = 0
        max_val = 0
        min_prob = 0.0
        max_prob = 0.0
        while min_val < 5000 and min_prob < 0.2:
            min_prob += prob[min_val]
            min_val += 1
        while True:
            max_prob += prob[max_val]
            max_val += 1
            if max_val >= 5000 or max_prob >= 0.98:
                break
        for m in range(0, data.shape[1]):
            for n in range(0, data.shape[2]):
                if data[i, m, n] > max_val:
                    img_new[i, m, n] = 255
                elif data[i, m, n] < min_val:
                    img_new[i, m, n] = 0
                else:
                    img_new[i, m, n] = (data[i, m, n] - min_val) / (max_val - min_val) * 255
    return img_new


if __name__ == '__main__':
    args = parse_args()
    tot = args.num
    mmcv.mkdir_or_exist(args.dst_dir)

    for delta in range(tot):
        src_file = args.src_file
        dir_name = osp.dirname(args.src_file)
        file_name = osp.basename(args.src_file)
        if delta is not 0:
            image_id = int(file_name.split('_')[0]) + delta
            file_name = str(image_id) + file_name[file_name.find('_'):]
            src_file = osp.join(dir_name, file_name)
        dst_file = file_name.split('.')[0] + '.png'
        dst_file = osp.join(args.dst_dir, dst_file)

        img = np.array(gdal.Open(src_file).ReadAsArray())
        print(f'source file shape: {img.shape}')

        if img.ndim is 2:
            h, w = img.shape
            img = img.reshape(1, h, w)
        img = linear(img)

        if img.shape[0] in [4, 8]:
            img = img[(2, 1, 0), :, :]
            img = img.transpose(1, 2, 0)
        elif img.shape[0] is 1:
            _, h, w = img.shape
            img = img.reshape(h, w)

        img = np.array(img, dtype=np.uint8)
        img = Image.fromarray(img)
        img.save(dst_file)
        print(f'saved file: {dst_file}')
