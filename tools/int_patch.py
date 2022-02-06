"""
integrate model output patches into the original remote sensing size.
"""
import argparse
import numpy as np
import gdal
import sys
sys.path.append('.')
from datasets.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(description='integrate patches into one big image')
    parser.add_argument('-d', '--dir', required=True, help='directory of input patches')
    parser.add_argument('-m', '--model', required=True, help='name of the model')
    parser.add_argument('-c', '--col', required=True, type=int, help='how many columns')
    parser.add_argument('-r', '--row', required=True, type=int, help='how many rows')
    parser.add_argument('--ms_chan', default=4, type=int, help='how many channels of MS')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    patch_size = 400  # size of the model output

    y_size = patch_size // 2 * (args.row - 1) + patch_size
    x_size = patch_size // 2 * (args.col - 1) + patch_size
    out = np.zeros(shape=[y_size, x_size, args.ms_chan], dtype=np.float32)
    cnt = np.zeros(shape=out.shape, dtype=np.float32)
    print(out.shape)

    i = 0
    y = 0
    for _ in range(args.row):
        x = 0
        for __ in range(args.col):
            ly = y
            ry = y + patch_size
            lx = x
            rx = x + patch_size
            cnt[ly:ry, lx:rx, :] = cnt[ly:ry, lx:rx, :] + 1
            img = f'{args.dir}/{i}_mul_hat.tif'
            img = gdal.Open(img).ReadAsArray().transpose(1, 2, 0)
            img = np.array(img, dtype=np.float32)
            out[ly:ry, lx:rx, :] = out[ly:ry, lx:rx, :] + img

            i = i + 1
            x = x + patch_size // 2
        y = y + patch_size // 2
    out = out / cnt
    save_image(f"{args.dir}/{args.model}.tif", out.transpose(2, 0, 1))

    print("finish!!")
