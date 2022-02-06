"""
This is a tool to clip the original .TIF images to small patches to build a dataset.
"""
import gdal
import mmcv
import numpy as np
import argparse
import sys
sys.path.append('.')
from datasets.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(description='clip into patches')
    parser.add_argument('-d', '--data_dir', required=True, help='root of data directory')
    parser.add_argument('-s', '--satellite', required=True, help='name of the satellite/dataset')
    parser.add_argument('-n', required=True, type=int, help='total number of images pairs')
    parser.add_argument('-p', '--patch_num', required=True, type=int,
                        help='random clip how many patches for one training scene')
    parser.add_argument('-r', '--rand_seed', type=int, default=0,
                        help='random seed to sample training patches')
    parser.add_argument('--no_low_train', action='store_true',
                        help='whether generate low-resolution training set or not')
    parser.add_argument('--no_full_train', action='store_true',
                        help='whether generate full-resolution training set or not')
    parser.add_argument('--no_low_test', action='store_true',
                        help='whether generate low-resolution testing set or not')
    parser.add_argument('--no_full_test', action='store_true',
                        help='whether generate ful-resolution testing set or not')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    in_dir = f'{args.data_dir}/Dataset/{args.satellite}/'

    with open(f'{in_dir}/clip_patch.log', 'w') as f:
        for k, v in args._get_kwargs():
            f.write(f'{k} = {v}\n')

    # train patch_size = 64
    patch_size = 64
    if not args.no_low_train:
        # train low-res
        cnt = 0
        out_dir = f"{args.data_dir}/Dataset/{args.satellite}/train_low_res"
        mmcv.mkdir_or_exist(out_dir)
        record = open(f'{out_dir}/record.txt', "w")
        rand = np.random.RandomState(args.rand_seed)
        # image_id from [1, n-1] is for train
        for num in range(1, args.n):
            mul = f'{in_dir}/{num}_mul.tif'
            lr = f'{in_dir}/{num}_lr.tif'
            lr_u = f'{in_dir}/{num}_lr_u.tif'
            pan = f'{in_dir}/{num}_pan.tif'

            dt_mul = gdal.Open(mul)
            dt_lr = gdal.Open(lr)
            dt_lr_u = gdal.Open(lr_u)
            dt_pan = gdal.Open(pan)

            img_mul = dt_mul.ReadAsArray()  # (c, h, w)
            img_lr = dt_lr.ReadAsArray()
            img_lr_u = dt_lr_u.ReadAsArray()
            img_pan = dt_pan.ReadAsArray()

            XSize = dt_lr.RasterXSize
            YSize = dt_lr.RasterYSize
            for _ in range(args.patch_num):
                x = rand.randint(XSize - patch_size)
                y = rand.randint(YSize - patch_size)

                save_image(f'{out_dir}/{cnt}_mul.tif',
                           img_mul[:, y * 4:(y + patch_size) * 4, x * 4:(x + patch_size) * 4])
                save_image(f'{out_dir}/{cnt}_lr_u.tif',
                           img_lr_u[:, y * 4:(y + patch_size) * 4, x * 4:(x + patch_size) * 4])
                save_image(f'{out_dir}/{cnt}_lr.tif',
                           img_lr[:, y:(y + patch_size), x:(x + patch_size)])
                save_image(f'{out_dir}/{cnt}_pan.tif',
                           img_pan[y * 4:(y + patch_size) * 4, x * 4:(x + patch_size) * 4])
                cnt += 1
            print("low-res train done %d" % num)
        record.write("%d\n" % cnt)
        record.close()

    if not args.no_full_train:
        # train full-res
        cnt = 0
        out_dir = f"{args.data_dir}/Dataset/{args.satellite}/train_full_res"
        mmcv.mkdir_or_exist(out_dir)
        record = open(f'{out_dir}/record.txt', "w")
        rand = np.random.RandomState(args.rand_seed)
        # image_id from [1, n-1] is for train
        for num in range(1, args.n):
            lr = f'{in_dir}/{num}_mul_o.tif'
            lr_u = f'{in_dir}/{num}_mul_o_u.tif'
            pan = f'{in_dir}/{num}_pan_o.tif'

            dt_lr = gdal.Open(lr)
            dt_lr_u = gdal.Open(lr_u)
            dt_pan = gdal.Open(pan)

            img_lr = dt_lr.ReadAsArray()
            img_lr_u = dt_lr_u.ReadAsArray()
            img_pan = dt_pan.ReadAsArray()

            XSize = dt_lr.RasterXSize
            YSize = dt_lr.RasterYSize
            for _ in range(args.patch_num):
                x = rand.randint(XSize - patch_size)
                y = rand.randint(YSize - patch_size)

                save_image(f'{out_dir}/{cnt}_lr_u.tif',
                           img_lr_u[:, y * 4:(y + patch_size) * 4, x * 4:(x + patch_size) * 4])
                save_image(f'{out_dir}/{cnt}_lr.tif',
                           img_lr[:, y:(y + patch_size), x:(x + patch_size)])
                save_image(f'{out_dir}/{cnt}_pan.tif',
                           img_pan[y * 4:(y + patch_size) * 4, x * 4:(x + patch_size) * 4])
                cnt += 1
            print("full-res train set done %d" % num)
        record.write("%d\n" % cnt)
        record.close()

    # test patch_size = 100
    patch_size = 100
    if not args.no_low_test:
        # test low-res
        cnt = 0
        out_dir = f"{args.data_dir}/Dataset/{args.satellite}/test_low_res"
        mmcv.mkdir_or_exist(out_dir)
        record = open(f'{out_dir}/record.txt', "w")
        # image_id from [0] is for test
        for num in range(1):
            mul = f'{in_dir}/{num}_mul.tif'
            lr = f'{in_dir}/{num}_lr.tif'
            lr_u = f'{in_dir}/{num}_lr_u.tif'
            pan = f'{in_dir}/{num}_pan.tif'

            dt_mul = gdal.Open(mul)
            dt_lr = gdal.Open(lr)
            dt_pan = gdal.Open(pan)
            dt_lr_u = gdal.Open(lr_u)

            img_mul = dt_mul.ReadAsArray()
            img_lr = dt_lr.ReadAsArray()
            img_pan = dt_pan.ReadAsArray()
            img_lr_u = dt_lr_u.ReadAsArray()

            XSize = dt_lr.RasterXSize
            YSize = dt_lr.RasterYSize

            row = 0
            col = 0
            # 按顺序切(100, 100)小块 overlap 1/2
            for y in range(0, YSize, patch_size // 2):
                if y + patch_size > YSize:
                    continue
                col = 0
                for x in range(0, XSize, patch_size // 2):
                    if x + patch_size > XSize:
                        continue
                    save_image(f'{out_dir}/{cnt}_mul.tif',
                               img_mul[:, y * 4:(y + 100) * 4, x * 4:(x + 100) * 4])
                    save_image(f'{out_dir}/{cnt}_lr_u.tif',
                               img_lr_u[:, y * 4:(y + 100) * 4, x * 4:(x + 100) * 4])
                    save_image(f'{out_dir}/{cnt}_lr.tif',
                               img_lr[:, y:(y + 100), x:(x + 100)])
                    save_image(f'{out_dir}/{cnt}_pan.tif',
                               img_pan[y * 4:(y + 100) * 4, x * 4:(x + 100) * 4])
                    cnt += 1
                    col += 1
                row += 1
            record.write("%d: %d * %d\n" % (num, row, col))
        record.write("%d\n" % cnt)
        record.close()
        print("full-res test set done")

    if not args.no_full_test:
        # test full-res
        cnt = 0
        out_dir = f"{args.data_dir}/Dataset/{args.satellite}/test_full_res"
        mmcv.mkdir_or_exist(out_dir)
        record = open(f'{out_dir}/record.txt', "w")
        # image_id from [0] is for test
        for num in range(1):
            lr = f'{in_dir}/{num}_mul_o.tif'
            lr_u = f'{in_dir}/{num}_mul_o_u.tif'
            pan = f'{in_dir}/{num}_pan_o.tif'

            dt_lr = gdal.Open(lr)
            dt_pan = gdal.Open(pan)
            dt_lr_u = gdal.Open(lr_u)

            img_lr = dt_lr.ReadAsArray()
            img_pan = dt_pan.ReadAsArray()
            img_lr_u = dt_lr_u.ReadAsArray()

            XSize = dt_lr.RasterXSize
            YSize = dt_lr.RasterYSize

            row = 0
            col = 0
            for y in range(0, YSize, patch_size // 2):
                if y + patch_size > YSize:
                    continue
                col = 0
                for x in range(0, XSize, patch_size // 2):
                    if x + patch_size > XSize:
                        continue

                    save_image(f'{out_dir}/{cnt}_lr_u.tif',
                               img_lr_u[:, y * 4:(y + 100) * 4, x * 4:(x + 100) * 4])
                    save_image(f'{out_dir}/{cnt}_lr.tif',
                               img_lr[:, y:(y + 100), x:(x + 100)])
                    save_image(f'{out_dir}/{cnt}_pan.tif',
                               img_pan[y * 4:(y + 100) * 4, x * 4:(x + 100) * 4])
                    cnt += 1
                    col += 1
                row += 1
            record.write("%d: %d * %d\n" % (num, row, col))
        record.write("%d\n" % cnt)
        record.close()
        print("full-res test set done")

    print("finish!!")
