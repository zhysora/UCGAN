"""
This is a tool to preprocess the original giant remote sensing .TIF images.
"""
import gdal
import cv2
import argparse
import mmcv
import sys
sys.path.append('.')
from datasets.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(description='handle raw images')
    parser.add_argument('-d', '--data_dir', required=True, help='root of data directory')
    parser.add_argument('-s', '--satellite', required=True, help='name of the satellite/dataset')
    parser.add_argument('-n', required=True, type=int, help='total number of images pairs')
    parser.add_argument('-m', '--mode', choices=['Gaussian', 'Bicubic'], default='Gaussian',
                        help='type of interpolation')
    parser.add_argument('--reduce_raw', action='store_true',
                        help='sometimes the raw resolution image for testing is too big, '
                             'use this to reduce to 1/4')
    return parser.parse_args()


def down_sample(img, mode='Bicubic'):  # Gaussian pyramid construction
    if mode == 'Bicubic':
        h, w = img.shape[:2]
        return cv2.resize(img, (w // 4, h // 4), interpolation=cv2.INTER_CUBIC)
    elif mode == 'Gaussian':
        return cv2.pyrDown(cv2.pyrDown(img))


def up_sample(img, mode='Bicubic'):  # Gaussian pyramid construction
    if mode == 'Bicubic':
        h, w = img.shape[:2]
        return cv2.resize(img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
    elif mode == 'Gaussian':
        return cv2.pyrUp(cv2.pyrUp(img))


if __name__ == "__main__":
    args = parse_args()
    in_dir = f'{args.data_dir}/Raw/{args.satellite}'
    out_dir = f'{args.data_dir}/Dataset/{args.satellite}'
    mmcv.mkdir_or_exist(out_dir)

    with open(f'{out_dir}/handle_raw.log', 'w') as f:
        for k, v in args._get_kwargs():
            f.write(f'{k} = {v}\n')

    # NO.0 for testing, others for training
    for i in range(0, args.n):
        # MUL -> mul(1), lr(1/4), lr_u(1/4*4)
        # PAN -> pan(1/4)
        # origin
        # MUL(crop 1/4) -> mul_o(1), mul_o_u(1*4)
        # PAN(crop 1/4) -> pan_o(1)
        newMul = f'{out_dir}/{i}_mul.tif'
        newLR = f'{out_dir}/{i}_lr.tif'
        newLR_u = f'{out_dir}/{i}_lr_u.tif'
        newPan = f'{out_dir}/{i}_pan.tif'

        newMul_o = f'{out_dir}/{i}_mul_o.tif'
        newMul_o_u = f'{out_dir}/{i}_mul_o_u.tif'
        newPan_o = f'{out_dir}/{i}_pan_o.tif'

        rawPan = (gdal.Open(f'{in_dir}/{i}-PAN.tif') or gdal.Open(f'{in_dir}/{i}-PAN.TIF')).ReadAsArray()
        rawMul = (gdal.Open(f'{in_dir}/{i}-MUL.tif') or gdal.Open(f'{in_dir}/{i}-MUL.TIF')).ReadAsArray()
        print("rawMul:", rawMul.shape, " rawPan:", rawPan.shape)

        rawMul = rawMul.transpose(1, 2, 0)  # (h, w, c)

        h, w = rawMul.shape[:2]
        h -= 10     # drop some edge pixels
        w -= 10
        h = h // 4 * 4  # as a multiple of 4
        w = w // 4 * 4
        rawMul = rawMul[:h, :w, :]
        rawPan = rawPan[:h * 4, :w * 4]

        imgMul = rawMul  # h * w * 4
        imgLR = down_sample(imgMul, mode=args.mode)  # (h / 4) * (w / 4) * 4
        imgLR_u = up_sample(imgLR, mode=args.mode)  # h * w * 4
        imgPan = rawPan
        imgPan = down_sample(imgPan, mode=args.mode)  # h * w

        if i == 0:  # test scene
            if args.reduce_raw:     # crop 1/4 middle part
                imgMul_o = rawMul[h // 4 * 2:h // 4 * 3, w // 4 * 2:w // 4 * 3, :]  # (h / 4) * (w / 4) * 4
                imgMul_o_u = up_sample(imgMul_o)  # h * w * 4
                imgPan_o = rawPan[h * 2:h * 3, w * 2:w * 3]  # h * w
            else:
                imgMul_o = rawMul
                imgMul_o_u = up_sample(imgMul_o)
                imgPan_o = rawPan
        else:   # train scene
            imgMul_o = rawMul  # (h / 4) * (w / 4) * 4
            imgMul_o_u = up_sample(imgMul_o)  # h * w * 4
            imgPan_o = rawPan  # h * w

        imgMul = imgMul.transpose(2, 0, 1)
        imgLR = imgLR.transpose(2, 0, 1)
        imgLR_u = imgLR_u.transpose(2, 0, 1)
        imgMul_o = imgMul_o.transpose(2, 0, 1)
        imgMul_o_u = imgMul_o_u.transpose(2, 0, 1)

        if i == 0:  # test scene, as a multiple of 50 for clip to patches
            a = h // 4 if args.reduce_raw else h
            b = w // 4 if args.reduce_raw else w

            a = a // 50 * 50
            b = b // 50 * 50
            save_image(newMul, imgMul[:, :a * 4, :b * 4])
            save_image(newLR, imgLR[:, :a, :b])
            save_image(newLR_u, imgLR_u[:, :a * 4, :b * 4])
            save_image(newPan, imgPan[:a * 4, :b * 4])

            save_image(newMul_o, imgMul_o[:, :a, :b])
            save_image(newMul_o_u, imgMul_o_u[:, :a * 4, :b * 4])
            save_image(newPan_o, imgPan_o[:a * 4, :b * 4])
        else:
            save_image(newMul, imgMul)
            save_image(newLR, imgLR)
            save_image(newLR_u, imgLR_u)
            save_image(newPan, imgPan)

            save_image(newMul_o, imgMul_o)
            save_image(newMul_o_u, imgMul_o_u)
            save_image(newPan_o, imgPan_o)

        print(f'done {i}')

    print('finish')
