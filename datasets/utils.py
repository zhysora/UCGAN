import gdal
import osr
import torch
from torch.nn.functional import interpolate
import numpy as np
from numpy.random import random


def _is_pan_image(filename):
    return filename.endswith("pan.tif")


def get_image_id(filename):
    return filename.split('_')[0]


def load_image(path):
    """ Load .TIF image to np.array

    Args:
        path (str): path of TIF image
    Returns:
        np.array: value matrix in [C, H, W] or [H, W]
    """
    img = np.array(gdal.Open(path).ReadAsArray(), dtype=np.double)
    return img


def save_image(path, array):
    """ Save np.array as .TIF image

    Args:
        path (str): path to save as TIF image
        np.array: shape like [C, H, W] or [H, W]
    """
    # Meaningless Default Value
    raster_origin = (-123.25745, 45.43013)
    pixel_width = 2.4
    pixel_height = 2.4

    if array.ndim == 3:
        chans = array.shape[0]
        cols = array.shape[2]
        rows = array.shape[1]
        origin_x = raster_origin[0]
        origin_y = raster_origin[1]

        driver = gdal.GetDriverByName('GTiff')

        out_raster = driver.Create(path, cols, rows, chans, gdal.GDT_UInt16)
        # print(path, cols, rows, chans, out_raster)
        out_raster.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0, pixel_height))
        for i in range(1, chans + 1):
            out_band = out_raster.GetRasterBand(i)
            out_band.WriteArray(array[i - 1, :, :])
        out_raster_srs = osr.SpatialReference()
        out_raster_srs.ImportFromEPSG(4326)
        out_raster.SetProjection(out_raster_srs.ExportToWkt())
        out_band.FlushCache()
    elif array.ndim == 2:
        cols = array.shape[1]
        rows = array.shape[0]
        origin_x = raster_origin[0]
        origin_y = raster_origin[1]

        driver = gdal.GetDriverByName('GTiff')

        out_raster = driver.Create(path, cols, rows, 1, gdal.GDT_UInt16)
        out_raster.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0, pixel_height))

        out_band = out_raster.GetRasterBand(1)
        out_band.WriteArray(array[:, :])


def data_augmentation(img_dict, aug_dict=None):
    """ Data augmentation for training set

    Args:
        img_dict (dict[str, torch.Tensor]): images in torch.Tensor, shape like [N, C, H, W]
        aug_dict (dict[str, float]): probability of each augmentation,
            example: {'ud_flip' : 0.5, 'lr_flip' : 0.5, 'r4_crop' : 0.3, 'r2_crop': 0.3}
    Returns:
        dict[str, torch.Tensor]: images after augmentation
    """

    def flip(x, dim):
        """ flip the image at axis=dim

        Args:
            x (torch.Tensor): image in torch.Tensor, shape like [N, C, H, W]
            dim (int): 2 or 3, up-down or left-right flip
        Returns:
            torch.Tensor: image after flipping, shape like [N, C, H, W]
        """
        index_list = [i for i in range(x.size(dim)-1, -1, -1)]
        return x[:, :, index_list, :] if dim is 2 else x[:, :, :, index_list]

    def crop_resize(imgs, crop_st, n=4):
        """ crop part of the image and up-sample to the same size

        Args:
            imgs (torch.Tensor): images in torch.Tensor, shape like [N, C, H, W]
            crop_st (Tuple[int, int]): start point of cropping at [H, W]
            n (int): zoom ratio (n - 1) / n
        Returns:
            torch.Tensor: images after the operation, shape like [N, C, H, W]
        """
        _, __, h, w = imgs.shape
        imgs = imgs[:, :, crop_st[0]:h//n*(n-1)+crop_st[0], crop_st[1]:w//n*(n-1)+crop_st[1]]
        imgs = interpolate(imgs, size=[h, w], mode='bicubic', align_corners=True)
        return imgs

    if type(aug_dict) is type(None):
        return img_dict

    need_aug = False
    for aug in aug_dict:    # transfer the probability to Ture/False
        aug_dict[aug] = (random() < aug_dict[aug])
        need_aug = (need_aug or aug_dict[aug])

    if not need_aug:
        return img_dict

    if 'r4_crop' in aug_dict and aug_dict['r4_crop']:
        d1 = int(img_dict['input_lr'].size(2) // 4 * random())
        d2 = int(img_dict['input_lr'].size(3) // 4 * random())
    if 'r2_crop' in aug_dict and aug_dict['r2_crop']:
        d3 = int(img_dict['input_lr'].size(2) // 2 * random())
        d4 = int(img_dict['input_lr'].size(3) // 2 * random())

    ret = dict(image_id=img_dict['image_id'])
    for img_name in img_dict:
        if img_name == 'image_id':
            continue
        imgs = img_dict[img_name]
        if 'ud_flip' in aug_dict and aug_dict['ud_flip']:
            ret[img_name] = flip(imgs, 2)
        if 'lr_flip' in aug_dict and aug_dict['lr_flip']:
            ret[img_name] = flip(imgs, 3)
        if 'r4_crop' in aug_dict and aug_dict['r4_crop']:
            ret[img_name] = crop_resize(
                imgs, (d1, d2) if img_name in ['input_lr', 'input_pan_l'] else (d1*4, d2*4), 4
            )
        if 'r2_crop' in aug_dict and aug_dict['r2_crop']:
            ret[img_name] = crop_resize(
                imgs, (d3, d4) if img_name in ['input_lr', 'input_pan_l'] else (d3*4, d4*4), 2
            )

    return ret


def data_normalize(img_dict, bit_depth):
    """ Normalize the data to [0, 1)

    Args:
        img_dict (dict[str, torch.Tensor]): images in torch.Tensor
        bit_depth (int): original data range in n-bit
    Returns:
        dict[str, torch.Tensor]: images after normalization
    """
    max_value = 2 ** bit_depth - .5
    ret = dict()
    for img_name in img_dict:
        if img_name == 'image_id':
            ret[img_name] = img_dict[img_name]
            continue
        imgs = img_dict[img_name]
        ret[img_name] = imgs / max_value
    return ret


def data_denormalize(img, bit_depth):
    """ Denormalize the data to [0, n-bit)

    Args:
        img (torch.Tensor | np.ndarray): images in torch.Tensor
        bit_depth (int): original data range in n-bit
    Returns:
        dict[str, torch.Tensor]: image after denormalize
    """
    max_value = 2 ** bit_depth - .5
    ret = img * max_value
    return ret
