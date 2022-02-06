import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def weight_init(m):
    r"""
        weight init based on xavier normalization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.zero_()


def torch2np(data):
    r""" transfer image from torch.Tensor to np.ndarray

    Args:
        data (torch.Tensor): image shape like [N, C, H, W]
    Returns:
        np.ndarray: image shape like [N, H, W, C] or [N, H, W]
    """
    if data.shape[1] is 1:
        return data.squeeze(1).cpu().detach().numpy()
    else:
        return data.cpu().detach().numpy().transpose(0, 2, 3, 1)


def smart_time(second):
    r""" transfer second into day-hour-min-sec format

    Args:
        second (float | int): time in second
    Returns:
        str: time in day-hour-min-sec format
    """
    second = int(second)
    day = second // (24 * 60 * 60)
    second = second % (24 * 60 * 60)
    hour = second // (60 * 60)
    second = second % (60 * 60)
    minute = second // 60
    second = second % 60

    time_str = ''
    if day > 0:
        time_str += f'{day}d'
    if hour > 0:
        time_str += f'{hour}h'
    if minute > 0:
        time_str += f'{minute}m'
    time_str += f'{second}s'

    return time_str


def get_lp(data):
    r""" get the low-frequency of input images,
    calculate the avg_filter of the input as low-frequency

    Args:
        data (torch.Tensor): image matrix, shape of [N, C, H, W]
    Returns:
        torch.Tensor: low-frequency part of input, shape of [N, C, H, W]
    """
    rs = F.avg_pool2d(data, kernel_size=5, stride=1, padding=2)
    return rs


def get_hp(data):
    r""" get the high-frequency of input images,
    first calculate the avg_filter of the input as low-frequency,
    subtract the low-frequency to get the high-frequency

    Args:
        data (torch.Tensor): image matrix, shape of [N, C, H, W]
    Returns:
        torch.Tensor: high-frequency part of input, shape of [N, C, H, W]
    """
    rs = F.avg_pool2d(data, kernel_size=5, stride=1, padding=2)
    rs = data - rs
    return rs


def set_batch_cuda(sample_batched):
    r""" move the input batch to cuda

    Args:
        sample_batched (dict[str, torch.Tensor | str]): input batch
    Returns:
        dict[str, torch.Tensor | str]: input batch in cuda
    """
    for key in sample_batched:
        if key == 'image_id':
            continue
        sample_batched[key] = sample_batched[key].cuda()
    return sample_batched


def up_sample(imgs, r=4, mode='bicubic'):
    r""" up-sample the images

    Args:
        imgs (torch.Tensor): input images, shape of [N, C, H, W]
        r (int): scale ratio, Default: 4
        mode (str): interpolate mode, Default: 'bicubic'
    Returns:
        torch.Tensor: images after un-sampling, shape of [N, C, H*r, W*r]
    """
    _, __, h, w = imgs.shape
    return F.interpolate(imgs, size=[h * r, w * r], mode=mode, align_corners=True)


def down_sample(imgs, r=4, mode='bicubic'):
    r""" down-sample the images

    Args:
        imgs (torch.Tensor): input images, shape of [N, C, H, W]
        r (int): scale ratio, Default: 4
        mode (str): interpolate mode, Default: 'bicubic'
    Returns:
        torch.Tensor: images after down-sampling, shape of [N, C, H//r, W//r]
    """
    _, __, h, w = imgs.shape
    return F.interpolate(imgs, size=[h // r, w // r], mode=mode, align_corners=True)


def channel_pooling(imgs, mode='avg'):
    r""" average or maximum pooling at channel-dim

    Args:
        imgs (torch.Tensor): input images, shape of [N, C, H, W]
        mode (str): 'avg' or 'max', Default: 'avg'
    Returns:
        torch.Tensor: images after pooling, shape of [N, 1, H, W]
    """
    if mode == 'avg':
        return torch.mean(imgs, dim=1, keepdim=True)
    elif mode == 'max':
        return torch.max(imgs, dim=1, keepdim=True)[0]
    else:
        raise SystemExit(f'no such pooling mode \"{mode}\"')


def calc_img_grad(imgs):
    r""" calculate the gradient of images by row and column

    Args:
        imgs (torch.Tensor): input images, shape of [N, C, H, W]
    Returns:
        torch.Tensor: gradient of images, shape of [N, C, H-1, W-1]
    """
    ret = (torch.abs(imgs[:, :, :-1, :-1] - imgs[:, :, 1:, :-1]) +
           torch.abs(imgs[:, :, :-1, :-1] - imgs[:, :, :-1, 1:])) / 2.
    return ret
