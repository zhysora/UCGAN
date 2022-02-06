import torch
import torch.nn as nn
from logging import Logger
from mmcv import Config

from .base_model import Base_model
from .builder import MODELS
from .common.utils import up_sample, down_sample, get_hp, get_lp, channel_pooling
from .common.modules import ResBlock, ResChAttnBlock, Patch_Discriminator, conv3x3, build_norm_layer


class EmbNet(nn.Module):
    def __init__(self, logger: Logger, ms_chans, n_blocks=1, n_feats=32, norm_type='BN',
                 basic_block=ResBlock):
        super(EmbNet, self).__init__()

        self.net = []
        self.net.append(conv3x3(ms_chans, n_feats))
        # 32 x 256 x 256
        if norm_type is not None:
            self.net.append(build_norm_layer(logger, n_feats, norm_type))
        for i in range(n_blocks):
            self.net.append(nn.ReLU(True))
            self.net.append(basic_block(logger, n_feats, norm_type))
        # 32 x 256 x 256
        self.net.append(nn.ReLU(True))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        # 32 x 256 x 256
        return self.net(x)


class FusionNet(nn.Module):
    def __init__(self, logger: Logger, n_blocks=3, n_feats=32, norm_type='BN',
                 basic_block=ResBlock):
        super(FusionNet, self).__init__()

        self.net = []
        self.net.append(conv3x3(n_feats*2, n_feats))
        if norm_type is not None:
            self.net.append(build_norm_layer(logger, n_feats, norm_type))
        self.net.append(nn.ReLU(True))
        self.net.append(basic_block(logger, n_feats, norm_type))
        for i in range(n_blocks - 1):
            self.net.append(nn.ReLU(True))
            self.net.append(basic_block(logger, n_feats, norm_type))
        # 32 x 256 x 256
        self.net.append(nn.ReLU(True))
        self.net = nn.Sequential(*self.net)

    def forward(self, lr_feat, pan_feat):
        net = self.net(torch.cat((lr_feat, pan_feat), dim=1))
        return net


class RestoreNet(nn.Module):
    def __init__(self, logger: Logger, ms_chans, n_blocks=1, n_feats=32, norm_type='BN', basic_block=ResBlock):
        super(RestoreNet, self).__init__()

        self.net = []
        # 32 x 256 x 256
        for i in range(n_blocks):
            self.net.append(basic_block(logger, n_feats, norm_type))
            self.net.append(nn.ReLU(True))
        self.net.append(conv3x3(n_feats, ms_chans))
        # 4 x 256 x 256
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        # 4 x 256 x 256
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, cfg, logger, ms_chans, hp_filter=False, num_blocks=(1, 3, 1), n_feats=32,
                 norm_type='BN', block_type='RCA'):
        r"""
        Args:
            cfg (Config): full config
            logger (Logger): logger
            ms_chans (int): bands of multi-spectral image
            hp_filter (bool): whether use high-pass filter or not
            num_blocks (tuple[int]): number of blocks in EmbNet, FusionNet, RestoreNet
            n_feats (int): feature channels in mid-layer
            norm_type (str | None): type of normalization layer or not use
            block_type (str): type of block, choice of ["RCA", "Res"]
        """
        super(Generator, self).__init__()
        self.hp_filter = hp_filter
        self.cfg = cfg
        if block_type == 'RCA':
            self.emb_net = EmbNet(logger, ms_chans, num_blocks[0], n_feats, norm_type, ResChAttnBlock)
            self.fusion_emb = FusionNet(logger, num_blocks[1], n_feats, norm_type, ResChAttnBlock)
            self.de_emb_net = RestoreNet(logger, ms_chans, num_blocks[2], n_feats, norm_type, ResChAttnBlock)
        elif block_type == 'Res':
            self.emb_net = EmbNet(logger, ms_chans, num_blocks[0], n_feats, norm_type, ResBlock)
            self.fusion_emb = FusionNet(logger, num_blocks[1], n_feats, norm_type, ResBlock)
            self.de_emb_net = RestoreNet(logger, ms_chans, num_blocks[2], n_feats, norm_type, ResBlock)
        else:
            raise SystemExit(f'no such kind of generator: \"{block_type}\"')

    def forward(self, pan, lr_u, lr=None):
        if self.hp_filter:
            pan_hp = get_hp(pan)
            lr_u_hp = get_hp(lr_u)

            pan_feat = self.emb_net(pan_hp)
            lr_u_feat = self.emb_net(lr_u_hp)
            fusion_feat = self.fusion_emb(lr_u_feat, pan_feat)
            fusion = self.de_emb_net(fusion_feat) + lr_u
        else:
            pan_feat = self.emb_net(pan)
            lr_u_feat = self.emb_net(lr_u)
            fusion_feat = self.fusion_emb(lr_u_feat, pan_feat)
            fusion = self.de_emb_net(fusion_feat)

        return fusion, fusion_feat, pan_feat, lr_u_feat


@MODELS.register_module()
class UCGAN(Base_model):
    def __init__(self, cfg, logger, train_data_loader, test_data_loader0, test_data_loader1):
        super(UCGAN, self).__init__(cfg, logger, train_data_loader, test_data_loader0, test_data_loader1)
        ms_chans = cfg.get('ms_chans', 4)
        model_cfg = cfg.get('model_cfg', dict())
        G_cfg = model_cfg.get('core_module', dict())
        D_cfg = model_cfg.get('Discriminator', dict())

        self.add_module('core_module', Generator(cfg=cfg, logger=logger, ms_chans=ms_chans, **G_cfg))
        self.add_module('Discriminator', Patch_Discriminator(logger=logger, in_channels=ms_chans*2+1, **D_cfg))
        self.to_pan_mode = model_cfg.get('to_pan_mode', 'max')

    def get_model_output(self, input_batch):
        input_pan = input_batch['input_pan']
        input_lr = input_batch['input_lr']
        input_lr_u = up_sample(input_lr)
        input_pan_rept = input_pan.expand(input_lr_u.shape)
        output, _, _, _ = self.module_dict['core_module'](input_pan_rept, input_lr_u)
        return output

    def train_iter(self, iter_id, input_batch, log_freq=10):
        G = self.module_dict['core_module']
        D = self.module_dict['Discriminator']
        G_optim = self.optim_dict['core_module']
        D_optim = self.optim_dict['Discriminator']

        input_pan = input_batch['input_pan']
        input_pan_l = input_batch['input_pan_l']
        input_lr = input_batch['input_lr']
        input_lr_u = up_sample(input_lr)
        input_pan_rept = input_pan.expand(input_lr_u.shape)

        output, _, _, _ = G(input_pan_rept, input_lr_u)
        fake_lr_u = up_sample(down_sample(output))
        fake_pan = channel_pooling(output, mode=self.to_pan_mode)
        output_cyc, _, _, _ = G(input_pan_rept, fake_lr_u)

        loss_g = 0
        loss_res = dict()
        loss_cfg = self.cfg.get('loss_cfg', {})

        if 'QNR_loss' in self.loss_module:
            QNR_loss = self.loss_module['QNR_loss'](pan=input_pan, ms=input_lr, pan_l=input_pan_l, out=output)
            loss_g = loss_g + QNR_loss * loss_cfg['QNR_loss'].w
            loss_res['QNR_loss'] = QNR_loss.item()
        if 'cyc_rec_loss' in self.loss_module:
            cyc_rec_loss = self.loss_module['cyc_rec_loss'](out=output_cyc, gt=output)
            loss_g = loss_g + cyc_rec_loss * loss_cfg['cyc_rec_loss'].w
            loss_res['cyc_rec_loss'] = cyc_rec_loss.item()
        if 'spectral_rec_loss' in self.loss_module:
            spectral_rec_loss = self.loss_module['spectral_rec_loss'](
                out=get_lp(input_lr_u), gt=get_lp(output)
            )
            loss_g = loss_g + spectral_rec_loss * loss_cfg['spectral_rec_loss'].w
            loss_res['spectral_rec_loss'] = spectral_rec_loss.item()
        if 'spatial_rec_loss' in self.loss_module:
            spatial_rec_loss = self.loss_module['spatial_rec_loss'](
                out=get_hp(fake_pan), gt=get_hp(input_pan)
            )
            loss_g = loss_g + spatial_rec_loss * loss_cfg['spatial_rec_loss'].w
            loss_res['spatial_rec_loss'] = spatial_rec_loss.item()
        if 'adv_loss' in self.loss_module:
            adv_loss, loss_d = self.loss_module['adv_loss'](
                fake=torch.cat((input_pan, fake_lr_u, output_cyc), dim=1),
                real=torch.cat((input_pan, input_lr_u, output), dim=1),
                D=D, D_optim=D_optim
            )
            loss_g = loss_g + adv_loss * loss_cfg['adv_loss'].w
            loss_res['adv_loss'] = (adv_loss.item(), loss_d.item())
        loss_res['full_loss'] = loss_g.item()

        G_optim.zero_grad()
        loss_g.backward()
        G_optim.step()

        self.print_train_log(iter_id, loss_res, log_freq)
