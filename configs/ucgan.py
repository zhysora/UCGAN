# ---> GENERAL CONFIG <---
name = 'ucgan_GF-2'
description = 'test ucgan on PSData3/GF-2 dataset'

model_type = 'UCGAN'
work_dir = f'data/PSData3/model_out/{name}'
log_dir = f'logs/{model_type.lower()}'
log_file = f'{log_dir}/{name}.log'
log_level = 'INFO'

only_test = True
checkpoint = f'data/PSData3/model_out/{name}/train_out/pretrained.pth'

# ---> DATASET CONFIG <---
ms_chans = 4
bit_depth = 10
train_set_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=['data/PSData3/Dataset/GF-2/train_full_res'],
        bit_depth=10),
    num_workers=8,
    batch_size=16,
    shuffle=True)
test_set0_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=['data/PSData3/Dataset/GF-2/test_full_res'],
        bit_depth=10),
    num_workers=4,
    batch_size=4,
    shuffle=False)
test_set1_cfg = dict(
    dataset=dict(
        type='PSDataset',
        image_dirs=['data/PSData3/Dataset/GF-2/test_low_res'],
        bit_depth=10),
    num_workers=4,
    batch_size=4,
    shuffle=False)
cuda = True
max_iter = 30000
save_freq = 1500
test_freq = 30000
eval_freq = 1500
norm_input = False

# ---> SPECIFIC CONFIG <---
optim_cfg = dict(
    core_module=dict(type='AdamW', lr=0.0001),
    Discriminator=dict(type='AdamW', lr=5e-05))
sched_cfg = dict(step_size=15000, gamma=0.9)
loss_cfg = dict(
    QNR_loss=dict(w=1.0),
    cyc_rec_loss=dict(type='l1', w=0.001),
    spectral_rec_loss=dict(type='l1', w=0.0005),
    spatial_rec_loss=dict(type='l1', w=0.0005),
    adv_loss=dict(type='LSGAN', soft_label=True, w=0.001))
model_cfg = dict(
    core_module=dict(
        hp_filter=True,
        num_blocks=(1, 3, 1),
        n_feats=32,
        norm_type='IN',
        block_type='RCA'),
    Discriminator=dict(n_feats=32, norm_type='IN'),
    to_pan_mode='max')
