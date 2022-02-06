import argparse
import torch
from torch.utils.data import DataLoader
import random
import numpy as np
import mmcv
from mmcv import Config
from mmcv.utils import get_logger
from logging import Logger
import traceback

from datasets.builder import build_dataset
from models.builder import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='pan-sharpening implementation')
    parser.add_argument('-c', '--config', required=True, help='config file path')
    return parser.parse_args()


def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(cfg, logger):
    # type: (mmcv.Config, Logger) -> None
    # Setting Random Seed
    if 'seed' in cfg:
        logger.info('===> Setting Random Seed')
        set_random_seed(cfg.seed, True)

    # Loading Datasets
    logger.info('===> Loading Datasets')
    if 'train_set_cfg' in cfg:
        train_set_cfg = cfg.train_set_cfg.copy()
        train_set_cfg['dataset'] = build_dataset(cfg.train_set_cfg['dataset'])
        train_data_loader = DataLoader(**train_set_cfg)
    else:
        train_data_loader = None

    # test on full-resolution
    test_set0_cfg = cfg.test_set0_cfg.copy()
    test_set0_cfg['dataset'] = build_dataset(cfg.test_set0_cfg['dataset'])
    test_data_loader0 = DataLoader(**test_set0_cfg)
    # test on reduced-resolution
    test_set1_cfg = cfg.test_set1_cfg.copy()
    test_set1_cfg['dataset'] = build_dataset(cfg.test_set1_cfg['dataset'])
    test_data_loader1 = DataLoader(**test_set1_cfg)

    # Building Model
    logger.info('===> Building Model')
    runner = build_model(cfg.model_type, cfg, logger, train_data_loader, test_data_loader0, test_data_loader1)

    # Setting GPU
    if 'cuda' in cfg and cfg.cuda:
        logger.info("===> Setting GPU")
        runner.set_cuda()

    # Weight Initialization
    if 'checkpoint' not in cfg:
        logger.info("===> Weight Initializing")
        runner.init()

    #  Resume from a Checkpoint (Optionally)
    if 'checkpoint' in cfg:
        logger.info("===> Loading Checkpoint")
        runner.load_checkpoint(cfg.checkpoint)

    # Copy Weights from a Checkpoint (Optionally)
    if 'pretrained' in cfg:
        logger.info("===> Loading Pretrained")
        runner.load_pretrained(cfg.pretrained)

    # Setting Optimizer
    logger.info("===> Setting Optimizer")
    runner.set_optim()

    # Setting Scheduler for learning_rate Decay
    logger.info("===> Setting Scheduler")
    runner.set_sched()

    # Print Params Count
    logger.info("===> Params Count")
    runner.print_total_params()
    runner.print_total_trainable_params()

    if ('only_test' not in cfg) or (not cfg.only_test):
        # Training
        logger.info("===> Training Start")
        runner.train()

        # Saving
        logger.info("===> Final Saving Weights")
        runner.save(iter_id=cfg.max_iter)

    # Testing
    logger.info("===> Final Testing")
    runner.test(iter_id=cfg.max_iter, save=True, ref=True)  # low-resolution testing
    runner.test(iter_id=cfg.max_iter, save=True, ref=False)  # full-resolution testing

    logger.info("===> Finish !!!")


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    mmcv.mkdir_or_exist(cfg.log_dir)
    logger = get_logger('mmFusion', cfg.log_file, cfg.log_level)
    logger.info(f'Config:\n{cfg.pretty_text}')

    try:
        main(cfg, logger)
    except:
        logger.error(str(traceback.format_exc()))
