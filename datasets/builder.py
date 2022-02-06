from mmcv.utils import Registry
from mmcv import Config
import torch.utils.data as data

# create a registry
DATASETS = Registry('dataset')


# create a build function
def build_dataset(cfg: Config, *args, **kwargs) -> data.Dataset:
    cfg_ = cfg.copy()
    dataset_type = cfg_.pop('type')
    if dataset_type not in DATASETS:
        raise KeyError(f'Unrecognized task type {dataset_type}')
    else:
        dataset_cls = DATASETS.get(dataset_type)

    dataset = dataset_cls(*args, **kwargs, **cfg_)
    return dataset
