from mmcv.utils import Registry

from .base_model import Base_model

# create a registry
MODELS = Registry('model')


# create a build function
def build_model(model_type: str, *args, **kwargs) -> Base_model:
    if model_type not in MODELS:
        raise KeyError(f'Unrecognized task type {model_type}')
    else:
        model_cls = MODELS.get(model_type)

    model = model_cls(*args, **kwargs)
    return model
