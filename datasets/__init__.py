from .memcached_dataset_tsn import McDataset_tsn
from .memcached_dataset_img import McDataset_img
from .memcached_dataset_slowfast import McDataset_slowfast


def get_train_dataset(config=None, transform=None):
    if not config:
        raise "config is None!"
    if not transform:
        raise "transforms are None!"
    if config.model.type == "tsn" or config.model.type == "tsm":
        return McDataset_tsn(config.train_root,
                             config.train_source,
                             config.model.num_segments,
                             transform=transform)
    elif config.model.type == "slowfast" or config.model.type == "classify":
        return McDataset_slowfast(config.train_root,
                                  config.train_source,
                                  temporal_transform=None,
                                  sample_duration=config.model.
                                  kwargs.sample_duration,
                                  transform=transform)
    elif config.model.type == "img":
        return McDataset_img(config.train_root,
                             config.train_source,
                             transform=transform)
    else:
        raise "Unexpected type of dataset!"


def get_val_dataset(config=None, transform=None):
    if not config:
        raise "config is None!"
    if not transform:
        raise "transforms are None!"
    if config.model.type == "tsn" or config.model.type == "tsm":
        return McDataset_tsn(config.val_root,
                             config.val_source,
                             config.model.num_segments,
                             random_shift=False,
                             transform=transform)
    elif config.model.type == "slowfast" or config.model.type == "classify":
        return McDataset_slowfast(config.val_root,
                                  config.val_source,
                                  temporal_transform=None,
                                  transform=transform)
    elif config.model.type == "img":
        return McDataset_img(config.val_root,
                             config.val_source,
                             transform=transform)
    else:
        raise "Unexpected type of dataset!"
