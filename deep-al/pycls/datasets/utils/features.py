import numpy as np
import torch
DATASET_FEATURES_DICT = {
    'train':
        {
            'CIFAR10':'../../results/cifar-10/pretext/features_seed{seed}.npy',
            'CIFAR100':'../../results/cifar-100/pretext/features_seed{seed}.npy',
            'TINYIMAGENET': '../../results/tiny-imagenet/pretext/features_seed{seed}.npy',
            'ISIC':'../../results/isic/pretext/features_seed{seed}.npy',
        },
    'test':
        {
            'CIFAR10': '../../results/cifar-10/pretext/test_features_seed{seed}.npy',
            'CIFAR100': '../../results/cifar-100/pretext/test_features_seed{seed}.npy',
            'TINYIMAGENET': '../../results/tiny-imagenet/pretext/test_features_seed{seed}.npy',
            'ISIC':'../../results/isic/pretext/test_features_seed{seed}.npy',
        }
}

def load_features(ds_name, seed=2026, train=True, normalized=True):
    " load pretrained features for a dataset "
    split = "train" if train else "test"
    fname = DATASET_FEATURES_DICT[split][ds_name].format(seed=seed)
    if fname.endswith('.npy'):
        features = np.load(fname)
    elif fname.endswith('.pth'):
        features = torch.load(fname)
    else:
        raise Exception("Unsupported filetype")
    if normalized:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features