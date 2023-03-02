import numpy
import os
import torch

class section_dataset(torch.utils.data.Dataset):
    """
        Data loader for the section-based deconvnet
    """
    def __init__(self, split='train', channel_delta=0, is_transform=True, augmentations=None):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass