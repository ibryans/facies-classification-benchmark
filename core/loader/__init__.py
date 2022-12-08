from core.loader.data_loader import *

def get_loader(arch):
    if 'patch' in arch: 
        return patch_dataset
    elif 'section' in arch:
        return section_dataset
    else:
        raise NotImplementedError()