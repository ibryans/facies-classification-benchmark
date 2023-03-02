from core.loader.data_loader_NL import *
from core.loader.data_loader_NS import *
from core.loader.data_loader_NZ import *

def get_loader(arch):
    if 'patch' in arch: 
        return patch_dataset
    elif 'section' in arch:
        return section_dataset
    else:
        raise NotImplementedError()
