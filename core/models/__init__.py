import torchvision.models as models
from core.models.patch_deconvnet import patch_deconvnet, patch_deconvnet_skip
from core.models.section_deconvnet import section_deconvnet, section_deconvnet_skip
from core.models.section_two_stream import section_two_stream

def get_model(name, pretrained, n_channels, n_classes):
    model = _get_model_instance(name)

    if name in ['section_deconvnet','patch_deconvnet']:
        model = model(n_channels=n_channels, n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=pretrained)
        model.init_vgg16_params(vgg16)
    else:
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'section_deconvnet': section_deconvnet,
            'patch_deconvnet':patch_deconvnet,
            'section_deconvnet_skip': section_deconvnet_skip,
            'patch_deconvnet_skip':patch_deconvnet_skip,
            'section_two_stream':section_two_stream
        }[name]
    except:
        print(f'Model {name} not available')
