'''
Author: Unknown
Date: 2024-03-01 10:02:36
LastEditTime: 2024-11-12 00:27:50
LastEditors: Unknown
Description: model entry point
FilePath: /Unknown/models/__init__.py
'''
from .resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .vgg import VGG11, VGG13, VGG16, VGG19
from .densenet import DenseNet121, DenseNet169, DenseNet201, DenseNet161, densenet_cifar
from .googlenet import googlenet
from .efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
from .vit import vit_b_16, vit_b_32, vit_l_16, vit_h_14, vit_tiny, vit_s_16, vit_s_32
from .ZenNet import zennet_size05M, zennet_size1M, zennet_size2M
from .gpt2.get_model import gpt2_model


model_dict = {
    'resnet18': ResNet18, 'resnet34': ResNet34, 'resnet50': ResNet50, 'resnet101': ResNet101, 'resnet152': ResNet152,
    'vgg11': VGG11, 'vgg13': VGG13, 'vgg16': VGG16, 'vgg19': VGG19,
    'densenet121': DenseNet121, 'densenet169': DenseNet169, 'densenet201': DenseNet201, 'densenet161': DenseNet161, 'densenet_cifar': densenet_cifar,
    'googlenet': googlenet,
    'efficientnet_b0': EfficientNetB0, 'efficientnet_b1': EfficientNetB1, 'efficientnet_b2': EfficientNetB2, 'efficientnet_b3': EfficientNetB3, 'efficientnet_b4': EfficientNetB4, 'efficientnet_b5': EfficientNetB5, 'efficientnet_b6': EfficientNetB6, 'efficientnet_b7': EfficientNetB7, 'efficientnet_v2_s': EfficientNetV2S, 'efficientnet_v2_m': EfficientNetV2M, 'efficientnet_v2_l': EfficientNetV2L,
    'vit_b_16': vit_b_16, 'vit_b_32': vit_b_32, 'vit_l_16': vit_l_16, 'vit_h_14': vit_h_14, 'vit_tiny': vit_tiny, 'vit_s_16': vit_s_16, 'vit_s_32': vit_s_32,
    'zennet_size05M': zennet_size05M, 'zennet_size1M': zennet_size1M, 'zennet_size2M': zennet_size2M,
    'gpt2': gpt2_model
}
AVAILABLE_MODELS = list(model_dict.keys())

def get_model(model_name: str, *args, **kwargs):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError('Model name must be one of %s' % AVAILABLE_MODELS)
    return model_dict[model_name](*args, **kwargs)
