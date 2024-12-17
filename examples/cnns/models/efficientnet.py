import torchvision
from torchvision.models.efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7, efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

def EfficientNetB0(num_classes=1000):
    return efficientnet_b0(num_classes=num_classes)

def EfficientNetB1(num_classes=1000):
    return efficientnet_b1(num_classes=num_classes)

def EfficientNetB2(num_classes=1000):
    return efficientnet_b2(num_classes=num_classes)

def EfficientNetB3(num_classes=1000):
    return efficientnet_b3(num_classes=num_classes)

def EfficientNetB4(num_classes=1000):
    return efficientnet_b4(num_classes=num_classes)

def EfficientNetB5(num_classes=1000):
    return efficientnet_b5(num_classes=num_classes)

def EfficientNetB6(num_classes=1000):
    return efficientnet_b6(num_classes=num_classes)

def EfficientNetB7(num_classes=1000):
    return efficientnet_b7(num_classes=num_classes)

def EfficientNetV2S(num_classes=1000):
    return efficientnet_v2_s(num_classes=num_classes)

def EfficientNetV2M(num_classes=1000):
    return efficientnet_v2_m(num_classes=num_classes)

def EfficientNetV2L(num_classes=1000):
    return efficientnet_v2_l(num_classes=num_classes)
