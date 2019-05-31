import torch
from config import cfg
from models.anl import AdvNoise


def get_anl_creator(eps):
    def anl_creator():
        return AdvNoise(eps=eps)
    return anl_creator

def get_vgg():
    from models.vgg import VGG
    return VGG('VGG16', get_anl_creator(cfg.anl.eps), 10)

def get_mobilenetv2():
    from models.mobilenetv2 import MobileNetV2
    return MobileNetV2(get_anl_creator(cfg.anl.eps), 10)

def get_wide_resnet():
    from models.wide_resnet import WideResNet
    return WideResNet(get_anl_creator(cfg.anl.eps), 28, 10, 10, 0.0)

def get_model():
    pair = {
        'vgg16': get_vgg,
        'mobileNetV2': get_mobilenetv2,
        'wide_resnet': get_wide_resnet
    }

    model = pair[cfg.model.name]()

    if cfg.base.cuda:
        model = model.cuda()

    if cfg.base.multi_gpus:
        model = torch.nn.DataParallel(model)

    return model
