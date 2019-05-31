import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
import math

from loader import get_cifar10
from models import get_model
from trainer import Trainer

from config import cfg, dotdict


def _step_lr(epoch):
    v = 0.0
    for max_e, lr_v in cfg.train.steplr:
        v = lr_v
        if epoch <= max_e:
            break
    return v


def get_lr_func():
    return _step_lr


def adjust_learning_rate(epoch, pack):
    if pack.optimizer is None:
        if cfg.train.optim == 'sgd' or cfg.train.optim is None:
            pack.optimizer = optim.SGD(
                pack.net.parameters(),
                lr=1,
                momentum=cfg.train.momentum,
                weight_decay=cfg.train.weight_decay,
                nesterov=cfg.train.nesterov
            )
        else:
            print('WRONG OPTIM SETTING!')
            assert False
        pack.lr_scheduler = optim.lr_scheduler.LambdaLR(
            pack.optimizer, get_lr_func())
        if cfg.base.fp16 and cfg.base.cuda:
            from apex.fp16_utils import FP16_Optimizer
            pack.optimizer = FP16_Optimizer(
                pack.optimizer, dynamic_loss_scale=True)

    pack.lr_scheduler.step(epoch)
    return pack.lr_scheduler.get_lr()


def recover_pack():
    train_loader, test_loader = get_cifar10()

    pack = dotdict({
        'net': get_model(),
        'train_loader': train_loader,
        'test_loader': test_loader,
        'trainer': Trainer(),
        'criterion': nn.CrossEntropyLoss(),
        'optimizer': None,
        'lr_scheduler': None
    })

    adjust_learning_rate(cfg.base.epoch, pack)
    return pack


def set_seeds():
    torch.manual_seed(cfg.base.seed)
    if cfg.base.cuda:
        torch.cuda.manual_seed_all(cfg.base.seed)
        torch.backends.cudnn.deterministic = True
    np.random.seed(cfg.base.seed)
    random.seed(cfg.base.seed)


def main():
    set_seeds()
    pack = recover_pack()

    for epoch in range(cfg.base.epoch + 1, cfg.train.max_epoch + 1):
        lr = adjust_learning_rate(epoch, pack)
        info = pack.trainer.train(pack)
        info.update(pack.trainer.test(pack))
        info.update({'LR': lr})
        print(epoch, info)


if __name__ == '__main__':
    main()
