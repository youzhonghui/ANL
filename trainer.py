from time import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from tqdm import tqdm
import numpy as np
from config import cfg

from models.anl import AdvNoise


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Trainer():
    def __init__(self):
        self.use_cuda = cfg.base.cuda

    def test(self, pack, topk=(1,)):
        pack.net.eval()
        loss_acc = 0.0
        hub = [[] for i in range(len(topk))]

        for data, target in pack.test_loader:
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()

            with torch.no_grad():
                output = pack.net(data)
                loss_acc += pack.criterion(output, target).data.item()
                acc = accuracy(output, target, topk)
                for acc_idx, score in enumerate(acc):
                    hub[acc_idx].append(score[0].item())

        loss_acc /= len(pack.test_loader)
        info = {
            'test_loss': loss_acc
        }

        for acc_idx, k in enumerate(topk):
            info['acc@%d' % k] = np.mean(hub[acc_idx])

        return info

    def train(self, pack):
        pack.net.train()
        loss_acc = 0.0
        begin = time()

        pack.optimizer.zero_grad()
        with tqdm(total=len(pack.train_loader), ncols=100) as pbar:
            for _, (data, label) in enumerate(pack.train_loader):
                if self.use_cuda:
                    data, label = data.cuda(), label.cuda()
                data = Variable(data, requires_grad=False)
                label = Variable(label)

                pack.net.apply(lambda m: type(m) == AdvNoise and m.set_clean())
                pack.optimizer.zero_grad()
                est_out = pack.net(data)
                loss = pack.criterion(est_out, label)
                loss.backward()

                pack.net.apply(lambda m: type(m) == AdvNoise and m.set_stay())
                pack.optimizer.zero_grad()
                out = pack.net(data)
                loss = pack.criterion(out, label)

                loss.backward()
                pack.optimizer.step()

                loss_acc += loss.item()
                pbar.update(1)

        info = {
            'train_loss': loss_acc / len(pack.train_loader),
            'epoch_time': time() - begin
        }
        return info
