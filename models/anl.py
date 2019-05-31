import torch
import torch.nn as nn


class AdvNoise(nn.Module):
    def __init__(self, eps):
        super(AdvNoise, self).__init__()

        self.adv = None
        self.eps = eps
        self.input_std = 0.0
        self.dim = None

        self.status = 'clean'

    def extra_repr(self):
        return 'eps=%f' % (self.eps)

    def set_clean(self):
        self.status = 'clean'

    def set_stay(self):
        self.status = 'stay'

    def zero_adv(self):
        if self.adv is not None:
            self.adv.data.zero_()

    def cal_r(self, r):
        r.normal_()
        r = torch.clamp(r * (abs(self.eps) / 4) + self.eps / 2, min(0, self.eps), max(0, self.eps)) * self.input_std
        return r

    def update_adv(self, grad):
        if self.training and self.status == 'clean':
            n = grad.data.view(self.adv.shape[0], -1).max(dim=1)[0].view(*self.dim)
            r = self.cal_r(torch.zeros_like(n))
            self.adv.data.set_(r * grad.data / (1e-6 + n))

    # pylint: disable=W0221,E1101
    def forward(self, x):
        if self.eps == 0.0:
            return x

        if self.adv is None:
            self.adv = torch.zeros_like(x, requires_grad=True)
            self.adv.register_hook(self.update_adv)

            self.dim = [1 if i != 0 else -1 for i in range(0, len(self.adv.shape))]

        if not self.training or x.shape != self.adv.shape:
            return x

        if self.status == 'clean':
            self.zero_adv()

        if self.status == 'stay':
            self.input_std = 0.9 * self.input_std + 0.1 * x.std().data.item()

        return x + self.adv
