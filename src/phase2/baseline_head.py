"""Default benchmark head compatibility, without a legacy FastAI install.

Semantics follow the GPL-3.0 PTB-XL benchmark create_head1d default path.
Only the default head is supported; unsupported options fail explicitly.
"""
import torch
from torch import nn


class FirstMaxPool(nn.Module):
    """Global max with first-index tie gradients, deterministic on CUDA."""
    def forward(self,x):
        return x.max(dim=-1,keepdim=True).values


class ConcatPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.mp=FirstMaxPool()
        self.ap=nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return torch.cat((self.mp(x), self.ap(x)), dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


def create_head1d(nf, nc, lin_ftrs=None, ps=.5, bn_final=False,
                  bn=True, act='relu', concat_pooling=True):
    if lin_ftrs is not None or bn_final or not bn or not concat_pooling or act != 'relu':
        raise ValueError('Only the registered default xResNet head is supported')
    return nn.Sequential(ConcatPool(), Flatten(), nn.BatchNorm1d(2*nf),
                         nn.Dropout(ps), nn.Linear(2*nf, nc))
