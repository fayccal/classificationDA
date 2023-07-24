import collections.abc as container_abcs
from itertools import repeat
import torch.nn as nn
import torch
import math
import warnings

# créé un tuple du nombre d'occurence donnée
def _ntuple(n):
    def parse(x):
        # check si x est itérable
        if isinstance(x, container_abcs.Iterable):
            return x
        # si il ne l'est pas créée un tuple de n occurence de x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

#mise a l'échelle du tensor
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma



#Applique la régularization  "drop path" ou "stochastic depth" sur le tensor d'entrée
def drop_path(x, drop_prob: float = 0., training: bool=False, scale_by_keep: bool=True):
    if drop_prob == 0. or not training:
        return x

    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    # génère un tensor random de shape x
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)

    if keep_prob > 0.0 and scale_by_keep:
        #scale le tensor par keep_prob
        random_tensor.div_(keep_prob)

    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        # renvoie une string de la probabilité du drop_path
        return f'drop_prob={round(self.drop_prob,3):0.3f}'

