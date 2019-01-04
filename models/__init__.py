'''Model Architectures used with PyTorch.'''
from .vgg import vgg11, vgg13, vgg16, vgg19
from .dpn import dpn26, dpn92
from .lenet import le_net
from .senet import se_net18
from .pnasnet import pnas_net_a, pnas_net_b
from .densenet import dense_net121, dense_net161, dense_net169, dense_net201
from .googlenet import google_net
from .shufflenet import shuffle_netg2, shuffle_netg3
from .shufflenetv2 import shuffle_netv2
from .resnet import res_net18, res_net34, res_net50, res_net101, res_net152
from .resnext import res_next29_2x64d, res_next29_32x4d, res_next29_4x64d, res_next29_8x64d
from .preact_resnet import (preact_res_net18, preact_res_net34,
                            preact_res_net50, preact_res_net101, preact_res_net152)
from .mobilenet import mobile_net
from .mobilenetv2 import mobile_net_v2
from.alexnet import alexnet
from.wrn import wrn
