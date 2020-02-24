import math

import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['vgg19']




class VGG(nn.Module):
    def __init__(self, in_channels, num_classes, depth=19, template='base', proportion=None, version='small'):
        super(VGG, self).__init__()

        # adjust model filter definition
        cfg = self._set_filters(template, depth)
        if proportion is not None: cfg = self._set_proportional_filters(cfg, proportion)
        print('Final Filters:', cfg)

        self.feature = self.make_layers(cfg, True, in_channels)

        if version == 'small':
            self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
            self.classifier = nn.Linear(cfg[-2]*2*2, num_classes)
        if version == 'full':
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(cfg[-2] * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )




    def make_layers(self, cfg, batch_norm=False, in_channels = 3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



    def _set_filters(self, template, depth=19):

        base_cfg = {
            11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        uniform_cfg = {
            11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19 : [344, 344, 'M', 344, 344, 'M', 344, 344, 344, 344, 'M', 344, 344, 344, 344, 'M', 344, 344, 344, 344, 'M'],
        }
        reverse_base_cfg = {
            11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19 : [512, 512, 'M', 512, 512, 'M', 512, 512, 512, 512, 'M', 256, 256, 256, 256, 'M', 128, 128, 64, 64, 'M']
        }
        quadratic_cfg = {
            11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19 : [512, 445, 'M', 387, 339, 'M', 301, 272, 253, 243, 'M', 243, 253, 272, 301, 'M', 339, 387, 445, 512, 'M'],
        }
        negative_quadratic_cfg = {
            11 : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            13 : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            16 : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            19 : [64, 176, 'M', 272, 352, 'M', 416, 464, 496, 512, 'M', 512, 496, 464, 416, 'M', 352, 272, 176, 64, 'M'],
        }
        if template == 'uniform': return uniform_cfg[depth]
        if template == 'base': return base_cfg[depth]
        if template == 'reverse-base': return reverse_base_cfg[depth]
        if template == 'quadratic': return quadratic_cfg[depth]
        if template == 'negative-quadratic': return negative_quadratic_cfg[depth]


    def _set_proportional_filters(self, filter_list, proportion):
        temp_list = []
        for item in filter_list:
            if isinstance(item, int):
                new_value = int(item*proportion)
                if new_value<1: new_value = 1
                temp_list.append(new_value)
            else:
                temp_list.append(item)
        return temp_list



def vgg19(args):
    if args.dataset in ['imagenet2012']: #['imagenet2012', 'tiny-imagenet', 'user-imagenet']:
        version = 'full'
    else:
        version = 'small'

    vgg_model = VGG(in_channels=args.input_channels, num_classes=args.number_classes,
                    depth=19, template=args.template, proportion=args.width_multiplier, version=version)
    return vgg_model

if __name__ == '__main__':
    args={'in_channels':3, 'num_classes':10, 'depth':19}
    net = vgg19()
    x = Variable(torch.FloatTensor(16, 3, 40, 40))
    y = net(x)
    print(y.data.shape)
