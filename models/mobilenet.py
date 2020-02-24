#https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py


'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['mobilenet']

# (128,2) means conv planes=128, conv stride=2, by default conv stride=1
defaultcfg = [32, 64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class mobilenet(nn.Module):

    #def __init__(self, num_classes=10):
    def __init__(self, dataset='cifar10', depth=1, init_weights=True, cfg=None, f_version='baseline', f_proportion=None):
        super(mobilenet, self).__init__()

        if cfg is None:
            cfg = defaultcfg

        print('Filterssssss:', cfg)
        if f_version != 'baseline': cfg = self._set_filters(f_version)
        if f_proportion is not None: cfg = self._set_proportional_filters(cfg, f_proportion)
        print('Final Filterssssss:', cfg)
        self.cfg = cfg


        if dataset == 'cifar10':
            num_classes = 10
            in_channels = 3
        elif dataset == 'cifar100':
            num_classes = 100
            in_channels = 3
        elif dataset == 'mnist':
            num_classes = 10
            in_channels = 1
        elif dataset == 'fashionmnist':
            num_classes = 10
            in_channels = 1
        elif dataset == 'arabic-mnist':
            num_classes = 10
            in_channels = 1



        self.conv1 = nn.Conv2d(in_channels, self.cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.cfg[0])
        self.layers = self._make_layers(in_planes=self.cfg[0])
        self.linear = nn.Linear(self.cfg[-1], num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg[1:]:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _set_filters(self, f_version):


        #defaultcfg = [32, 64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
        #fixed_cfg = [467, 467, (467,2), 467, (467,2), 467, (467,2), 467, 467, 467, 467, 467, (467,2), 467]
        #parabola_cfg = [32, 218, (376,2), 505, (605,2), 720, (735,2), 720, 677, 605, 505, 376, (218,2), 32]
        #inverted_cfg = [32, 218, (376,2), 505, (605,2), 720, (735,2), 720, 677, 605, 505, 376, (218,2), 32] 
        #down_parabola_cfg = [735, 720, (605,2), 505, (376,2), 218, (32,2), 218, 376, 505, 605, 677, (720,2), 735]  

        fixed_cfg = [381, 381, (381,2), 381, (381,2), 381, (381,2), 381, 381, 381, 381, 381, (381,2), 1024]
        defaultcfg = [32, 64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
        reverse_base_cfg = [1024, 512, (512,2), 512, (512,2), 512, (512,2), 256, 256, 128, 128, 64, (32,2), 1024] 
        parabola_cfg = [941, 661, (432,2), 254, (127,2), 51, (26,2), 51, 127, 254, 432, 661, (941,2), 1024]  
        down_parabola_cfg = [32, 207, (350,2), 461, (540,2), 588, (604,2), 588, 540, 461, 350, 207, (32,2), 1024]

        if f_version == 'Uniform': return fixed_cfg
        if f_version == 'Base': return defaultcfg
        if f_version == 'Reverse-Base': return reverse_base_cfg
        if f_version == 'Quadratic': return parabola_cfg
        if f_version == 'Negative-Quadratic': return down_parabola_cfg

        return


    def _set_proportional_filters(self, filter_list, f_proportion):
        temp_list = []
        for item in filter_list:
            if isinstance(item, int):
                new_value = int(item*f_proportion)
                if new_value<1: new_value = 1
                temp_list.append(new_value)
            elif isinstance(item, tuple):
                new_value = int(item[0]*f_proportion)
                if new_value<1: new_value = 1
                temp_list.append((new_value, item[1]))
            else:
                temp_list.append(item)
        return temp_list




def test():
    #net = MobileNet()
    net = mobilenet()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()

