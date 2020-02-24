'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
# Al check https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py for reduced resnet

import torch
import torch.nn as nn
import torch.nn.functional as F



__all__ = ['resnet']


defaultcfg = [64, 128, 256, 512]

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, final=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            #'''
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            '''
            if final:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, 200, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(200)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                )
            '''

    def forward(self, x):
        ###print('Sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee      Block input' , x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        ###out = self.conv1(x)
        ###print('Sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee      Block conv1' , out.size())
        ###out = self.bn1(out)
        ###print('Sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee      Block bn1' , out.size())
        ###out = F.relu(out)
        ###print('Sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee      Block Pass' , out.size())
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        ###print('Sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee      Block' , out.size())
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, final=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    #def __init__(self, block, num_blocks, num_classes=10):
    def __init__(self, block, num_blocks, dataset='cifar10', cfg=None, f_version='baseline', f_proportion=None):
        super(ResNet, self).__init__()

        if cfg == None:
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


        #self.in_planes = 64
        self.in_planes = cfg[0]

        self.conv1 = nn.Conv2d(in_channels, cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cfg[0])
        self.layer1 = self._make_layer(block, cfg[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, cfg[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, cfg[3], num_blocks[3], stride=2)
        self.linear = nn.Linear(cfg[-1]*block.expansion, num_classes)
        ###print('LINEAR     ', cfg[-1]*block.expansion )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        ###print('STRIDES     ' , strides, '  >>  ', planes)
        ###i=0
        ###final = False
        for stride in strides:
            ###i=i+1
            ###if i==len(strides):final=True
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        ###print('Sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee      layer1' , out.size())
        out = self.layer1(out)
        ###print('Sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee      layer2' , out.size())
        out = self.layer2(out)
        ###print('Sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee      layer3' , out.size())
        out = self.layer3(out)
        ###print('Sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee      layer4' , out.size())
        out = self.layer4(out)
        ###print('Sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee      pool' , out.size())
        out = F.avg_pool2d(out, 4)
        ###print('Sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee      view' , out.size())
        out = out.view(out.size(0), -1)
        ###print('Sizeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee      linear' , out.size())
        out = self.linear(out)
        ###print(5/0)
        return out

    def _set_filters(self, f_version):
        fixed_cfg = [200, 200, 200, 200] # fixed 240 per block     236 per layer
        reverse_base_cfg = [512, 256, 128, 64]
        parabola_cfg = [416, 64, 64, 416]
        down_parabola_cfg = [64, 416, 416, 64]


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
                temp_list.append(int(item*f_proportion))
            else:
                temp_list.append(item)
        return temp_list






def ResNet18(**kwargs):
    return ResNet(BasicBlock, [2,2,2,2],**kwargs)

def ResNet34(**kwargs):
    return ResNet(BasicBlock, [3,4,6,3],**kwargs)

def ResNet50(**kwargs):
    return ResNet(Bottleneck, [3,4,6,3],**kwargs)

def ResNet101(**kwargs):
    return ResNet(Bottleneck, [3,4,23,3],**kwargs)

def ResNet152(**kwargs):
    return ResNet(Bottleneck, [3,8,36,3],**kwargs)



def resnet(depth=18, **kwargs):
    """
    Constructs a ResNet model.
    """
    if depth==18:
        return ResNet18(**kwargs)
    elif depth==34:
        return ResNet34(**kwargs)
    elif depth==50:
        return ResNet50(**kwargs)
    elif depth==101:
        return ResNet101(**kwargs)
    elif depth==152:
        return ResNet152(**kwargs)

    print('Undefined version')
    return None



def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()




