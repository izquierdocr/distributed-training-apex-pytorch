import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from .channel_selection import channel_selection


__all__ = ['densenet']

"""
densenet with basic block.
"""

class BasicBlock(nn.Module):
    def __init__(self, inplanes, cfg, expansion=1, growthRate=12, dropRate=0):
        super(BasicBlock, self).__init__()
        planes = expansion * growthRate
        #--print('Making basic block batch entering', inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        ###self.select = channel_selection(inplanes)
        ###self.conv1 = nn.Conv2d(cfg, growthRate, kernel_size=3, 
        ###                       padding=1, bias=False)
        ###print('CCCCCCFFFFFGGGG:',cfg)
        #--print('BAD Making basic block conv entering', cfg, 'and coming out', growthRate)
        #--print('Making basic block conv entering', inplanes, 'and coming out', growthRate)
        #self.conv1 = nn.Conv2d(cfg, growthRate, kernel_size=3,
        self.conv1 = nn.Conv2d(inplanes, growthRate, kernel_size=3,  
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        ###out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        ###print('===========>>>>>')
        if self.dropRate > 0:
            out = F.dropout(out, p=self.dropRate, training=self.training)


        out = torch.cat((x, out), 1)

        return out

class Transition(nn.Module):
    def __init__(self, inplanes, outplanes, cfg):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        ###self.select = channel_selection(inplanes)
        #--print('BAD Making transition entering', cfg, 'and coming out', outplanes)
        #--print('Making transition entering', inplanes, 'and coming out', outplanes)
        #self.conv1 = nn.Conv2d(cfg, outplanes, kernel_size=1,
        self.conv1 = nn.Conv2d(inplanes, outplanes, kernel_size=1, 
                               bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        ###out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = F.avg_pool2d(out, 2)
        return out

class densenet(nn.Module):

    def __init__(self, depth=40, 
        dropRate=0, dataset='cifar10', growthRate=12, compressionRate=1, cfg = None, f_version='baseline', f_proportion=None):
        super(densenet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3
        block = BasicBlock

        #growthRate = 10
        ###print('growthRate   =    ', growthRate)
        ###print('n   =    ', n)

        self.growthRate = growthRate
        self.dropRate = dropRate

        if cfg == None:
            cfg = []
            start = growthRate*2
            for i in range(3):
                cfg.append([start+12*i for i in range(n+1)])
                start += growthRate*12
                ###print('Filterssssss:', i  ,' ---', cfg)
            cfg = [item for sub_list in cfg for item in sub_list]

        # [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300]
        # [24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 168, 180, 192, 204, 216, 228, 240, 252, 264, 276, 288, 300, 312, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456]

        assert len(cfg) == 3*n+3, 'length of config variable cfg should be 3n+3'

        initial_filters = 2


        print('Filterssssss:', cfg)
        if f_version != 'baseline': cfg = self._set_filters(f_version)
        ####if f_proportion is not None: cfg = self._set_proportional_filters(cfg, f_proportion)
        if f_proportion is not None:
            if growthRate * f_proportion >=0.5 and growthRate * f_proportion<1: initial_filters = 1
            growthRate = int(growthRate * f_proportion) if int(growthRate * f_proportion)>1 else 1
        self.growthRate = growthRate
        print('Growth Rate:', growthRate, 'Starting multiplier:', initial_filters)

        #cfg = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        #j = 1
        #for i in range(len(cfg)):
        #    if (i+1)%3 == 0:
        #        cfg[i] = 30*j
        #        j=j+1

        print('Final Filterssssss:', cfg)
        self.cfg = cfg

        # self.inplanes is a global variable used across multiple
        # helper functions
        self.inplanes = int(growthRate * initial_filters)
        ### BAD  self.inplanes = cfg[0]
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1,
                               bias=False)
        #--print('dense1')
        self.dense1 = self._make_denseblock(block, n, cfg[0:n])
        #--print('trans1')
        self.trans1 = self._make_transition(compressionRate, cfg[n])
        #--print('dense2')
        self.dense2 = self._make_denseblock(block, n, cfg[n+1:2*n+1])
        #--print('trans2')
        self.trans2 = self._make_transition(compressionRate, cfg[2*n+1])
        #--print('dense3')
        self.dense3 = self._make_denseblock(block, n, cfg[2*n+2:3*n+2])
        #--print('bn')
        self.bn = nn.BatchNorm2d(self.inplanes)
        ###self.select = channel_selection(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        #--print('avgpool')
        self.avgpool = nn.AvgPool2d(8)

        #--print('fully')
        if dataset == 'cifar10':
            #--print('FC with', self.inplanes)
            #self.fc = nn.Linear(cfg[-1], 10)
            self.fc = nn.Linear(self.inplanes, 10)
        elif dataset == 'cifar100':
            #--print('FC with', self.inplanes)
            #self.fc = nn.Linear(cfg[-1], 100)
            self.fc = nn.Linear(self.inplanes, 100)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_denseblock(self, block, blocks, cfg):
        layers = []
        assert blocks == len(cfg), 'Length of the cfg parameter is not right.'
        for i in range(blocks):
            # Currently we fix the expansion ratio as the default value
            layers.append(block(self.inplanes, cfg = cfg[i], growthRate=self.growthRate, dropRate=self.dropRate))
            ###layers.append(block(self.inplanes, cfg = cfg[i], growthRate=cfg[i+1], dropRate=self.dropRate))
            #--print('MAKE DENSE:', cfg[i], self.growthRate)
            self.inplanes += self.growthRate

        return nn.Sequential(*layers)

    def _make_transition(self, compressionRate, cfg):
        # cfg is a number in this case.
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        #--print('TRANSITION:', inplanes, outplanes, cfg)
        self.inplanes = outplanes
        return Transition(inplanes, outplanes, cfg)

    def forward(self, x):
        #--print('######===========>>>>>  DOING FORWARD PASS')
        x = self.conv1(x)

        #--print('dense1')
        x = self.dense1(x)
        ###print('######===========>>>>>')
        #--print('trans1')
        x = self.trans1(x)
        #x = self.trans1(self.dense1(x))
        #--print('trans2')
        x = self.trans2(self.dense2(x))
        #--print('dense3')
        x = self.dense3(x)
        x = self.bn(x)
        ###x = self.select(x)
        x = self.relu(x)

        #--print('avgpool')
        x = self.avgpool(x)
        #--print('view')
        x = x.view(x.size(0), -1)
        #--print('fc', x.size())
        x = self.fc(x)

        return x




    def _set_fixed_filters(self, filter_list, f_size):
        temp_list = []
        for item in filter_list:
            if isinstance(item, int):
                temp_list.append(f_size)
            else:
                temp_list.append(item)
        return temp_list


    def _set_proportional_filters(self, filter_list, f_proportion):
        temp_list = []
        for item in filter_list:
            if isinstance(item, int):
                temp_list.append(int(item*f_proportion))
            else:
                temp_list.append(item)
        return temp_list

