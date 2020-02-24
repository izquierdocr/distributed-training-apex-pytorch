'''GoogLeNet with PyTorch.'''
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/googlenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['inception']


defaultcfg = {
        'a3' : [192,  64,  96, 128, 16, 32, 32],
        'b3' : [256, 128, 128, 192, 32, 96, 64],
        'a4' : [480, 192,  96, 208, 16,  48,  64],
        'b4' : [512, 160, 112, 224, 24,  64,  64],
        'c4' : [512, 128, 128, 256, 24,  64,  64],
        'd4' : [512, 112, 144, 288, 32,  64,  64],
        'e4' : [528, 256, 160, 320, 32, 128, 128],
        'a5' : [832, 256, 160, 320, 32, 128, 128],
        'b5' : [832, 384, 192, 384, 48, 128, 128],
    }


class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class inception(nn.Module):
    #def __init__(self):
    def __init__(self, dataset='cifar10', depth=19, init_weights=True, cfg=None, f_version='baseline', f_proportion=None):
        super(inception, self).__init__()

        if cfg is None:
            cfg = defaultcfg

        print('Filterssssss:', cfg)
        if f_version != 'baseline': cfg = self._set_filters(f_version)
        if f_proportion is not None: cfg = self._set_proportional_filters(cfg, f_proportion)
        print('Final Filterssssss:', cfg)

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


        self.pre_layers = nn.Sequential(
            nn.Conv2d(in_channels, cfg['a3'][0], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg['a3'][0]),
            nn.ReLU(True),
        )

        '''
        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, num_classes)
        '''

        self.a3 = Inception(*cfg['a3'])
        self.b3 = Inception(*cfg['b3'])

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(*cfg['a4'])
        self.b4 = Inception(*cfg['b4'])
        self.c4 = Inception(*cfg['c4'])
        self.d4 = Inception(*cfg['d4'])
        self.e4 = Inception(*cfg['e4'])

        self.a5 = Inception(*cfg['a5'])
        self.b5 = Inception(*cfg['b5'])

        self.avgpool = nn.AvgPool2d(8, stride=1)
        final__filters_indexes = [1,3,5,6]
        self.linear = nn.Linear( sum( [cfg['b5'][i] for i in final__filters_indexes] ), num_classes)





    def forward(self, x):
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


    def _set_filters(self, f_version):

        fixed_cfg = {
            'a3' : [128, 128, 128, 128, 128, 128, 128],
            'b3' : [512, 128, 128, 128, 128, 128, 128],
            'a4' : [512, 128, 128, 128, 128, 128, 128],
            'b4' : [512, 128, 128, 128, 128, 128, 128],
            'c4' : [512, 128, 128, 128, 128, 128, 128],
            'd4' : [512, 128, 128, 128, 128, 128, 128],
            'e4' : [512, 128, 128, 128, 128, 128, 128],
            'a5' : [512, 128, 128, 128, 128, 128, 128],
            'b5' : [512, 128, 128, 128, 128, 128, 128],
        }
        reverse_base_cfg = {
            'a3' : [832, 384, 192, 384, 48, 128, 128],
            'b3' : [832, 256, 160, 320, 32, 128, 128],
            'a4' : [528, 256, 160, 320, 32, 128, 128],
            'b4' : [512, 112, 144, 288, 32,  64,  64],
            'c4' : [512, 128, 128, 256, 24,  64,  64],
            'd4' : [512, 160, 112, 224, 24,  64,  64],
            'e4' : [480, 192,  96, 208, 16,  48,  64],
            'a5' : [256, 128, 128, 192, 32, 96, 64],
            'b5' : [192,  64,  96, 128, 16, 32, 32],
        }
        parabola_cfg = {
            'a3' : [227, 227, 227, 227, 227, 227, 227],
            'b3' : [908, 154, 154, 154, 154, 154, 154],
            'a4' : [616, 101, 101, 101, 101, 101, 101],
            'b4' : [404, 69, 69, 69, 69, 69, 69],
            'c4' : [276, 59, 59, 59, 59, 59, 59],
            'd4' : [236, 69, 69, 69, 69, 69, 69],
            'e4' : [276, 101, 101, 101, 101, 101, 101],
            'a5' : [404, 154, 154, 154, 154, 154, 154],
            'b5' : [616, 227, 227, 227, 227, 227, 227],
        }
        down_parabola_cfg = {
            'a3' : [61, 61, 61, 61, 61, 61, 61],
            'b3' : [244, 112, 112, 112, 112, 112, 112],
            'a4' : [448, 148, 148, 148, 148, 148, 148],
            'b4' : [592, 170, 170, 170, 170, 170, 170],
            'c4' : [680, 177, 177, 177, 177, 177, 177],
            'd4' : [708, 170, 170, 170, 170, 170, 170],
            'e4' : [680, 148, 148, 148, 148, 148, 148],
            'a5' : [592, 112, 112, 112, 112, 112, 112],
            'b5' : [448, 61, 61, 61, 61, 61, 61],

            #'a3' : [492, 123, 123, 123, 123, 123, 123],
            #'b3' : [468, 117, 117, 117, 117, 117, 117],
            #'a4' : [400, 100, 100, 100, 100, 100, 100],
            #'b4' : [288, 72, 72, 72, 72, 72, 72],
            #'c4' : [128, 32, 32, 32, 32, 32, 32],
            #'d4' : [288, 72, 72, 72, 72, 72, 72],
            #'e4' : [400, 100, 100, 100, 100, 100, 100],
            #'a5' : [468, 117, 117, 117, 117, 117, 117],
            #'b5' : [492, 123, 123, 123, 123, 123, 123],
        }

        if f_version == 'Uniform': return fixed_cfg
        if f_version == 'Base': return defaultcfg
        if f_version == 'Reverse-Base': return reverse_base_cfg
        if f_version == 'Quadratic': return parabola_cfg
        if f_version == 'Negative-Quadratic': return down_parabola_cfg

        return


    def _set_proportional_filters(self, filter_list, f_proportion):
        previous_filters = None
        for incention_module in filter_list.keys():
            temp_list = []
            for item in filter_list[incention_module]:
                if isinstance(item, int):
                    new_value = int(item*f_proportion)
                    if new_value<1: new_value = 1
                    temp_list.append(new_value)
                    #temp_list.append(129) #135
                else:
                    temp_list.append(item)
            filter_list[incention_module] = temp_list
            if previous_filters is not None:
                filter_list[incention_module][0] = previous_filters
            final__filters_indexes = [1,3,5,6]
            previous_filters = sum( [filter_list[incention_module][i] for i in final__filters_indexes] )
        return filter_list



def test():
    #net = GoogLeNet()
    net = inception()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y.size())

# test()



