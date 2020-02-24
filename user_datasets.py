from __future__ import print_function
#from torchvision import datasets, transforms
#import torch.utils.data as data
from torch.utils.data import Dataset
#from torch.utils.data.sampler import SubsetRandomSampler

import os
import numpy as np
#import pandas as pd
from PIL import Image 


class ArabicMNIST(Dataset):
    """Arabic Handwritten Digits Dataset"""

    def __init__(self, train=True, root='', transform=None, **kwargs):

        if train:
            data_files = ['train_image.csv', 'train_label.csv']
        else:
            data_files = ['test_image.csv', 'test_label.csv']

        #root = os.path.join(root, 'arabic-digits')
        path_data = os.path.join(root, data_files[0])
        #self.data_image = pd.read_csv(path_data)
        self.data_image = np.genfromtxt(path_data, delimiter=',', dtype= 'uint8')

        path_data = os.path.join(root, data_files[1])
        #self.data_label = pd.read_csv(path_data)
        self.data_label = np.genfromtxt(path_data, delimiter=',', dtype= int)

        self.transform = transform

    def __len__(self):
        return len(self.data_image)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
        #    idx = idx.tolist()

        image = self.data_image[idx]
        image = np.reshape(image, (28, 28))
        image = Image.fromarray(image.astype('uint8'), 'L')
        label = self.data_label[idx]

        if self.transform:
            image = self.transform(image)

        return (image, label)




if __name__ == '__main__':

    '''
    dataloader = datasets.MNIST
    path_data = '~/datasets/mnist'
    transform_train = None
    trainset = dataloader(root=path_data, train=False, download=False, transform=transform_train)
    print('MNIST')
    print(len(trainset))
    print(len(trainset[0]))
    print(trainset[0][0].size)
    print(trainset[0][1])
    '''

    print('----------------------')

    dataloader = ArabicMNIST
    path_data = '/home/ri16164/datasets/arabic-digits'
    transform_train = None
    trainset = dataloader(root=path_data, train=False, transform=transform_train)
    print('Arabic-MNIST')
    print(len(trainset))
    print(len(trainset[0]))
    print(trainset[0][0].size)
    print(trainset[22][1])


    print('Done.')
