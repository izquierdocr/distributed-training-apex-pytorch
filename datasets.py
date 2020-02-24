
from torch.utils.data.distributed import DistributedSampler

from torchvision import datasets, transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from user_datasets import ArabicMNIST
from user_defined_imagenet import UserDefinedImagenet
from tiny_imagenet import TinyImagenet

import os
import numpy as np


def get_data_loaders(args):


    dataset = args.dataset.lower()
    path_data = os.path.join(args.path_data, dataset)

    split =  args.train_split if (args.train_split>0.0 and args.train_split<1.0) else None
    kwargs = {'num_workers': args.workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    print('===>>> Preparing dataset %s' % dataset)
    dataset_info = {}
    if dataset == 'cifar10':
        # ToDo: add number of samples in train/val/test?
        args.input_size = (32, 32)
        args.input_channels = 3
        args.number_classes = 10
        dataset_loader = datasets.CIFAR10
        mean_data = (0.4914, 0.4822, 0.4465)
        std_data = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean_data, std_data),
        ])
        transform_val = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean_data, std_data),
        ])
    elif dataset == 'cifar100':
        args.input_size = (32, 32)
        args.input_channels = 3
        args.number_classes = 100
        dataset_loader = datasets.CIFAR100
        mean_data = (0.4914, 0.4822, 0.4465)
        std_data = (0.2023, 0.1994, 0.2010)
        transform_train = transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(32),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean_data, std_data),
        ])
        transform_val = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean_data, std_data),
        ])
    elif dataset == 'mnist':
        args.input_size = (28, 28)
        args.input_channels = 1
        args.number_classes = 10
        dataset_loader = datasets.MNIST
        mean_data = (0.1307,)
        std_data = (0.3081,)
        transform_train = transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(28),
                              #transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean_data, std_data),
        ])
        transform_val = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean_data, std_data),
        ])
    elif dataset == 'fashionmnist':
        args.input_size = (28, 28)
        args.input_channels = 1
        args.number_classes = 10
        dataset_loader = datasets.FashionMNIST
        mean_data = (0.5, )
        std_data = (0.5, )
        transform_train = transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(28),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean_data, std_data),
        ])
        transform_val = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean_data, std_data),
        ])
    elif dataset == 'arabic-mnist':
        args.input_size = (28, 28)
        args.input_channels = 1
        args.number_classes = 10
        dataset_loader = ArabicMNIST
        mean_data = (0.5, )
        std_data = (0.5, )
        transform_train = transforms.Compose([
                              transforms.Pad(4),
                              transforms.RandomCrop(28),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean_data, std_data),
        ])
        transform_val = transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize(mean_data, std_data),
        ])
    elif dataset == 'imagenet2012':
        args.input_size = (224, 224)
        args.input_channels = 3
        args.number_classes = 1000
        dataset_loader = datasets.ImageNet
        mean_data = (0.485, 0.456, 0.406)
        std_data = (0.229, 0.224, 0.225)
        transform_train = transforms.Compose([
                          transforms.RandomResizedCrop(224),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean_data, std_data),
        ])
        transform_val = transforms.Compose([
                          transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize(mean_data, std_data),
        ])
        
    elif dataset == 'tiny-imagenet':
        args.input_size = (64, 64)
        args.input_channels = 3
        args.number_classes = 200
        dataset_loader = TinyImagenet
        mean_data = (0.485, 0.456, 0.406)
        std_data = (0.229, 0.224, 0.225)
        transform_train = transforms.Compose([
                          transforms.RandomResizedCrop(64),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean_data, std_data),
        ])
        transform_val = transforms.Compose([
                          transforms.Resize(64),
                          transforms.ToTensor(),
                          transforms.Normalize(mean_data, std_data),
        ])
    elif dataset == 'user-imagenet':
        args.input_size = (224, 224)
        args.input_channels = 3
        args.number_classes = 5
        dataset_loader = UserDefinedImagenet
        mean_data = (0.485, 0.456, 0.406)
        std_data = (0.229, 0.224, 0.225)
        transform_train = transforms.Compose([
                          transforms.RandomResizedCrop(224),
                          transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean_data, std_data),
        ])
        transform_val = transforms.Compose([
                          transforms.Resize(256),
                          transforms.CenterCrop(224),
                          transforms.ToTensor(),
                          transforms.Normalize(mean_data, std_data),
        ])


    else:
        print('===>>> Dataset not in list')


###########

    '''
    if split is None:
        trainset = dataloader(root=path_data, split='train', download=False, transform=transform_train)
        valset = dataloader(root=path_data, split='val', download=False, transform=transform_val)
        testset = None

        train_loader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, **kwargs)
        val_loader = data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, **kwargs)
        test_loader = None

    else:
        trainset = dataloader(root=path_data, split='train', download=False, transform=transform_train)
        valset = dataloader(root=path_data, split='train', download=False, transform=transform_val)
        testset = dataloader(root=path_data, split='val', download=False, transform=transform_val)
   '''
######


    if split is None:
        if dataset in ['imagenet2012', 'tiny-imagenet', 'user-imagenet']:
            trainset = dataset_loader(root=path_data, split='train', download=False, transform=transform_train)
            valset = dataset_loader(root=path_data, split='val', download=False, transform=transform_val)
            testset = None
        else:
            trainset = dataset_loader(root=path_data, train=True, download=True, transform=transform_train)
            valset = dataset_loader(root=path_data, train=False, download=True, transform=transform_val)
            testset = None


        if args.distributed:
            train_sampler = DistributedSampler(trainset, num_replicas=args.world_size, rank=args.rank)

            train_loader = DataLoader(trainset, batch_size=args.train_batch, shuffle=False,
                              num_workers=args.workers, pin_memory=args.pin_memory,
                              sampler=train_sampler)

            val_sampler = DistributedSampler(valset, num_replicas=args.world_size, rank=args.rank)

            val_loader = DataLoader(valset, batch_size=args.test_batch, shuffle=False,
                              num_workers=args.workers, pin_memory=args.pin_memory,
                              sampler=val_sampler)



        else:
            train_loader = DataLoader(trainset, batch_size=args.train_batch, shuffle=True, **kwargs)
            val_loader = DataLoader(valset, batch_size=args.test_batch, shuffle=False, **kwargs)
        
        test_loader = None

    else:
        trainset = dataset_loader(root=path_data, train=True, download=False, transform=transform_train)
        valset = dataset_loader(root=path_data, train=True, download=False, transform=transform_val)
        testset = dataset_loader(root=path_data, train=False, download=False, transform=transform_val)

        num_samples = len(trainset)
        indices = list(range(num_samples))
        np.random.shuffle(indices) # shuffle=True incompatible with sampler in DataLoader
        split_index = int(np.floor(split * num_samples))
        train_idx, valid_idx = indices[:split_index], indices[split_index:]
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = data.DataLoader(trainset, sampler=train_sampler, batch_size=args.train_batch, **kwargs)
        val_loader = data.DataLoader(valset, sampler=valid_sampler, batch_size=args.test_batch, **kwargs)
        test_loader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, **kwargs)

    return train_loader, val_loader, test_loader


