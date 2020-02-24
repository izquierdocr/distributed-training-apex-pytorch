from torchvision import datasets

def UserDefinedImagenet(root='', split='train', transform=None, **kwargs):
    """Selected imagenet classes Dataset"""

    if split:
        return datasets.ImageFolder(root=root + '/train', transform=transform)
    else:
        return datasets.ImageFolder(root=root + '/val', transform=transform)

