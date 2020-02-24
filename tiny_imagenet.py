
from torchvision import datasets


def TinyImagenet(root='', split='train', transform=None, **kwargs):
    """Selected imagenet classes Dataset"""

    if split:
        return datasets.ImageFolder(root=root + '/train', transform=transform)
    else:
        return datasets.ImageFolder(root=root + '/val', transform=transform)





# --------------------------------------------------------------------------
#  For splitting the images in folders and use only ImageFolder function

import os
import csv


def process_train(root):

    path_data = os.path.join(root, 'train')
    image_classes = os.listdir(path_data)


    for class_name in image_classes:
        print(class_name)
        os.remove(os.path.join(path_data, class_name, class_name + "_boxes.txt"))

        file_path_data = os.path.join(path_data, class_name, 'images')
        image_names = os.listdir(file_path_data)
        #print(image_names)
        for file_name in image_names:
            print(file_name)
            os.rename(os.path.join(file_path_data, file_name), os.path.join(path_data, class_name, file_name))
            #break
        os.rmdir(file_path_data)
        #break



def process_val(root):

    all_classes_file = os.path.join(root, 'wnids.txt')
    #all_clasess = []
    with open(all_classes_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            #all_classes.append(row[0])
            os.mkdir(os.path.join(root, 'val', row[0]))
    #print(all_classes)


    path_data = os.path.join(root, 'val')
    image_classes_file = os.path.join(path_data, 'val_annotations.txt')
    with open(image_classes_file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            print(row[0], row[1])
            os.rename(os.path.join(path_data, 'images', row[0]), os.path.join(path_data, row[1], row[0]))

    os.remove(os.path.join(path_data, "val_annotations.txt"))
    os.rmdir(os.path.join(path_data, "images"))




if __name__ == '__main__':

    print('----------------------')

    print('Processing training folder...')
    #process_train('/media/hal/DATA/Datasets/tiny-imagenet')

    print('Processing validation folder...')
    #process_val('/media/hal/DATA/Datasets/tiny-imagenet')

    print('Done.')
