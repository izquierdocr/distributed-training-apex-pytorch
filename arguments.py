import argparse


class Range(object):
    '''
     Uses:
          parser.add_argument('--foo', type=float, choices=Range(0.0, 1.0))
          parser.add_argument('--bar', type=float, choices=[Range(0.0, 1.0), Range(2.0,3.0)])
    '''
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self




def parse_arguments():
    # EXAMPLE:  python new_training_scratch.py --dataset cifar10 --path-data ./data --train-batch 16 --test-batch 16 --epochs 2 --resume ./logs --save ./logs --arch vgg19 --distributed -n 1 -nr 0 --master-ip 192.168.1.2 --master-port 8888


    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    # dataset
    parser.add_argument('--dataset', type=str.lower, required=True,
                        choices=['cifar10', 'cifar100', 'mnist', 'fashionmnist', 'arabic-mnist',
                                 'tiny-imagenet', 'imagenet2012'],
                        help='training dataset')
    parser.add_argument('--train-split', type=float, default=0.0,
                        help='proportion of train set to be split to derive validation (default: 0.0. No test set)')
    parser.add_argument('-p', '--path-data', type=str, required=True,
                        help='root directory where all the datasets are stored')
    parser.add_argument('--train-batch', type=int, metavar='N', required=True,
                        help='input batch size for training')
    parser.add_argument('--test-batch', type=int, metavar='N', required=True,
                        help='input batch size for testing')

    # training
    parser.add_argument('--epochs', type=int, metavar='N', required=True,
                        help='number of epochs to train')
    # if history is found in the model, take the initial epoch from there
    parser.add_argument('--start-epoch', default=0, type=int,
                        help=argparse.SUPPRESS)
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    #  ToDo: add type of varying lr and epochs of change
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')

    # logs and files
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--checkpoint-file', type=str, default='checkpoint.pth.tar',
                        help='file name to save checkpoints of model state')
    parser.add_argument('--best-model-file', type=str, default='model_best.pth.tar',
                        help='file name to save best performance model state')
    parser.add_argument('--loggers-file', type=str, default='loggers.json',
                        help='file name to save loggers information in text format')
    parser.add_argument('--summary-file', type=str, default='./logs/summary_all.txt',
                        help='file name to save summary of all historic training processes performed')


    # saving models and logs
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH',
                        help='path to save prune model (default: current directory)')
    # statistics will be saved inside checkpoint file
    #parser.add_argument('--summary-file', default='statistics.txt', type=str,
    #                    help='file to concentrate summary for all trained networks')

    # process
    # how to activate distributed, multigpu data/model
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpus', default=[0], type=int, nargs='+', choices=range(8),
                        help='gpu(s) to be used')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--workers', type=int, default=4, metavar='W',    #### IN DISTRIBUTED SAYS 0
                        help='number of worker threads for loading data (default: 4)')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='enable pin memory for threaded data loading')

    # distributed
    parser.add_argument('--distributed', action='store_true', default=False,
                        help='enables distributed training')
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of distributed nodes to run togheter (default: 1)')
    # can receive a list of gpu's by node but no speed up because other nodes will be slower the the one with more gpus
    # so the solutions is better to run 1 gpu by node and use several nodes in computers with several gpus
    #parser.add_argument('-g', '--gpus', default=1, type=int,
    #                    help='number of gpus per node')
    parser.add_argument('--node-rank', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--master-ip', type=str, default='localhost',
                        help='IP address where the master node will be running')
    parser.add_argument('--master-port', type=int, choices=range(8000, 10000), default=8888,
                        help='Port number where the master node will be running')
    parser.add_argument('--amp-level', type=str.upper, default='O0', choices=['O0', 'O1', 'O2', 'O3'],
                        help='Level from Apex AMP mixed precision (Default O0: fp32)')

    # model
    parser.add_argument('--arch', type=str.lower, required=True,
                        choices=['vgg19', 'mobilenet1', 'inception1', 'resnet50'], 
                        help='architecture to use')
    parser.add_argument('--width-multiplier', default=1.0, type=float, choices=Range(0.0, 5.0),
                        help='A number > 0 to grow original number of filters in each layer (1.0 is original size)')
    parser.add_argument('--template', default='base', type=str.lower,
                        choices=['base', 'reverse-base', 'uniform', 'quadratic', 'negative-quadratic'],  
                        help='version of template filter distribution')
    return parser.parse_args()



if __name__ == '__main__':
    print('Configure general libraries')
