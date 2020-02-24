'''

python new_training_scratch.py --dataset mnist --path-data /media/hal/DATA/Datasets --train-batch 16 --test-batch 16 --epochs 2 --resume ./logs --save ./logs --arch vgg19 --distributed --nodes 2 -node-rank 0 --master-ip 192.168.1.2 --master-port 8888 --gpus 0

python new_training_scratch.py --dataset mnist --path-data ./data --train-batch 16 --test-batch 16 --epochs 2 --resume ./logs --save ./logs --arch vgg19 --distributed --nodes 2 -node-rank 1 --master-ip 192.168.1.2 --master-port 8888 --gpus 0

/media/hal/DATA/Datasets/

'''


import os
import random
import shutil


import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from apex.parallel import DistributedDataParallel as DDP
from apex import amp
import torch.nn as nn


from arguments import parse_arguments
from loggers import initialise_loggers, save_loggers, save_summary, AverageMeter, TimeLapse, ListMeter
from datasets import get_data_loaders
from metrics import get_model_metrics

import models


def main():

    args = parse_arguments()

    #print('>>>>>>>>>>>SLURM_NODELIST', os.environ['SLURM_NODELIST'])
    #print('>>>>>>>>>>>SLURM_STEP_NODELIST', os.environ['SLURM_STEP_NODELIST'])
    #print('>>>>>>>>>>>SLURM_NODEID', os.environ['SLURM_NODEID'])
    #print('>>>>>>>>>>>SLURM_PROCID', os.environ['SLURM_PROCID'])
    #print('>>>>>>>>>>>SLURM_LOCALID', os.environ['SLURM_LOCALID'])

    print('===>>>', 'Starting distributed...')
    # start threads for each of chosen gpus within one node
    mp.spawn(distributed_node, nprocs=len(args.gpus), args=(args,))
    print('===>>>', 'Distributed finished.')
    #print(args)



def distributed_node(process_id, args):

    print('===>>> Process', process_id, 'in Node', args.node_rank)
    total_time = TimeLapse()
    total_time.start()

    args.process_id = process_id
    loggers = initialise_loggers(args)
    set_training_environment(args)

    train_loader, val_loader, test_loader = get_data_loaders(args)

    checkpoint = recover_saved_session(args)

    model, optimizer, criterion, saved_loggers = build_model(args, checkpoint)

    if saved_loggers is not None:
        loggers = saved_loggers
    else:
        get_model_metrics(model, args, loggers)


    ### ToDo
    ### This is needed to add new fields in partially trained files due to saved_loggers delete no saved keys
    ### Maybe can be solved automatically by adding a function after updating that compares a new empty loggers
    ### with the existing logger and add non existing keys (or deleting currently used ones?)
    if 'epoch_number' not in loggers: loggers['epoch_number'] = ListMeter() 


    #print_loggers(loggers)

    print('===>>> Model have been trained for {} epochs from {} required'.format(args.start_epoch, args.epochs) )

    for epoch in range(args.start_epoch, args.epochs):

        ###args.epoch = epoch
        loggers['last_epoch'] = epoch

        adjust_lr(optimizer, args, loggers)

        train(model, train_loader, optimizer, criterion, args, loggers)
        evaluate(model, val_loader, criterion, args, loggers)
        if test_loader is not None:
            evaluate(model, test_loader, criterion, epoch, args, loggers, validation=False)



        # Save checkpoint only in master process
        if args.node_rank==0 and args.process_id==0:

            is_best = loggers['epoch_val_performance'].val > loggers['best_val_performance']
            if is_best:
                loggers['best_val_performance'] = loggers['epoch_val_performance'].val
                loggers['train_best_val_performance'] = loggers['epoch_train_performance'].val
                if test_loader is not None:
                    loggers['test_best_val_performance'] = loggers['epoch_test_performance'].val
                loggers['epoch_best_val_performance'] = loggers['last_epoch']

            save_checkpoint( { 'epoch': epoch + 1, 
                               'model': model.state_dict(),
                               'optimizer': optimizer.state_dict(),
                               'amp': amp.state_dict(),
                               'loggers': loggers,
                             }, is_best, args)
            save_loggers(loggers, args)
            #print_loggers(loggers)


    total_time.stop()
    loggers['total_time'] = loggers['total_time'] + total_time.time()   ###### Have to solve how to recover time if process is broken before reachen final epoch where the total time is computed
    save_loggers(loggers, args)
    save_summary(loggers, args)
    #print('This session time:', total_time.time())
    print('Total time for ', args.epochs, 'epochs:', loggers['total_time'])




def train(model, data_loader, optimizer, criterion, args, loggers):
    avg_loss = AverageMeter()
    performance = AverageMeter()
    avg_data_time = AverageMeter()
    avg_process_time = AverageMeter()
    data_time = TimeLapse()
    process_time = TimeLapse()

    model.train()
    data_time.start()
    check_time = TimeLapse()
    check_time.start()
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        #data, target = Variable(data), Variable(target)
        data_time.stop()
        process_time.start()
        #print('\nDATA:', check_time.time())
        #check_time.start()

        output = model(data)

        #print('\nFORWARD:', check_time.time())
        #check_time.start()

        loss = criterion(output, target)

        #print('\nLOSS:', check_time.time())
        #check_time.start()
        # compute gradient and do SGD step
        optimizer.zero_grad()

        #print('\nGRADIENT:', check_time.time())
        #check_time.start()
        if args.distributed:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()  # use amp for mixed precision

        #print('\nBACK:', check_time.time())
        #check_time.start()
        optimizer.step()
        process_time.stop()
        #print('\nOPTIMIZER:', check_time.time())
        #check_time.start()

        #evaluate model with train set
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        #print('\nEVALUATION:', check_time.time())
        #check_time.start()

        #'''
        if args.distributed:
            # We manually reduce and average the metrics across processes. In-place reduce tensor.
            ###reduced_loss = loss.data
            reduced_loss = reduce_tensor(loss.data, args)
            prec1 = reduce_tensor(prec1, args)
            ###prec5 = reduce_tensor(prec5, args)
        else:
            reduced_loss = loss.data

        # item incurs in a host<->device sync
        avg_loss.update(reduced_loss.item())
        performance.update(prec1.item())
        #'''
        torch.cuda.synchronize()


        ##print('\nLOGGERS:', check_time.time())

        # times are measured for each individual process and only master saves its own
        avg_data_time.update(data_time.time())
        avg_process_time.update(process_time.time())


        if batch_idx % args.log_interval == 0:
            print('\r=> Train: {}/{} [{}/{} ({:.1f}%)] Loss: {:.5f} Acc: {:.4f} '.format(
                loggers['last_epoch']+1, args.epochs, batch_idx, len(data_loader),
                100. * batch_idx / len(data_loader), avg_loss.val, performance.val), end='')

        ##check_time.start()
        data_time.start()

    #### ToDo:  check cyclic lr
    loggers['epoch_number'].update(loggers['last_epoch']+1)
    loggers['epoch_train_loss'].update(avg_loss.avg)
    loggers['epoch_train_performance'].update(performance.avg)
    loggers['train_time_load_data'].update(avg_data_time.avg)
    loggers['train_time_process_data'].update(avg_process_time.avg)


    print('\r=> Train: {}/{} ({:.1f}%) Loss: {:.5f} Acc: {:.4f}  Data: {:.4f}  Proc: {:.4f} '.format(
                loggers['last_epoch']+1, args.epochs, 100. * batch_idx / len(data_loader), avg_loss.avg, performance.avg,
                avg_data_time.avg, avg_process_time.avg) )

    #print('\nEPOCH:', check_time.time())
    #print('\nFINAL:', check_time.time())




def evaluate(model, data_loader, criterion, args, loggers, validation=True):
    avg_loss = AverageMeter()
    performance = AverageMeter()
    avg_data_time = AverageMeter()
    avg_process_time = AverageMeter()
    data_time = TimeLapse()
    process_time = TimeLapse()

    model.eval()
    data_time.start()
    for batch_idx, (data, target) in enumerate(data_loader):
        data_time.stop()
        process_time.start()

        if args.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        #data, target = Variable(data), Variable(target)

        with torch.no_grad():
            output = model(data)
            loss = criterion(output, target)

        process_time.stop()

        #evaluate model with train set
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        if args.distributed:
            # We manually reduce and average the metrics across processes. In-place reduce tensor.
            ###reduced_loss = loss.data
            reduced_loss = reduce_tensor(loss.data, args)
            prec1 = reduce_tensor(prec1, args)
            ###prec5 = reduce_tensor(prec5, args)
        else:
            reduced_loss = loss.data

        # item incurs in a host<->device sync
        avg_loss.update(reduced_loss.item())
        performance.update(prec1.item())
        torch.cuda.synchronize()

        # times are measured for each individual process and only master saves its own
        avg_data_time.update(data_time.time())
        avg_process_time.update(process_time.time())

        if batch_idx % args.log_interval == 0:
            print('\r=> Eval: {}/{} [{}/{} ({:.1f}%)] Loss: {:.5f} Acc: {:.4f} '.format(
                loggers['last_epoch']+1, args.epochs, batch_idx, len(data_loader),
                100. * batch_idx / len(data_loader), avg_loss.val, performance.val), end='')

        data_time.start()

    if validation:
        loggers['epoch_val_loss'].update(avg_loss.avg)
        loggers['epoch_val_performance'].update(performance.avg)
        loggers['val_time_load_data'].update(avg_data_time.avg)
        loggers['val_time_process_data'].update(avg_process_time.avg)
    else:
        loggers['epoch_test_loss'].update(avg_loss.avg)
        loggers['epoch_test_performance'].update(performance.avg)
        loggers['test_time_load_data'].update(avg_data_time.avg)
        loggers['test_time_process_data'].update(avg_process_time.avg)

    print('\r=> Eval: {}/{} ({:.1f}%) Loss: {:.5f} Acc: {:.4f}  Data: {:.4f}  Proc: {:.4f} '.format(
                loggers['last_epoch']+1, args.epochs, 100. * batch_idx / len(data_loader), avg_loss.avg, performance.avg,
                avg_data_time.avg, avg_process_time.avg) )


def reduce_tensor(tensor, args):
    # Reduces the tensor data across all machines
    rt = tensor.clone()  # The function operates in-place.
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt


def adjust_lr(optimizer, args, loggers):
    ### Notice that when state is loaded, param_group['lr'] is not the original args.lr 
    ### but one according to previous final epoch.
    if loggers['epoch_learning_rate'].count > 0:
        last_lr = loggers['epoch_learning_rate'].val
    else:
        last_lr = args.lr

    avg_lr = AverageMeter()
    if loggers['last_epoch'] in [int(args.epochs*0.5), int(args.epochs*0.75)]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
            avg_lr.update(param_group['lr'])
        last_lr = avg_lr.avg
    loggers['epoch_learning_rate'].update(last_lr)



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




def set_training_environment(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.gpu_id = args.gpus[args.process_id]

    if 'SLURM_NODEID' in os.environ:
        args.node_rank = int(os.environ['SLURM_NODEID'])
    if 'SLURM_NODELIST' in os.environ:
        args.master_ip = args.master_ip[0:3] + args.master_ip[4:6]

    if args.distributed:
        args.world_size = len(args.gpus) * args.nodes
        args.rank = args.node_rank * len(args.gpus) + args.process_id

        # debugging network connections
        #os.environ['NCCL_DEBUG'] = 'INFO'
        #os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'

        environment = 'BCp4'    #   BCp4    DeepRoom  Local
        #if environment == 'BCp4':

        if environment == 'DeepRoom':
            os.environ['NCCL_SOCKET_IFNAME'] = 'enp0s31f6'
            
        #os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['MASTER_ADDR'] = args.master_ip
        os.environ['MASTER_PORT'] = str(args.master_port)

        #print('INNN>>>CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
        #print('INNN>>>SLURM_NODELIST', os.environ['SLURM_NODELIST'])
        #print('INNN>>>SLURM_NODEID', os.environ['SLURM_NODEID'])
        #print('INNN>>>SLURM_PROCID', os.environ['SLURM_PROCID'])
        #print('INNN>>>SLURM_LOCALID', os.environ['SLURM_LOCALID'])

        '''
        /home/wp/.miniconda3/envs/ramon-pytorch/lib/python3.7/multiprocessing/semaphore_tracker.py:144: 
        UserWarning: semaphore_tracker: There appear to be 20 leaked semaphores to clean up at shutdown len(cache))
        '''
        os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'

        #os.environ['OMP_NUM_THREADS'] = '1'  # speed up distributed backward and optimizer

        dist.init_process_group(
            backend='nccl',    #    nccl    gloo     mpi
            init_method='env://',    #       'env://'         'tcp://192.168.1.2:8888'
            world_size=args.world_size,
            rank=args.rank)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        # if not running in SLURM then assign which GPU to use (slurm assign GPU ID automatically
        if 'SLURM_NODEID' not in os.environ: 
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpus)
        torch.cuda.set_device(args.gpu_id)
        torch.cuda.manual_seed(args.seed)
        assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if not os.path.exists(args.save):
        os.makedirs(args.save)


def build_model(args, checkpoint):

    #'''
    ### or the final layer can be built in this module 
    model = models.__dict__[args.arch](args)
    #'''
    #model = ConvNet()


   # define loss function (criterion) and optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    saved_loggers = None

    if args.cuda:
        model.cuda()
        cudnn.benchmark = True
        criterion =  criterion.cuda()

    if args.distributed:
    # Wrap the model for distributed training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_level) ## O0 only float O2 mixed precision
        model = DDP(model)

    #   https://github.com/NVIDIA/apex
    #   Note that we recommend calling the load_state_dict methods after amp.initialize.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        saved_loggers = checkpoint['loggers']
        amp.load_state_dict(checkpoint['amp'])
        args.start_epoch = checkpoint['epoch']

    return model, optimizer, criterion, saved_loggers



def recover_saved_session(args):
    # recover session data if exist
    checkpoint = None
    if args.resume:
        args.resume = os.path.join(args.resume, args.checkpoint_file)
        if os.path.isfile(args.resume):
            print("===>>> Loading checkpoint '{}'".format(args.resume), end='')
            checkpoint = torch.load(args.resume)
            print("\r===>>> Checkpoint '{}' loaded".format(args.resume))
        else:
            print("\r===>>> No checkpoint found at '{}'".format(args.resume))
    return checkpoint



def save_checkpoint(checkpoint, is_best, args):
    print("===>>> Saving checkpoint...", end='')
    torch.save(checkpoint, os.path.join(args.save, args.checkpoint_file))
    if is_best:
        shutil.copyfile(os.path.join(args.save, args.checkpoint_file), os.path.join(args.save, args.best_model_file))
    print("\r===>>> Checkpoint saved at", args.checkpoint_file)


if __name__ == '__main__':
    main()

