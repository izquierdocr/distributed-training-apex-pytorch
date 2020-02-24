
from datetime import date
import time
import os
import json


class TimeLapse(object):
    def __init__(self):
        self.initial_time = None
        self.final_time = None

    def start(self):
        self.initial_time = time.time()
        self.final_time = None

    def stop(self):
        self.final_time = time.time()

    def time(self):
        if self.initial_time is None:
            return None
        if self.final_time is not None:
            return self.final_time - self.initial_time
        else:
            return time.time() - self.initial_time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ListMeter(object):
    """Stores and computes the average and all values"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.list = []

    def update(self, val, n=1):
        self.list.append(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialise_loggers(args):
    loggers = {}

    # general data
    loggers['total_time'] = 0.0
    loggers['date'] = date.today().strftime("%d/%m/%Y")

    loggers['best_val_performance'] = 0.0
    loggers['train_best_val_performance'] = 0.0
    loggers['test_best_val_performance'] = 0.0
    loggers['epoch_best_val_performance'] = 0.0
    loggers['last_epoch'] = 0

    loggers['model_parameters'] = 0.0
    loggers['model_memory_cuda'] = 0.0
    loggers['model_memory_computed'] = 0.0
    loggers['model_flops'] = 0.0
    loggers['model_input_size'] = 0.0
    loggers['model_passes_size'] = 0.0
    loggers['model_parameters_size'] = 0.0


    # averaged data by epoch
    loggers['train_time_load_data'] = AverageMeter()
    loggers['val_time_load_data'] = AverageMeter()
    loggers['test_time_load_data'] = AverageMeter()
    loggers['train_time_process_data'] = AverageMeter()
    loggers['val_time_process_data'] = AverageMeter()
    loggers['test_time_process_data'] = AverageMeter()

    # training history by epoch
    loggers['epoch_number'] = ListMeter()
    loggers['epoch_learning_rate'] = ListMeter()
    loggers['epoch_train_loss'] = ListMeter()
    loggers['epoch_val_loss'] = ListMeter()
    loggers['epoch_test_loss'] = ListMeter()
    loggers['epoch_train_performance'] = ListMeter()
    loggers['epoch_val_performance'] = ListMeter()
    loggers['epoch_test_performance'] = ListMeter()

    return loggers


def summarise_epochs(loggers):
    new_loggers = {}
    list_title = []
    list_table = []
    for key, value in loggers.items():
        if type(value) == AverageMeter:
            new_loggers[key] = value.avg
        elif type(value) == ListMeter:
            if len(value.list) > 0:
                list_title.append(key)
                list_table.append(value.list)
        elif type(value) == TimeLapse:
            new_loggers[key] = value.time()
        else:
            new_loggers[key] = value

    list_table = list(map(list, zip(*list_table)))
    new_loggers['epoch_titles'] = list_title
    new_loggers['epoch_values'] = list_table
    return new_loggers


def print_loggers(loggers):
    for key, value in summarise_epochs(loggers).items():
        print(key, end=': ')
        print(value)



def save_loggers(loggers, args):
    print("===>>> Saving loggers...", end='')
    full_filename = os.path.join(args.save, args.loggers_file)

    args_dictionary = vars(args)
    loggers_dictionary = summarise_epochs(loggers)
    all_info = {**args_dictionary, **loggers_dictionary}

    with open(full_filename, 'w') as fp:
        json.dump(all_info, fp, indent=4)

    print("\r===>>> Loggers saved at", full_filename)




def save_summary(loggers, args):
    print("===>>> Saving global summary...", end='')

    args_dictionary = vars(args)
    loggers_dictionary = summarise_epochs(loggers)
    all_info = {**args_dictionary, **loggers_dictionary}

    fields_to_save = ["dataset", "train_split", "train_batch", "test_batch", "epochs", "lr", "seed", "amp_level", "arch", "width_multiplier", "template", "total_time", "date", "best_val_performance", "epoch_best_val_performance", "model_parameters", "model_memory_cuda", "model_memory_computed", "model_flops", "train_time_load_data", "val_time_load_data", "train_time_process_data", "val_time_process_data"]

    file_exist = os.path.isfile(args.summary_file) 
    f = open(args.summary_file,"a+")
    separator = ','

    if not file_exist:
        f.write(separator.join(fields_to_save) + '\n')

    values=[]
    for key in fields_to_save:
        value = all_info[key]
        if type(value) is not str:
            value = str(value)
        values.append(value)
        
    f.write( separator.join(values) + '\n' )
    f.close() 

    print("\r===>>> Global summary added to", args.summary_file)





if __name__ == '__main__':
    print('Initialise all metrics taken before/during/after training')


