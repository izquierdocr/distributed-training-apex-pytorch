
from itertools import product
import subprocess
import sys
import time
import argparse
import random
from datetime import datetime

PATH_ELEMENTS = []
LAST_VALID_TASK = 0



def process_arguments(arguments_dictionary):
    # Generate paths from elements in the dictionary
    list_of_options = []
    for option, value in arguments_dictionary.items():
        argument_name = '--' + option
        argument_value = value

        if option in ["save", "resume"]:
            argument_value = PATH_ELEMENTS[0]
            for element in PATH_ELEMENTS[1:]:
                if element in arguments_dictionary.keys():
                    argument_value = argument_value + arguments_dictionary[element]
                else:
                    argument_value = argument_value + element
        list_of_options.append(argument_name)
        list_of_options.append(argument_value)

    return list_of_options



def find_argument(value, args):
    return args[args.index(value) + 1]


def run_in_blue_crystal_phase_4(program_name='main.py', program_arguments='', job_name='dl-series'):

    bashCommand = ''

    bashCommand += '#!/bin/bash' + '\n'
    bashCommand += '#SBATCH --job-name=' + job_name + '\n'
    bashCommand += '#SBATCH --partition=gpu' + '\n'   # gpu gpu_veryshort
    bashCommand += '#SBATCH --gres=gpu:1' + '\n'   # 1
    bashCommand += '#SBATCH --nodes=' + find_argument('--nodes',program_arguments) + '\n'    # 1
    bashCommand += '#SBATCH --ntasks-per-node=1' + '\n'   # 1
    bashCommand += '#SBATCH --cpus-per-task=4' + '\n'   # 1
    bashCommand += '#SBATCH --mem=60000M' + '\n'
    bashCommand += '#SBATCH --time=5-00:00:00' + '\n'   # 7 days max in gpu and 1 hour in gpu_veryshort
    bashCommand += '#SBATCH -o outputs/out_%j.txt' + '\n'
    bashCommand += '#SBATCH -e outputs/err_%j.txt' + '\n'
    bashCommand += '#SBATCH --mail-type=END,FAIL' + '\n'   # notification- BEGIN,END,FAIL,ALL
    bashCommand += '#SBATCH --mail-user=ri16164@bristol.ac.uk' + '\n'
    bashCommand += '#SBATCH --no-requeue' + '\n'

    #! modules needed
    bashCommand += 'module load CUDA/8.0.44-GCC-5.4.0-2.26' + '\n'
    bashCommand += 'module load libs/cudnn/5.1-cuda-8.0' + '\n'
    bashCommand += 'cudaDevs=$(echo $CUDA_VISIBLE_DEVICES | sed -e \'s/,/ /g\')' + '\n'
    bashCommand += 'module load languages/anaconda2/5.0.1' + '\n'
    bashCommand += 'source activate ramon-pytorch' + '\n'
    bashCommand += 'export PATH=$HOME/.conda/envs/ramon-pytorch/bin:$PATH' + '\n'

    #! application name
    bashCommand += 'application="srun python ' + program_name + '"' + '\n'

    #! Run options for the application
    bashCommand += 'options="' + ' '.join(program_arguments) + ' --master-ip $SLURM_JOB_NODELIST --node-rank $SLURM_NODEID' + '"\n'

    #! change the working directory
    #! (default is home directory)
    bashCommand += 'cd $SLURM_SUBMIT_DIR' + '\n'

    bashCommand += 'echo Running on host `hostname`' + '\n'
    bashCommand += 'echo Time is `date`' + '\n'
    bashCommand += 'echo Directory is `pwd`' + '\n'
    bashCommand += 'echo Slurm job ID is $SLURM_JOBID' + '\n'
    bashCommand += 'echo This jobs runs on the following machines:' + '\n'
    bashCommand += 'echo $SLURM_JOB_NODELIST' + '\n'
    bashCommand += 'echo "Using GPUs at index $cudaDevs"' + '\n'

    bashCommand += 'echo Running application $application' + '\n'
    bashCommand += 'echo with parameters $options' + '\n'

    #! Run the threaded exe
    bashCommand += 'START=$(date +%s.%N)' + '\n'
    bashCommand += '$application $options' + '\n'
    bashCommand += 'END=$(date +%s.%N)' + '\n'

    bashCommand += 'DIFF=$(echo "$END - $START" | bc)' + '\n'
    bashCommand += 'echo Execution time in seconds is $DIFF' + '\n'


    exit_code = 0
    try:
        process = subprocess.Popen(['sbatch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, encoding='utf8')
        process.stdin.write(bashCommand)
        print (process.communicate()[0])
        process.stdin.close()
        #print('======>>>>>>', bashCommand)
        print('Process sent to queue.')

    except:

        print("I couldn't execute that command...")
        print("Unexpected error:", sys.exc_info()[0])
        exit_code = 1

    return exit_code



def run_in_command_line(program_name='main.py', program_arguments=''):

    exit_code = 0
    child_process = subprocess.Popen( ['python', program_name] + list(filter(len, program_arguments)) ) 
    exit_code = child_process.wait()

    return exit_code





def mnist():   # Change parameters only in this section
    global PATH_ELEMENTS
    global LAST_VALID_TASK
    o = {}

    LAST_VALID_TASK = 0 # Number of the last task that finished correctly to skip previous. 0: no task performed before

    o["distributed"] = ['']   # only flag with no options
    o["nodes"] = ['1']
    ###o["node-rank"] = ['0']    ####   from BC
    ###o["master-ip"] = ['localhost']    ####   from BC
    o["master-port"] =  [str(random.randint(8888, 8999))] # ['8888']
    o["amp-level"] = ['O0']

    o["gpus"] = ['0']    ####   in BCp4 always the first gpu is 0 even when physically can be different
    o["seed"] = ['7'] #['1','5','7']
    o["summary-file"] = ['./logs/templates_mnist.txt']

    o["dataset"] = ['mnist'] #['mnist', 'fashionmnist', 'arabic-mnist']
    o["train-split"] = ['0.0']

    o["arch"] = ['vgg19']  #['vgg19', 'resnet50', 'mobilenet1']
    o["width-multiplier"] = ['1.0'] # ['1.6', '1.3', '1.0', '0.8', '0.5', '0.25', '0.1', '0.05']
    o["template"] = ['base'] #['reverse-base', 'uniform', 'quadratic', 'negative-quadratic', 'base']

    o["lr"] = ['0.1']
    o["epochs"] = ['30'] #30
    o["train-batch"] = ['128']
    o["test-batch"] = ['16']

    o["path-data"] = ['/mnt/storage/scratch/ri16164/datasets']    ##  /mnt/storage/scratch/ri16164/datasets   /media/hal/DATA/Datasets 

    PATH_ELEMENTS = ['./logs/', 'dataset', '/', 'arch', '_', 'template', '_', 'width-multiplier', '_', 'seed'] #, '_', 'train-split']
    o["save"] = ['']
    o["resume"] = [''] # comment for disabling resume

    #o[""] = ['']

    return o


def cifar():   # Change parameters only in this section
    global PATH_ELEMENTS
    global LAST_VALID_TASK
    o = {}

    LAST_VALID_TASK = 0 # Number of the last task that finished correctly to skip previous. 0: no task performed before

    o["distributed"] = ['']   # only flag with no options
    o["nodes"] = ['1']
    ###o["node-rank"] = ['0']    ####   from BC
    ###o["master-ip"] = ['localhost']    ####   from BC
    o["master-port"] =  [str(random.randint(8888, 8999))] # ['8888']
    o["amp-level"] = ['O0']

    o["gpus"] = ['0']    ####   in BCp4 always the first gpu is 0 even when physically can be different
    o["seed"] = ['7'] #['1','5','7']
    o["summary-file"] = ['./logs/templates_cifar.txt']

    o["dataset"] = ['cifar100'] #['cifar10', 'cifar100']
    o["train-split"] = ['0.0']

    o["arch"] = ['vgg19']  #['vgg19', 'resnet50', 'inception1', 'mobilenet1']
    o["width-multiplier"] = ['1.0'] # ['1.6', '1.3', '1.0', '0.8', '0.5', '0.25', '0.1', '0.05']
    o["template"] = ['base'] #['reverse-base', 'uniform', 'quadratic', 'negative-quadratic', 'base']

    o["lr"] = ['0.1']
    o["epochs"] = ['160'] #160
    o["train-batch"] = ['128']
    o["test-batch"] = ['16']

    o["path-data"] = ['/mnt/storage/scratch/ri16164/datasets']    ##  /mnt/storage/scratch/ri16164/datasets   /media/hal/DATA/Datasets 

    PATH_ELEMENTS = ['./logs/', 'dataset', '/', 'arch', '_', 'template', '_', 'width-multiplier', '_', 'seed'] #, '_', 'train-split']
    o["save"] = ['']
    o["resume"] = [''] # comment for disabling resume

    #o[""] = ['']

    return o




def imagenet():   # Change parameters only in this section
    global PATH_ELEMENTS
    global LAST_VALID_TASK
    o = {}

    LAST_VALID_TASK = 0 # Number of the last task that finished correctly to skip previous. 0: no task performed before

    o["distributed"] = ['']   # only flag with no options
    o["nodes"] = ['8']
    ###o["node-rank"] = ['0']    ####   from BC
    ###o["master-ip"] = ['localhost']    ####   from BC

    o["master-port"] = [str(random.randint(8888, 8999))] # ['8888']
    o["amp-level"] = ['O0']

    o["gpus"] = ['0']    ####   in BCp4 is automatically assigned to env variable
    o["seed"] = ['1'] #['1','5','7']
    o["summary-file"] = ['./logs/templates_imagenet.txt']

    o["dataset"] = ['imagenet2012'] #['imagenet2012']
    o["train-split"] = ['0.0']

    o["arch"] = ['vgg19']  #['vgg19', 'resnet50', 'inception1', 'mobilenet1']
    o["width-multiplier"] = ['1.0'] # ['1.6', '1.3', '1.0', '0.8', '0.5', '0.25', '0.1', '0.05']
    o["template"] = ['quadratic'] #['reverse-base', 'uniform', 'quadratic', 'negative-quadratic', 'base']

    o["lr"] = ['0.01'] # 0.01
    o["epochs"] = ['90']  #90
    o["train-batch"] = ['16'] # base 64    reverse 16    quadratic 16     uniform 32   negative 32
    o["test-batch"] = ['8']

    o["path-data"] = ['/mnt/storage/scratch/ri16164/datasets']    ##  /mnt/storage/scratch/ri16164/datasets   /media/hal/DATA/Datasets 

    PATH_ELEMENTS = ['./logs/', 'dataset', '/', 'arch', '_', 'template', '_', 'width-multiplier', '_', 'seed'] #, '_', 'train-split']
    o["save"] = ['']
    o["resume"] = [''] # comment for disabling resume

    #o[""] = ['']

    return o




def tiny_imagenet():   # Change parameters only in this section
    global PATH_ELEMENTS
    global LAST_VALID_TASK
    o = {}

    LAST_VALID_TASK = 0 # Number of the last task that finished correctly to skip previous. 0: no task performed before

    o["distributed"] = ['']   # only flag with no options
    o["nodes"] = ['1']
    ###o["node-rank"] = ['0']    ####   from BC
    ###o["master-ip"] = ['localhost']    ####   from BC
    o["master-port"] = [str(random.randint(8888, 8999))] # ['8888']
    o["amp-level"] = ['O0']

    o["gpus"] = ['1']    ####   in BCp4 is automatically assigned to env variable
    o["seed"] = ['7'] #['1','5','7']
    o["summary-file"] = ['./logs/templates_tiny-imagenet.txt']

    o["dataset"] = ['tiny-imagenet'] #['tiny-imagenet']
    o["train-split"] = ['0.0']

    o["arch"] = ['vgg19']  #['vgg19', 'resnet50', 'inception1', 'mobilenet1']
    o["width-multiplier"] = ['1.0'] # ['1.6', '1.3', '1.0', '0.8', '0.5', '0.25', '0.1', '0.05']
    o["template"] = ['reverse-base', 'uniform', 'quadratic', 'negative-quadratic', 'base']

    o["lr"] = ['0.01'] # 0.01    left 0.1(91.3)   right 0.01(95.1)
    o["epochs"] = ['90']  #90
    o["train-batch"] = ['128']  # base 256    reverse X16    quadratic X16     uniform 128   negative X32
    o["test-batch"] = ['16']

    o["path-data"] = ['/media/hal/DATA/Datasets']    ##  /mnt/storage/scratch/ri16164/datasets   /media/hal/DATA/Datasets 

    PATH_ELEMENTS = ['./logs/', 'dataset', '/', 'arch', '_', 'template', '_', 'width-multiplier', '_', 'seed'] #, '_', 'train-split']
    o["save"] = ['']
    o["resume"] = [''] # comment for disabling resume

    #o[""] = ['']

    return o












def test():   # Change parameters only in this section
    global PATH_ELEMENTS
    global LAST_VALID_TASK
    o = {}

    LAST_VALID_TASK = 0 # Number of the last task that finished correctly to skip previous. 0: no task performed before

    o["distributed"] = ['']   # only flag with no options
    o["nodes"] = ['1']
    ###o["node-rank"] = ['0']    ####   from BC
    ###o["master-ip"] = ['localhost']    ####   from BC
    o["master-port"] = ['8999']
    o["amp-level"] = ['O0']

    o["gpus"] = ['0']    ####   in BCp4 always the first gpu is 0 even when physically can be different
    o["seed"] = ['1'] #['1','5','7']
    o["summary-file"] = ['./logs/templates_DELETE.txt']

    o["dataset"] = ['mnist'] #['mnist', 'fashionmnist', 'arabic-mnist', 'tiny-imagenet']
    o["train-split"] = ['0.0']

    o["arch"] = ['vgg19']  #['vgg19', 'resnet50', 'mobilenet1']
    o["width-multiplier"] = ['1.0'] # ['1.6', '1.3', '1.0', '0.8', '0.5', '0.25', '0.1', '0.05']
    o["template"] = ['base'] #['reverse-base', 'uniform', 'quadratic', 'negative-quadratic', 'base']

    o["lr"] = ['0.1']
    o["epochs"] = ['4']
    o["train-batch"] = ['128']
    o["test-batch"] = ['16']

    o["path-data"] = ['/media/hal/DATA/Datasets']    ##  /mnt/storage/scratch/ri16164/datasets   /media/hal/DATA/Datasets 

    PATH_ELEMENTS = ['./logs/DELETE/', 'dataset', '/', 'arch', '_', 'template', '_', 'width-multiplier', '_', 'seed'] #, '_', 'train-split']
    o["save"] = ['']
    o["resume"] = [''] # comment for disabling resume

    #o[""] = ['']

    return o
















def transfer_learning_mnist_arabic():   # Change parameters only in this section
    global PATH_ELEMENTS
    global LAST_VALID_TASK
    o = {}

    LAST_VALID_TASK = 0 # Number of the last task that finished correctly to skip previous. 0: no task performed before

    o["gpu-id"] = ['0'] # 0 1
    o["seed"] = ['7'] #['1','5','7']
    o["summary-file"] = ['templates_mnist.txt']

    o["dataset"] = ['mnist', 'arabic-mnist'] #['cifar10', 'cifar100', 'mnist', 'arabic-mnist']
    o["train-split"] = ['0.0']

    o["version"] = ['Reverse-Base', 'Uniform', 'Quadratic', 'Negative-Quadratic', 'Base']
    o["arch"] = ['vgg 19', 'resnet 50', 'inception 1', 'mobilenet 1']
    o["depth"] = [''] # just to keep arch and depth together when they are split

    o["width-multiplier"] = ['1.0', '0.1', '0.08', '0.05', '0.01']

    o["lr"] = ['0.01'] # 0.01
    o["epochs"] = ['20'] # 160
    o["train-batch"] = ['128']
    o["test-batch"] = ['16']


    PATH_ELEMENTS = ['/media/hal/DATA/Ramon/rethinking/', 'dataset', '/', 'arch', 'depth', '_', 'version', '_', 'width-multiplier', '_', 'seed', '_', 'train-split']
    o["save"] = ['']
    o["resume"] = [''] # comment for disabling resume

    #o[""] = ['']

    return o



def tiny_imagenet_training():   # Change parameters only in this section
    global PATH_ELEMENTS
    global LAST_VALID_TASK
    o = {}

    LAST_VALID_TASK = 0 # Number of the last task that finished correctly to skip previous. 0: no task performed before

    #o["gpu-id"] = ['0'] # 0 1     # commented when gpu_id is added from slurm command in run_in_blue_crystal_phase_4
    o["seed"] = ['5'] # ['1','5','7']
    o["summary-file"] = ['tiny_imagenet_bcp4.txt']

    o["dataset"] = ['tiny-imagenet']
    o["train-split"] = ['0.0']

    o["version"] = ['Quadratic'] #['Reverse-Base', 'Uniform', 'Quadratic', 'Negative-Quadratic', 'Base']
    o["arch"] = ['resnet 50'] #['vgg 19', 'resnet 50', 'inception 1', 'mobilenet 1']
    o["depth"] = [''] # just to keep arch and depth together when they are split

    o["width-multiplier"] = ['1.0']

    o["epochs"] = ['90'] # 90
    o["train-batch"] = ['16']
    o["test-batch"] = ['8']


    PATH_ELEMENTS = ['./logs/', 'dataset', '/', 'arch', 'depth', '_', 'template', '_', 'width-multiplier', '_', 'seed', '_', 'train-split']
    o["save"] = ['']
    o["resume"] = [''] # comment for disabling resume

    o["path-data"] = ['/mnt/storage/scratch/ri16164/datasets']

    #o[""] = ['']

    return o





def tiny_imagenet_finding_learning_rate():   # Change parameters only in this section
    global PATH_ELEMENTS
    global LAST_VALID_TASK
    o = {}

    LAST_VALID_TASK = 9 # Number of the last task that finished correctly to skip previous. 0: no task performed before

    o["gpu-id"] = ['1'] # 0 1
    o["seed"] = ['5'] # ['1','5','7']
    o["summary-file"] = ['tiny_imagenet_bcp4_find_lr.txt']

    o["dataset"] = ['tiny-imagenet']
    o["train-split"] = ['0.0']

    o["version"] = ['Uniform', 'Negative-Quadratic', 'Base'] # ['Reverse-Base', 'Uniform', 'Quadratic', 'Negative-Quadratic', 'Base']
    o["arch"] = ['vgg 19', 'mobilenet 1'] #['vgg 19', 'resnet 50', 'inception 1', 'mobilenet 1']
    o["depth"] = [''] # just to keep arch and depth together when they are split

    o["width-multiplier"] = ['1.0']

    o["epochs"] = ['120'] # 90
    o["lr"] = ['0.5', '0.1', '0.05']
    o["train-batch"] = ['64']
    o["test-batch"] = ['64']


    PATH_ELEMENTS = ['./logs/lr_exploration/', 'dataset', '/', 'arch', 'depth', '_', 'version', '_', 'width-multiplier', '_', 'seed', '_', 'train-split']
    o["save"] = ['']
    o["resume"] = [''] # comment for disabling resume

    #o["path-data"] = ['/mnt/storage/scratch/ri16164/datasets']

    #o[""] = ['']

    return o





def ms_coco_training():   # Change parameters only in this section
    global PATH_ELEMENTS
    global LAST_VALID_TASK
    o = {}

    LAST_VALID_TASK = 0 # Number of the last task that finished correctly to skip previous. 0: no task performed before

    o["gpu-id"] = ['1'] # 0 1     # commented when gpu_id is added from slurm command in run_in_blue_crystal_phase_4
    o["seed"] = ['1'] # ['1','5','7']
    o["summary-file"] = ['ms_coco.txt']

    o["dataset"] = ['mscoco2017']
    o["train-split"] = ['0.0']

    o["version"] = ['Negative-Quadratic'] #['Reverse-Base', 'Uniform', 'Quadratic', 'Negative-Quadratic', 'Base']
    o["arch"] = ['vgg 19'] #['vgg 19', 'resnet 50', 'inception 1', 'mobilenet 1']
    o["depth"] = [''] # just to keep arch and depth together when they are split

    o["width-multiplier"] = ['0.5']

    o["epochs"] = ['50'] # 26
    o["train-batch"] = ['1']
    o["test-batch"] = ['1']


    PATH_ELEMENTS = ['./logs/', 'dataset', '/', 'arch', 'depth', '_', 'version', '_', 'width-multiplier', '_', 'seed', '_', 'train-split']
    o["save"] = ['']
    o["resume"] = [''] # comment for disabling resume

    #   /mnt/storage/scratch/ri16164/datasets    BCp4
    #   /media/hal/DATA/Datasets/
    #   /media/deepthought/SCRATCH
    #   /media/viki/SCRATCH/
    #   /media/alexa/SCRATCH/
    #   /media/eve/SCRATCH/
    o["path-data"] = ['/media/alexa/SCRATCH']

    #o[""] = ['']

    return o


'''

viki 0 vgg Base
viki 1 vgg Negative-Quadratic
deepthought 0 mobilenet Base
deepthought 1 vgg Uniform
alexa 0 mobilenet Negative-Quadratic   0.5
#alexa 1 mobilenet Uniform
alexa 1 vgg Negative-Quadratic 0.5
eve 1 mobilenet Base 0.5
eve 0 vgg Base 0.5




unset HISTFILE
cd /home/hal/Ramon/filter-templates/mscoco/
conda activate ramon-pytorch


ls /media/alexa/SCRATCH
ls /media/eve/SCRATCH
lsblk
cp -r /media/hal/DATA/Datasets/mscoco2017 /media/alexa/SCRATCH/
cp -r /media/hal/DATA/Datasets/mscoco2017 /media/eve/SCRATCH/

'''





def parse_arguments():
    parser = argparse.ArgumentParser(description='Multiple running model')
    parser.add_argument('--slurm', action='store_true', default=False,
                        help='enables running in SLURM job schedule')

    parser.add_argument('--config', type=str.lower, required=True,
                        choices=['cifar', 'xnist', 'tinyimagenet', 'imagenet', 'test'],
                        help='training dataset')
    return parser.parse_args()


def select_config(config_mode):

    if config_mode == 'cifar': config = cifar()
    if config_mode == 'mnist': config = xnist()
    if config_mode == 'tinyimagenet': config = tiny_imagenet()
    if config_mode == 'imagenet': config = imagenet()

    if config_mode == 'test': config = test()

    #global_options = user_defined_options()   # default
    #global_options = transfer_learning_mnist_arabic()
    #global_options = cifar_x_nist()
    #global_options = tiny_imagenet_training()
    #global_options = tiny_imagenet_finding_learning_rate()
    #global_options = ms_coco_training()
    return config




if __name__ == '__main__':

    random.seed(datetime.now())

    args = parse_arguments()
    total_initial_time = time.time()
    global_options = select_config(args.config)

    task_index = 1
    completed_task = 0
    broken_tasks = []

    combinations_of_tasks = product(*(global_options.values()))
    max_tasks = len(tuple(combinations_of_tasks))


    for single_options in product(*(global_options.values())):
        arguments_dictionary = dict( zip( global_options.keys(), list(single_options) ) )
        arguments = process_arguments(arguments_dictionary)

        #tested_combination = [0, 7, 'Negative-Quadratic', 'cifar10']
        #if all(i == j for i, j in zip(single_options, tested_combination)):
        #    print('THIS', list(single_options))

        print('Performing process', task_index, '/', max_tasks, 'with arguments:', arguments)

        if task_index <= LAST_VALID_TASK:
            print('Skipping already completed task...')
        else:
            exit_code = 0 # just for not failing when commenting runs
            #if task_index not in [1,5,6,8,9]:
            if args.slurm:
                exit_code = run_in_blue_crystal_phase_4('training_scratch.py', arguments, job_name='dl-series')
            else:
                exit_code = run_in_command_line('training_scratch.py', arguments)

            print('The process terminated with code', exit_code)
            if exit_code==0:
                completed_task = completed_task + 1
            else:
                broken_tasks.append([task_index] + arguments)

        task_index += 1
        if task_index>50: break  # execute only 1 job

    print('Completed', completed_task, 'from', max_tasks, '. Skipped', LAST_VALID_TASK, 'tasks.') 
    print('Failling tasks:')
    print(broken_tasks)
    total_time = time.time() - total_initial_time
    print('Total time for the whole training/scheduling process: %.8f sec' % (total_time))
    print('Processes finished')

