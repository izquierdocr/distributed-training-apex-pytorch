import torch

from torchsummary import summary
from flops_counter import get_model_complexity_info


def get_model_metrics(model, args, loggers):

    local_model = model
    if isinstance(model, torch.nn.DataParallel):
        local_model = model.module

    input_size = (args.input_channels,) + args.input_size

    summary_info = summary(local_model, input_size, device=args.gpu_id, batch_size=-1)  #  args.train_batch)
    loggers['model_parameters'] = summary_info[0]
    loggers['model_trainable_parameters'] = summary_info[1]
    # next metrics given in MB. Transform to B
    loggers['model_input_size'] = summary_info[2] * (1024**2)
    loggers['model_passes_size'] = summary_info[3] * (1024**2)
    loggers['model_parameters_size'] = summary_info[4] * (1024**2)
    loggers['model_memory_computed'] = summary_info[5] * (1024**2)
    loggers['model_memory_cuda'], _ = get_memory_cuda()

    with torch.cuda.device(args.gpu_id):
        flops, params = get_model_complexity_info(local_model, input_size, as_strings=False, print_per_layer_stat=False)
    loggers['model_flops'] = flops


def get_memory_cuda():
    gpu_memory_cached = torch.cuda.max_memory_cached()
    gpu_memory_allocated = torch.cuda.max_memory_allocated()
    return gpu_memory_cached, gpu_memory_allocated


