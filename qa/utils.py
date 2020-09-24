import logging
import os 
import random
import numpy as np

import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #if args.n_gpu > 0:
    #    torch.cuda.manual_seed_all(seed)
        
def to_list(tensor):
    return tensor.detach().cpu().tolist()
  
def initialize_device(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
        
        if args.no_cuda:  # if working without GPUs, maximize CPU computation (single machine)
            torch.set_num_threads(args.threads) 
            
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
        
    return device

def initialize_logging(args, module):
    logger = logging.getLogger(module)

    logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
      "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
      args.local_rank,
      args.device,
      args.n_gpu,
      bool(args.local_rank != -1),
      args.fp16,
    )
    
    return logger

  
  
