# 
#
#
#
#
#
#

import os
import torch

from qa.utils import *
from qa.data.loader import load_and_cache_examples
from qa.model_utils import load_pretrained_model, train_model
    
def main(args):
    
    # Set up compute device (CPU/GPU/Distributed training)
    args.device = initialize_device(args)

    # Set up logging
    logger = initialize_logging(args, module='Train')

    # Set seed
    set_seed(args)

    # Make sure only the first process in distributed training will download model & vocab
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()
        
    model, tokenizer = load_pretrained_model(args)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()
    
    # send model to compute device
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
    
    global_step, tr_loss = train_model(args, train_dataset, model, tokenizer)
    
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        logger.info("Saving model checkpoint to %s", args.output_dir)
        
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
            
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
  
    logger.info("Training complete.")
    

if __name__ == "__main__":
    model_name = "twmkn9/distilbert-base-uncased-squad2"

    from qa.arguments import args 
    
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )
    main(args)

