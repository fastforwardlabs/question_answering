# Here we'll put code to evaluate on a dev set 

import os
import numpy as np

from qa.utils import * 
from qa.data.loader import load_and_cache_examples
from qa.model_utils import load_pretrained_model, evaluate_model


def main(args):  

  args.device = initialize_device(args)

  logger = initialize_logging(args, module='Eval')

  # Set seed
  set_seed(args)
  
  # Evaluation
  results = {}
  if args.local_rank in [-1, 0]:
    # Check output_dir for recently trained model checkpoint
    if os.path.exists(args.output_dir):
      logger.info("Loading checkpoints saved during training for evaluation")
      checkpoints = [args.output_dir]
      
      if args.eval_all_checkpoints: # Or evaluate on all checkpoints if asked
        checkpoints = list(
            os.path.dirname(c)
            for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
    # If no recently trained model exists, default to model_name_or_path
    else:
      logger.info("Loading %s for evaluation", args.model_name_or_path)
      checkpoints = [args.model_name_or_path]

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
      global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
      args.model_name_or_path = checkpoint
    
      model, tokenizer = load_pretrained_model(args)
      model.to(args.device)

      # Evaluate
      result = evaluate_model(args, model, tokenizer, prefix=global_step)

      result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
      results.update(result)

  logger.info("Results: {}".format(results))
  
  return result
  
  
if __name__ == "__main__":
  from arguments import args
  main(args)