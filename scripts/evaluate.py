# Here we'll put code to evaluate on a dev set 

import os
import glob
import numpy as np

from qa.utils import * 
from qa.data.loader import load_and_cache_examples
from qa.model_utils import load_pretrained_model, evaluate_model

def main(args):  

  args.device = initialize_device(args)

  logger = initialize_logging(args, module='Eval')

  # Set seed
  set_seed(args.seed)
  
  # Evaluation
  results = {}
  if args.local_rank in [-1, 0]:
      logger.info("Loading %s for evaluation", args.model_name_or_path)
      checkpoints = [args.model_name_or_path]

      if args.eval_all_checkpoints: # Or evaluate on all checkpoints if asked
          try:
              checkpoints = list(
                  os.path.dirname(c)
                  for c in sorted(glob.glob(args.model_name_or_path + "/**/pytorch_model.bin", recursive=True))
              )
          except:
              logger.info("No checkpoints found in "+args.model_name_or_path)

      logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
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
  from qa.arguments import args
  main(args)