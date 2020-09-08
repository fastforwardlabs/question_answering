# Here we'll put code to evaluate on a dev set 

import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange

from transformers import (
  WEIGHTS_NAME,
  AutoModelForQuestionAnswering,
  AutoTokenizer,
  squad_convert_examples_to_features
)
from transformers.data.metrics.squad_metrics import (
  compute_predictions_log_probs,
  compute_predictions_logits,
  squad_evaluate,
)
from transformers.data.processors.squad import SquadResult

from utils import * 


def evaluate_model(args, model, tokenizer, prefix=""):
  dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

  if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(args.output_dir)

  args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

  eval_sampler = SequentialSampler(dataset)
  eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

  # multi-gpu evaluate
  if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
      model = torch.nn.DataParallel(model)

  # Eval!
  logger.info("***** Running evaluation {} *****".format(prefix))
  logger.info("  Num examples = %d", len(dataset))
  logger.info("  Batch size = %d", args.eval_batch_size)

  all_results = []
  start_time = timeit.default_timer()

  for batch in tqdm(eval_dataloader, desc="Evaluating"):
      model.eval()
      batch = tuple(t.to(args.device) for t in batch)

      with torch.no_grad():
          inputs = {
              "input_ids": batch[0],
              "attention_mask": batch[1],
              "token_type_ids": batch[2],
          }

          if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
              del inputs["token_type_ids"]

          feature_indices = batch[3]

          # XLNet and XLM use more arguments for their predictions
          if args.model_type in ["xlnet", "xlm"]:
              inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
              # for lang_id-sensitive xlm models
              if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                  inputs.update(
                      {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                  )
          outputs = model(**inputs)

      for i, feature_index in enumerate(feature_indices):
          eval_feature = features[feature_index.item()]
          unique_id = int(eval_feature.unique_id)

          output = [to_list(output[i]) for output in outputs]

          # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
          # models only use two.
          if len(output) >= 5:
              start_logits = output[0]
              start_top_index = output[1]
              end_logits = output[2]
              end_top_index = output[3]
              cls_logits = output[4]

              result = SquadResult(
                  unique_id,
                  start_logits,
                  end_logits,
                  start_top_index=start_top_index,
                  end_top_index=end_top_index,
                  cls_logits=cls_logits,
              )

          else:
              start_logits, end_logits = output
              result = SquadResult(unique_id, start_logits, end_logits)

          all_results.append(result)

  evalTime = timeit.default_timer() - start_time
  logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

  # Compute predictions
  output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
  output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

  if args.version_2_with_negative:
      output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
  else:
      output_null_log_odds_file = None

  # XLNet and XLM use a more complex post-processing procedure
  if args.model_type in ["xlnet", "xlm"]:
      start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
      end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

      predictions = compute_predictions_log_probs(
          examples,
          features,
          all_results,
          args.n_best_size,
          args.max_answer_length,
          output_prediction_file,
          output_nbest_file,
          output_null_log_odds_file,
          start_n_top,
          end_n_top,
          args.version_2_with_negative,
          tokenizer,
          args.verbose_logging,
      )
  else:
      predictions = compute_predictions_logits(
          examples,
          features,
          all_results,
          args.n_best_size,
          args.max_answer_length,
          args.do_lower_case,
          output_prediction_file,
          output_nbest_file,
          output_null_log_odds_file,
          args.verbose_logging,
          args.version_2_with_negative,
          args.null_score_diff_threshold,
          tokenizer,
      )

  # Compute the F1 and exact scores.
  results = squad_evaluate(examples, predictions)
  return results


def main(args):  
  print("in main ... ")
  print(args)
  return

  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
      device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
      args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
  else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
      torch.cuda.set_device(args.local_rank)
      device = torch.device("cuda", args.local_rank)
      torch.distributed.init_process_group(backend="nccl")
      args.n_gpu = 1
  args.device = device

  # Setup logging
  logging.basicConfig(
      format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
      datefmt="%m/%d/%Y %H:%M:%S",
      level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
  )
  logger.warning(
      "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
      args.local_rank,
      device,
      args.n_gpu,
      bool(args.local_rank != -1),
      args.fp16,
  )

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
      model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  
      tokenizer = AutoTokenizer.from_pretrained(checkpoint, do_lower_case=args.do_lower_case)
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