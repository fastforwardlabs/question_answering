import os
import logging

import torch

from transformers.data.processors.squad import (
    SquadV1Processor, 
    SquadV2Processor, 
    squad_convert_examples_to_features
)

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
  
    logger = logging.getLogger('Eval.data_loader') if evaluate else logging.getLogger('Train.data_loader')
    
    if args.local_rank not in [-1, 0] and not evaluate:
      # Make sure only the first process in distributed training process the dataset, and the others will use the cache
      torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
      input_dir,
      "cached_{}_{}_{}".format(
          "dev" if evaluate else "train",
          list(filter(None, args.model_name_or_path.split("/"))).pop(),
          str(args.max_seq_length),
      ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
          features_and_dataset["features"],
          features_and_dataset["dataset"],
          features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If data_dir is not specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        features, dataset = squad_convert_examples_to_features(
          examples=examples,
          tokenizer=tokenizer,
          max_seq_length=args.max_seq_length,
          doc_stride=args.doc_stride,
          max_query_length=args.max_query_length,
          is_training=not evaluate,
          return_dataset="pt",
          threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
      # Make sure only the first process in distributed training process the dataset, and the others will use the cache
      torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features

    return dataset