# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

# This script is adapted from the original HuggingFace Transformers functionality
# found at https://github.com/huggingface/transformers/blob/v2.11.0/examples/question-answering/run_squad.py

import os
import glob
import numpy as np

from qa.utils import *
from qa.data.loader import load_and_cache_examples
from qa.model_utils import load_pretrained_model, evaluate_model


def main(args):

    args.device = initialize_device(args)

    logger = initialize_logging(args, module="Eval")

    # Set seed
    set_seed(args.seed)

    # Evaluation
    results = {}
    if args.local_rank in [-1, 0]:
        logger.info("Loading %s for evaluation", args.model_name_or_path)
        checkpoints = [args.model_name_or_path]

        if args.eval_all_checkpoints:  # Or evaluate on all checkpoints if asked
            try:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(
                        glob.glob(
                            args.model_name_or_path + "/**/pytorch_model.bin",
                            recursive=True,
                        )
                    )
                )
            except:
                logger.info("No checkpoints found in " + args.model_name_or_path)

        logging.getLogger("transformers.modeling_utils").setLevel(
            logging.WARN
        )  # Reduce model loading logs
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            args.model_name_or_path = checkpoint

            model, tokenizer = load_pretrained_model(args)
            model.to(args.device)

            # Evaluate
            result = evaluate_model(args, model, tokenizer, prefix=global_step)

            result = dict(
                (k + ("_{}".format(global_step) if global_step else ""), v)
                for k, v in result.items()
            )
            results.update(result)

        logger.info("Results: {}".format(results))

    return result


if __name__ == "__main__":
    from qa.arguments import args

    main(args)

