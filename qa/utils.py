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

import logging
import os
import pathlib
import random

import numpy as np
import torch

# This function is adapted from the original HuggingFace Transformers functionality
# found at https://github.com/huggingface/transformers/blob/v2.11.0/examples/question-answering/run_squad.py
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    #    torch.cuda.manual_seed_all(seed)

# This function is adapted from the original HuggingFace Transformers functionality
# found at https://github.com/huggingface/transformers/blob/v2.11.0/examples/question-answering/run_squad.py
def to_list(tensor):
    return tensor.detach().cpu().tolist()

# This function is adapted from the original HuggingFace Transformers functionality
# found at https://github.com/huggingface/transformers/blob/v2.11.0/examples/question-answering/run_squad.py
def initialize_device(args):
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

        if (
            args.no_cuda
        ):  # if working without GPUs, maximize CPU computation (single machine)
            torch.set_num_threads(args.threads)

    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1

    return device

# This function is adapted from the original HuggingFace Transformers functionality
# found at https://github.com/huggingface/transformers/blob/v2.11.0/examples/question-answering/run_squad.py
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


def create_path(pathname):
    """Creates the directory for the given path if it doesn't already exist."""
    dir = str(pathlib.Path(pathname).parent)
    if not os.path.exists(dir):
        os.makedirs(dir)


def absolute_path(*paths):
    """Given a path relative to this project's top-level directory, returns the
    full path in the OS.
    Args:
        paths: A list of folders/files.  These will be joined in order with "/"
            or "\" depending on platform.
    Returns:
        The full absolute path in the OS.
    """
    # First parent gets the scripts directory, and the second gets the top-level.
    result_path = pathlib.Path(__file__).resolve().parent.parent
    for path in paths:
        result_path /= path
    return str(result_path)
