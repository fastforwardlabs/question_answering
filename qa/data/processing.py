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

# The functions in this file are adapted from the original HuggingFace functions
# that can be found here 
# They have been gently modified to allow more generic json structures

import os
import json
from tqdm import tqdm

# The following function is adapted from the original HuggingFace functionality
# found at https://github.com/huggingface/transformers/blob/v2.11.0/src/transformers/data/processors/squad.py
def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

# The following class is adapted from the original HuggingFace class found in 
# https://github.com/huggingface/transformers/blob/v2.11.0/src/transformers/data/processors/squad.py
class SquadLikeProcessor:
    """
    Processor for the Question Answering data sets with SQuAD-like structure.
    """

    def get_train_examples(self, data_dir, train_file):
        """
        Returns the training examples from the data directory.
        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            train_file: Required json file containing train set
        """
        if data_dir is None:
            data_dir = ""

        filename = os.path.join(data_dir, train_file)
        if not os.path.exists(filename):
            raise ValueError(
                f"{filename} does not exist -- please provide valid train_file!"
            )

        with open(filename, "r", encoding="utf-8") as reader:
            input_data = json.load(reader)["data"]

        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, dev_file):
        """
        Returns the evaluation example from the data directory.
        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            dev_file: Required json file containing dev set
        """
        if data_dir is None:
            data_dir = ""

        filename = os.path.join(data_dir, dev_file)
        if not os.path.exists(filename):
            raise ValueError(
                f"{filename} does not exist -- please provide valid dev_file!"
            )

        with open(filename, "r", encoding="utf-8") as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]

                # Some data structures have document-level identifiers.
                # Check for two of the most common ("title" --> SQuAD)
                if ("document_id" or "title") in paragraph:
                    try:
                        document_id = paragraph["document_id"]
                    except:
                        document_id = paragraph["title"]
                else:
                    document_id = None

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    is_impossible = qa.get("is_impossible", False)
                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = SquadLikeExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        document_id=document_id,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        is_impossible=is_impossible,
                        answers=answers,
                    )
                    examples.append(example)
        return examples

# The following class is adapted from the original HuggingFace class found in 
# https://github.com/huggingface/transformers/blob/v2.11.0/src/transformers/data/processors/squad.py
class SquadLikeExample:
    """
    A single training/test example for the Squad dataset, as loaded from disk.
    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        document_id,
        answer_text,
        start_position_character,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.document_id = document_id
        self.answer_text = answer_text
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start and end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(
                    start_position_character + len(answer_text) - 1,
                    len(char_to_word_offset) - 1,
                )
            ]
