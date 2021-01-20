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


# This script is designed to be run using the Jobs abstraction and trains a 
# QA model on increasing-sized data sets from a specialized domain (medical data).
# This procedure and the results were discussed in the accompanying blog post:  
# https://qa.fastforwardlabs.com/domain%20adaptation/transfer%20learning/specialized%20datasets/qa/medical%20qa/2020/07/22/QA-for-Specialized-Data.html
# 
# It first creates subsets of the original training dataset. These subsets contain 
# an increasing number of examples -- i.e., the first subset has only 500 examples, 
# the second has 1000 (the 500 from the previous subset plus 500 additional examples), 
# and so on. Once these subsets are created, the training and evaluation scripts are 
# executed for each subset. 
#
# NOTE: it's important that your config.txt contains the correct data directory
#       for the subsets
#
# This training script trains a **new** model on each subset and saves that model 
# to an output directory (designated in config.txt) resulting in six models, each
# trained on increasing dataset sizes. 
#
# The predictions and results outputs are saved to the output_dir designated 
# in the config.txt

from qa.data.utils import create_increasing_sized_train_sets

# These sizes were used in our blog post to train a model on increasing numbers
# of training data for the the medical dataset
DEFAULT_SIZES = [500, 1000, 1500, 2000, 2500, 3000]

# These sizes are for testing purposes - no real training set should contain only 5 examples!
MINI_SIZES = [5,10,15]

original_dataset = "/home/cdsw/data/medical/covid_bioasq_train.json"
create_increasing_sized_train_sets(original_dataset, sizes=MINI_SIZES)

for num in MINI_SIZES:
  !python3 scripts/train.py @scripts/config_multi-train.txt \
      --train_file covid_bioasq_train_{num}.json \
      --output_dir models/deepset-bert-base-cased-squad2/medical_{num}/
  
  !python3 scripts/evaluate.py @scripts/config_multi-train.txt \
      --model_name_or_path models/deepset-bert-base-cased-squad2/medical_{num}/ \
      --output_dir data/predictions/deepset-bert-base-cased-squad2/medical_{num}/ \
      --predict_file covid_bioasq_train_{num}.json