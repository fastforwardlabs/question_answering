
# This script is designed to be run using the Jobs abstraction.
# This scripts trains a QA model on increasing-sized data from a specialized domain. [LINK TO BLOG POST]
# 
# It first creates subsets of the original dataset. These subsets contain an increasing
# number of examples -- i.e., the first subset has only 500 examples, the second has 
# 1000 (the 500 from the previous subset plus 500 additional examples), and so on.
#
# Once these subsets are created, the training script is executed for each subset. 
# NOTE: it's important that your config.txt contains the correct data directory
#       for the subsets
# This training script trains a model on each subset and saves that model to an
# output directory (designated in config.txt). After completion of the loop, 
# there will be six models trained on increasing dataset sizes. 
#
# Finally, the evaluation script is run for each of the trained models. 
# The prediction and results outputs are saved to the output_dir designated 
# in the config.txt

from qa.data.utils import create_increasing_sized_train_sets

DEFAULT_SIZES = [500, 1000, 1500, 2000, 2500, 3000]
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