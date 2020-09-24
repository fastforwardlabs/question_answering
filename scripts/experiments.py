
# This script is designed to be run using the Experiments or Jobs CDSW abstraction.
# It performs the steps necessary to conduct an experiment in which a QA model
# is trained on increasing-sized data from a specialized domain. [LINK TO BLOG POST]
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

# TODO: Make a config.txt specifically for this experiment

from qa.data.utils import create_increasing_sized_train_sets

original_dataset = "/Users/mbeck/Projects/ff14/data/medical/covid_bioasq_train.json"
create_increasing_sized_train_sets(original_dataset)

for num in [500, 1000, 1500, 2000, 2500, 3000]:
    !python scripts/train.py @scripts/config.txt \
        --train_file covid_bioasq_train_{num}.json \
        --output_dir models/deepset-minilm-uncased-squad2/med{num}/

    !python scripts/evaluate.py @scripts/config.txt \
        --model_name_or_path models/deepset-minilm-uncased-squad2/med{num}/
        --output_dir data/predictions/deepset-minilm-uncased-squad2/med{num}/

