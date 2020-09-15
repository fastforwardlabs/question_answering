# Question Answering
This repo is complimentary to the one(s) created during the FF1 project. This repo modifies the code in those repositories to make them more user friendly for use on the Cloudera CDSW/CML platforms


## Set up
-- install requirements.txt (once I make one)
-- add qa package to requirements.txt


## Weird stuff
* training / eval on CDSW yields drastically different results than when trained in Colab notebook
We're talking 10% difference in model performance between the two. Need to dig into this. 


## How to do stuff

### Train on a dataset
Specify the training set in the config file
> !python3 scripts/train.py @scripts/config.txt

### Train on many datasets (as in a Job)
Setting the `--train-file` flag after the config file overrides the config file. This is useful if you want to programmatically train on many datasets in succession. 
> !python3 scripts/train.py @scripts/config.txt --train-file train_file_name.json

This overriding-the-config-behavior works for any of the config parameters. 
