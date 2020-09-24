s# Question Answering
This repo is complimentary to that developed during FF14, and consists of modified FF14 code so as to make it more user friendly for the Cloudera CDSW/CML platforms. 

Datasets: 
* SQuAD2.0 (LINK)
* COVID-QA (LINK)
* BioASQ (Registration required) (LINK) 

## Things you can do in this CML Prototype
* train a QA model on domain specific / specialized training data using the Jobs abstraction
* evaluate a QA model on a validation set
* expose and serve a QA model using the Model abstraction (i.e. for use in an MLViz application)
* explore training and evaluation data: including the required JSON schema, types of questions, lengths of passages, and more
* explore performance of several popular pre-trained QA models (BERT, RoBERTa, DistilBERT, MiniLM, XLM-RoBERTa)
* examine the training process through a TensorBoard application

## Set up
-- install requirements.txt (once I make one)
-- add qa package to requirements.txt


## Apps
* MLViz (broken) I had an example of how QAModel could connect to 
* tensorboard (available if a training session has been executed)
* QA Data Visualizer
* QA Model Explorer


## How to do stuff
This repo has 

Examples of how to do things


### Train on a dataset
Specify the training set in the config file

`!python3 scripts/train.py @scripts/config.txt`

### Train on many datasets (as in a Job)
Setting the `--train-file` flag after the config file overrides the config file. 
This is useful if you want to programmatically train on many datasets in succession. 

`!python3 scripts/train.py @scripts/config.txt --train-file train_file_name.json`

This overriding-the-config-behavior works for any of the config parameters. 
