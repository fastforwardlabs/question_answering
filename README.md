# Question Answering
This repo is complimentary to that developed during FF14, and consists of modified FF14 code so as to make it 
more user friendly for the Cloudera CDSW/CML platforms. This repo focuses primarily on training and evaluation 
methods for QA Models, including apps that allow you to explore QA data, compare pre-trained QA models, 
and perform basic QA interactions with a trained model. This repo does not include methods for pairing the 
QA Model with a search engine to make a full Information-Retrieval Question Answering Pipeline as discussed in 
our accompanying blog series. For a demonstration of that, see NeuralQA. 

Datasets: 
* SQuAD2.0 (LINK)
* COVID-QA (LINK)
* BioASQ (Registration required) (LINK) 

## Things you can do in this CML Prototype
* train a QA model on domain specific / specialized training data using the Jobs abstraction
* evaluate a QA model on a validation set
* expose and serve a QA model using the Model abstraction (i.e. for use in an MLViz application)
* explore training and evaluation data: including the required JSON schema, types of questions, 
  lengths of passages, and more
* explore performance of several popular models that have already been trained for question answering 
  (BERT, RoBERTa, DistilBERT, MiniLM, XLM-RoBERTa)
* examine the training process through a TensorBoard application

## Set up
-- install requirements.txt (once I make one)
-- add qa package to requirements.txt


## Apps
* MLViz (broken) I had an example of how QAModel could connect to the MLViz app but now it's not working. Sad. 
* tensorboard (available if a training session has been executed)
* QA Data Visualizer
* QA Model Explorer


# How to do stuff
The `scripts` directory is the main entry point and contains scripts designed to be 
executed either in Sessions or in another appropriate CDSW abstraction. 

While most of these scripts are straightforward (e.g. QA Model deployment via the Models abstraction), 
the training and evaluation workflows require a bit more explanation. At their core, these scripts are 
based heavily on a [script](https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_squad.py)
developed by the [HuggingFace](https://huggingface.co/) team for training [Transformer models](http://jalammar.github.io/illustrated-transformer/) 
(like BERT) for the task of [extractive question answering](https://huggingface.co/transformers/v3.2.0/task_summary.html#extractive-question-answering). 
However, this script is cumbersome to use in CDSW/CML, so it has been modified to work in this environment. 
As such, the `train.py` and `evaluate.py` scripts require a configuration file (`config.txt`) as an argument. 
The config file contains required and optional parameters that provide an impressive amount of flexibility 
and functionality. Many of these parameters are geared towards experienced NLP practictioners; most should 
be left as their defaults. 

The first three arguments in `config.txt` are required so we describe them briefly here:
* `--model_type`: Must be a valid model type allowed by HuggingFace (e.g. "bert" or "roberta") 
* `--model_name_or_path`: Can be either of the following: 
  * A string with the _identifier_ name of a pretrained model to load from cache or download from the 
  [HF model repository](https://huggingface.co/models), e.g., "bert-base-uncased" or "deepset/bert-base-cased-squad2"
  * A path to a directory containing model weights, e.g., /home/cdsw/models/my_model_directory/
* `--output_dir`: path of directory to store trained model weights (if training) or predictions and performance results (if evaluating)```


Note: The first time an _identifier_ name is called, those model weights will be downloaded and cached from the HF model repo. 
This can take a long time and a lot of disk space for large models. Additional calls to that particular model name will load 
model weights from the cache.

Note: When training or evaluating models, you must have an Engine Profile with a minimum of 2 vCPU / 16 GiB. 
Any less and the session / job will fail due to memory constraints. A GPU is _strongly_ preferred for these tasks!

Note:  Setting the `--train-file` flag after the config file overrides the config file. 
This is useful if you want to programmatically train on many datasets in succession. 
This overriding-the-config-behavior works for any of the config parameters. 

-------------------------

## Training and evaluating a QA model in a Session

#### To Train
In `config.txt`, first specify the following information: 
* the model you want to train 
* the output directory where trained model weights will be stored
* the directory and filename of the training data 

Then run the following in an open Session:
`!python3 scripts/train.py @scripts/config.txt`

#### To Evaluate 
In `config.txt`, first specify the following information: 
* the model you wish to evaluate 
* the output directory where results and predictions will be stored
* the directory and filename of the validation data

Then run the following in an open Session:
`!python3 scripts/evaluate.py @scripts/config.txt`


## Training and evaluating a QA model as a Job
The `scripts/multi-train.py` script demonstrates how to train and evaluate in succession. 
This script is designed to be run as a Job by selecting this file in the **Script** field of the 
Create a Job menu. 

## Serving a QA Model
Once you have a QA Model that you're satisfied with you can expose it via the Model abstraction. 
Select `scripts/model.py` under the **Script** field in the Create a Model menu. 

-------------------------

## To Do 
- [] provide a bit more functionality to the `models.py` script
- [x] fix the data fetching 
- [] document code
- [x] make the README not bad
- [x] make model_explorer launch script
- [] add the medical data to be cloned with the repo? 
