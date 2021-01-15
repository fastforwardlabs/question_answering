# Deep Learning for Question Answering
This repo accompanies the Cloudera Fast Forward Question Answering [blog series](https://qa.fastforwardlabs.com/) in which each blog post dives deep into various aspects of question answering systems.  For a high level introduction to general QA systems, including the Information-Retrieval based system included here, check out [this post](https://qa.fastforwardlabs.com/methods/background/2020/04/28/Intro-to-QA.html). 

The primary outputs of this repository are three small applications that allow the user to experience a real-world Question Answering system, as well as interact with the data and models that support such a system. Each is briefly described in turn below. 

### WikiQA
This app is a real-world IR-QA system built on top of Wikipedia's search engine.

<img src="images/Screenshot_WikiQA.png" alt="WikiQA question answering interface">

### Model Explorer
This app allows the user to compare several pre-trained QA Models on various performance measures. 

<img src="images/Screenshot_ModelExplorer.png" alt="=Model Explorer interface" width="40%">

### Data Visualizer
This app lets the user walk through the structure required to train and evaluate Transformer neural networks for the question answring task.

<img src="images/Screenshot_DataVisualizer.png" alt="Data Visualizer interface" width="40%">

Instructions are given both for general use (on a laptop, say), and for Cloudera CML and CDSW.
We'll first describe what's here, then go through how to run everything.

## Structure

The folder structure of the repo is as follows

```
.
├── apps      # Three small Streamlit applications.
├── cml       # Contains scripts that facilitate the project launch on CML.
├── data      # Contains pre-computed evaluation data required for the apps. Starter data will also be downloaded here during project launch on CML.
├── scripts   # This is where all the code that does something lives.
└── qa        # A small library of useful functions.
```

There's also an `images` folder that contains images for this README, which can be ignored.
Let's examine each of the important folders in turn.

### `apps`
The applications accompanying this project come with launcher scripts to assist launching an Application with CDSW/CML. To launch the applications in another environment, run the code inside the launcher file, with the prefixed `!` removed. You may need to specify different ports. 

For example, to launch WikiQA in a CML/CDSW session use 

`!streamlit run apps/wikiqa.py  --server.port $CDSW_APP_PORT --server.address 127.0.0.1`

In a local environment (your laptop, say), you can instead simply call

`streamlit run apps/wikiqa.py`

These three apps are largely, but not entirely, independent of the rest of the content contained in this repo, which was developed for practioners to more easily train and evaluate QA models in CML/CDSW. We briefly cover the structure here but save usage details until the end of this README. 

### `qa`
This is a small Python library designed for practioners who want to train and evaluate Transformer models for question answering tasks. This library is not largely used by any of the three apps mentioned above. 

Its structure is as follows:
```
qa
├── data
│   ├── fetch.py     
│   ├── loader.py
│   ├── processing.py
│   └── utils.py
├── arguments.py
├── model_utils.py
└── utils.py
```

### `scripts`
These scripts are to be used in conjunction with the `qa` library above and are thus not used by the apps. 

```
scripts
├── config.txt
├── config_multi-train.py
├── evaluate.py
├── model.py
├── multi-train.py
└── train.py
```

### `cml`
These scripts serve as launch instructions to facilitate the automated project setup on CML. Each script is triggered by the declarative pipeline as defined in the `.project-metadata.yaml` file found in the project's root directory.

```
cml
├── install_dependencies.py
└── download_data_and_models.py
```
The three applications require two open-source QA datasets, as well as five pre-trained QA models. `download_data_and_models.py` ensures that these datasets and models are immediately available once the automated project setup on CML has completed. 

If you are performing manual setup of this repo, you can either run that script explicitly, or simply start using the apps as intended. The functions are designed such that, if a model is called that has not yet been downloaded, the code will automatically initiate a model download. 

### Installation
The code and applications within were developed against Python 3.6.9, and are likely also to function with more recent versions of Python.

To install dependencies, first create and activate a new virtual environment through your preferred means, then pip install from the requirements file. I recommend:

```python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

In CML or CDSW, no virtual env is necessary. Instead, inside a Python 3 session (with at least 2 vCPU / 8 GiB Memory), simply run

```python
!pip3 install -r requirements.txt     # notice `pip3`, not `pip`
```

Next, install the `qa` module from this repository, with

```python
pip3 install -e .
```

from inside the root directory of this repo.

### Data
The apps and scripts designed to work with the `qa` library make use of two open source question answering datasets. 

* [SQuAD2.0](https://rajpurkar.github.io/SQuAD-explorer/) is a canonical QA dataset that is the current benchmark against which all QA models are evaluated. 
* [COVID-QA](https://github.com/deepset-ai/COVID-QA) is a smaller dataset of medical research papers and questions focused on the novel coronavirus. 

-------------------------

## For NLP Practitioners
The `qa` module and `scripts` were designed for training and evaluating Transformer models for question answering on CML/CDSW. For example, one can train a vanilla version of BERT on the SQuAD dataset thus teaching it to perform extractive QA. The remainder of this README provides details and instructions for using the `qa` library in conjunction with the `scripts`, as well as how to harness the CML/CDSW abstractions to launch Jobs, Experiments, and deploy Models. 

### Overview of the Train and Evaluate workflows
At their core, the `scripts/train.py` and `scripts/evaluate.py` scripts are based heavily on a [script](https://github.com/huggingface/transformers/blob/master/examples/question-answering/run_squad.py)
developed by the [HuggingFace](https://huggingface.co/) team for training [Transformer models](http://jalammar.github.io/illustrated-transformer/) 
(like BERT) for the task of [extractive question answering](https://huggingface.co/transformers/v3.2.0/task_summary.html#extractive-question-answering). 
However, that script is cumbersome to use in CDSW/CML, so it has been modified to work in this environment. The result of this modification is that these scripts require a configuration file (`config.txt`) as an argument. The config file contains required and optional parameters that provide an impressive amount of flexibility and functionality when training and evaluating Transformer models. **Many of these parameters are geared towards experienced NLP practictioners; most should be left as their defaults.** 

The first three arguments in `config.txt` are required so we describe them briefly here:
* `--model_type`: Must be a valid model type allowed by HuggingFace (e.g. "bert" or "roberta") 
* `--model_name_or_path`: Can be either of the following: 
  * A string with the _identifier_ name of a pretrained model to load from cache or download from the 
  [HF model repository](https://huggingface.co/models), e.g., "bert-base-uncased" or "deepset/bert-base-cased-squad2"
  * A path to a directory containing model weights, e.g., `/home/cdsw/models/my_model_directory/`
* `--output_dir`: path of directory to store trained model weights (if training) or predictions and performance results (if evaluating)

Note: The first time an identifier name is called, those model weights will be downloaded and cached from the HF model repo. This can take a long time and a lot of disk space for large models. Additional calls to that particular model name will load model weights from the cache.

Note: When training or evaluating models on the SQuAD2.0 dataset, you must use an Engine Profile with a minimum of 2 vCPU / 16 GiB. Any less and the session / job will fail due to memory constraints. A GPU is _strongly_ preferred for these tasks!

Note: Adding an argument (e.g. `--train-file  my_training_file.json`) after the call to the config file will override the value for that argument in the config file. This is useful if you want to programmatically train or evaluate on many datasets in succession. See an example in `scripts/multi-train.py`. 

### Training and evaluating a QA model in a Session

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


### Training and evaluating a QA model as a Job
The `scripts/multi-train.py` script demonstrates how to train and evaluate in succession. This script is designed to be run as a Job by selecting this file in the **Script** field of the Create a Job menu. 

### Serving a QA Model
Once you have a QA Model that you're satisfied with you can expose it via the Model abstraction. To do so, update the `DEFAULT_MODEL` global key at the top of `scripts/model.py` to reflect the model name or model directory you wish to serve. Then select this script under the **Script** field in the Create a Model menu. 

-------------------------