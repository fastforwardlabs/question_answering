# Training and evaluating your own Transformers for QA
The scripts in this directory are designed to work with the `qa` module to facilitate training and evaluating Transformer models for question answering on CML/CDSW. For example, one can train a vanilla version of BERT on the SQuAD dataset (or your own data!) thus teaching it to perform extractive QA for your use case. What follows are detailed instructions for using the `qa` library in conjunction with these scripts, as well as how to harness the CML/CDSW abstractions to launch Jobs, Experiments, and deploy trained Models. 

## Important Notes
1. At their core, the `train.py` and `evaluate.py` scripts are based heavily on a [script](https://github.com/huggingface/transformers/blob/v2.11.0/examples/question-answering/run_squad.py) developed by the [HuggingFace](https://huggingface.co/) team for training [Transformer models](http://jalammar.github.io/illustrated-transformer/) (like BERT) for the task of [extractive question answering](https://huggingface.co/transformers/v3.2.0/task_summary.html#extractive-question-answering). However, that script is cumbersome to use in CDSW/CML, so it has been modified to work in this environment. The result is that these scripts require a configuration file (`config.txt`) as an argument (examples below). The config file contains required and optional parameters that provide an impressive amount of flexibility and functionality when training and evaluating Transformer models. **Many of these parameters are geared towards experienced NLP practictioners; most should be left as their defaults.** 

2. The first time a model identifier name is called, those model weights will be downloaded and cached from the HuggingFace model repo. This can take a while and a considerable amount of disk space for large models (e.g., XLM-RoBERTa is several GiBs). Additional calls to that particular model name will load model weights from the cache.

3. When training or evaluating models on the SQuAD2.0 dataset, you must use an Engine Profile with a minimum of 2 vCPU / 16 GiB. Any less and the session / job will fail due to memory constraints. 

4. A GPU is _strongly_ preferred for these tasks! For example, training BERT on SQuAD2.0 **with** a modest GPU takes about 2 hours, and without one it can take a full day.

## The `config.txt` file
We encourage you to first familiarize yourself with the parameters in the `config.txt` file. While many of these can be left as their defaults when getting started, tweaking some of these can lead to better QA models. 

The first three arguments in `config.txt` are required for any of the training and evaluation scripts:
* `--model_type`: Must be a valid model type allowed by HuggingFace (e.g. "bert" or "roberta") 
* `--model_name_or_path`: Can be either of the following: 
  * A string with the _identifier_ name of a pretrained model to load from cache or download from the 
  [HF model repository](https://huggingface.co/models), e.g., "bert-base-uncased" or "deepset/bert-base-cased-squad2"
  * A path to a directory containing model weights, e.g., `/home/cdsw/models/my_model_directory/`. A valid model directory must contain a `pytorch_model.bin` file which holds the binary model weights. 
* `--output_dir`: path of directory to store trained model weights (if training) or predictions and performance results (if evaluating)

### General usage
We stress that the `qa` library is not intended for direct interactive use. Functionality must be harnessed through the scripts, which in turn **require** the `config` file. Once the three required parameters are updated in the `config` file, execute a script (in a Session, for example) like so: 

`!python3 scripts/evaluate.py @scripts/config.txt`

Here, the `@` symbol tells the `evaluate.py` script to expect a filename (the `config` file) directly following the symbol. Each uncommented line in `config.txt` is then parsed into a command line argument. 

You can also add additional command line arguments like so: 

`!python3 scripts/evaluate.py @scripts/config.txt --train-file my_training_file.json`

In this case, `my_training_file.json` would overwrite whatever value was stored in the corresponding `--train-file` slot in the `config.txt`. This is useful if you want to programmatically update parameters, for example, when running Experiments or in a Job. See an example in `scripts/multi-train.py`. 

### Other important parameters

* `--version_2_with_negative`: This poorly-named flag is a relic from SQuAD2.0 and should be used whenever working with a dataset that contains both _answerable_ and _unanswerable_ questions (like SQuAD2.0)
* `--do_lower_case`: This flag must be used when working with a model whose tokenizer uses only lower case. For example, `bert-base-cased` is a cased model, so this flag should be commented out; while `twmkn9/distilbert-base-uncased-squad2` is an _uncased_ model so this flag should _not_ be commented out.
* `--no_cuda`: Towards the bottom, under Misc. parameters, you'll find this gem. Uncommenting this parameter forces the training and evaluation scripts to use only CPUs, even if GPUs are available. It is currently not commented because we assume most CML/CDSW instances may not have immediate access to GPUs. If you do have GPUs and would like to use them, comment this out. 


## Training and evaluating a QA model in a Session

### To Train
In `config.txt`, first specify the following information: 
* the model you want to train 
* the output directory where trained model weights will be stored
* the directory and filename of the training data 

Then run the following in an open Session:
`!python3 scripts/train.py @scripts/config.txt`

This kicks off a training session, which will output a lot of information to the screen detailing the training process, including a progress bar. 

### To Evaluate 
In `config.txt`, first specify the following information: 
* the model you wish to evaluate 
* the output directory where results and predictions will be stored
* the directory and filename of the validation data

Then run the following in an open Session:
`!python3 scripts/evaluate.py @scripts/config.txt`

This kicks off an evaluation session, which will output a lot of information to the screen detailing the evaluation process, including a progress bar. 

## Training and evaluating a QA model as a Job
The `multi-train.py` script demonstrates how to train and evaluate in succession. This script is designed to be run as a Job by selecting this file in the **Script** field of the Create a Job menu. 

## Monitoring training with Tensorboard
During training, the learning rate and training loss are automatically logged. These logs are stored under a `runs` directory that is created during the training process. Tensorboard can be used to visualize these `runs` to monitor the training process. To do so, simply execute the following in a Session: 

`!python3 apps/launch_tensorboard.py` 

You can also launch Tensorboard using the Application abstraction, as is done for the Streamlit apps that are the focus of this prototype. 

You can also change or add additional values to track during the training process by modifiying lines `#346` and `#347` in the `train_model` function in `qa/model_utils.py`. 


## Serving a QA Model
Once you have a QA Model that you're satisfied with you can expose it via the Model abstraction through the `model.py` script. To do so, update the `DEFAULT_MODEL` global key at the top of `model.py` to reflect the model name or model directory you wish to serve. Then select this script under the **Script** field in the Create a Model menu. 

> Note - In the next iteration of this prototype we'll include functionality that will allow you to use the model you train and serve as part of the WikiQA demo. Stay tuned!
>