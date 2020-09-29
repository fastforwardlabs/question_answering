import os
import json
import numpy as np
import pandas as pd
from qa.utils import set_seed

DEFAULT_SIZES = [500, 1000, 1500, 2000, 2500, 3000]

def randomize_indices(data):
  idx = np.arange(len(data))
  return np.random.permutation(idx)

def partition_data(data, indices_or_sections=None, seed=42):
  """
  data should be a ... what??? list? 
  partitions can be a number (as in, the number of partitions) or a list of data sizes? 
  """
  set_seed(seed)

  dd = np.array(data)

  idx = randomize_indices(data)
  idx_chunks = np.array_split(idx, indices_or_sections)
  partitions = [list(dd[chunk]) for chunk in idx_chunks]
  return partitions

def create_increasing_sized_train_sets(json_data_file, sizes=DEFAULT_SIZES, **kwargs):
  """
  json_data_file: filename of the original, full dataset from which to create subsets
  sizes: list of dataset sizes to partition the original dataset into
          these will translate into increasing sized datasets, with each 
          successive dataset consisting of the previous subset's examples plus
          the number of additional examples identified in the splits
          
  Takes filename of a json dataset and the desired sizes and creates 
  subsets of increasing size. These subsets are saved to the directory 
  associated with the json_data_file. 
  """
  outfile_prefix = os.path.splitext(json_data_file)[0]
  json_data = json.load(open(json_data_file, 'r'))

  data_chunks = partition_data(json_data['data'][0]['paragraphs'], sizes, **kwargs)
  
  new_json = {'data':[{'paragraphs':[]}]}
  for chunk in data_chunks:
    try:
      num_examples += len(chunk)
    except:
      num_examples = len(chunk)

    new_json['data'][0]['paragraphs'] += chunk
    json.dump(new_json, open(f"{outfile_prefix}_{num_examples}.json", "w"))
    
    
def load_results(data_dir):
    data = json.load(open(data_dir+"/results_.json", "r"))
    return pd.DataFrame(data, index=[0])

def load_predictions(data_dir):
    preds = json.load(open(data_dir+"/predictions_.json", 'r'))
    return pd.Series(preds)
  
def load_nbest_predictions(data_dir):
    """
    I have not tested this and I have no idea if it works.
    """
    preds = json.load(open(data_dir+"/nbest_predictions_.json",'r'))
    return pd.DataFrame(data, index=[0])

