# In this file we'll process data 
#  basic function to convert text to squad format
#  convert squad format stuff to "examples" and "features" for use with transformers
#  

import os 

import torch


from transformers.data.processors.squad import SquadV1Processor, SquadV2Processor


