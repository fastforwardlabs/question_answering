import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import json

#SQUAD_DIR = "/Users/mbeck/Projects/ff14/data/raw/squad/"
#MED_DIR = "/Users/mbeck/Projects/ff14/data/raw/covid/"

SQUAD_DIR = "/home/cdsw/data/squad/"
MED_DIR = "/home/cdsw/data/covidQA/"

COLORS = ["#00828c", "#ff8300"]

@st.cache(allow_output_mutation=True)
def load_data(data_type, set_type='train'):
  if 'squad' in data_type.lower():
    filename = f"{SQUAD_DIR}/{set_type}-v2.0.json"
  if 'medical' in data_type.lower():
    #filename = f"{MED_DIR}/covid_bioasq_{set_type}.json"
    filename = f"{MED_DIR}/COVID-QA.json"
  
  data = json.load(open(filename, 'r'))
  stats = compute_dataset_statistics(data)
  return data, stats

def compute_dataset_statistics(data):
  stats = {
    "Questions per paragraph":[],
    "Paragraphs per article":[],
    "Tokens per context":[],
    "Tokens per answer":[]
  }
  for entry in data['data']:
    stats['Paragraphs per article'].append(len(entry['paragraphs']))

    for paragraph in entry['paragraphs']:
      stats['Tokens per context'].append(len(paragraph['context'].split(" ")))

      questions = paragraph['qas']
      stats['Questions per paragraph'].append(len(questions))

      for question in questions:
        for answer in question['answers']:
          stats['Tokens per answer'].append(len(answer['text'].split(" ")))
  return stats

def run_tutorial(dataset, data):
  if 'squad' in dataset.lower():
    return squad_tutorial("train", data)
  if 'medical' in dataset.lower():
    return medical_tutorial("train", data)

def squad_tutorial(set_type, data):
  st.subheader("Question & Answers")
  st.markdown("In the SQuAD2.0 training set, the model will train on a \
    (_question_, _context_) pair, text that is tokenized and fed to the model. \
    During training, the model will predict an answer for that question given \
    the context. Training loss is computed based on how closely the predicted answer \
    matches the true answer associated with each question.")

  st.markdown("Each question in the SQuAD2.0 training set has a unique `id`, \
    an `is_impossible` flag indicating whether this question can be answered \
    within the context, and a list of true answers. (All questions are answerable \
    in the training set.)")
  st.markdown("Answers consist of the answer `text` \
    and the `answer_start` -- an integer indicating the starting index of the \
    answer within the context.")

  st.json(data['data'][0]['paragraphs'][0]['qas'][0])


  st.subheader("Paragraph")
  st.markdown("A SQuAD2.0 paragraph consists of the `context` and a list of \
    all related questions (`qas`) and their associated answers. The context and each of its \
    related questions are passed in pairs to the model during training. This means \
    the model sees the same paragraph many times!")
  st.json(data['data'][0]['paragraphs'][0])


  st.subheader("Articles")
  st.markdown("Each `paragraph` is just one of many belonging to a wikipedia article. \
    The SQuAD2.0 training set is composed of XXX articles. \
    Here's a summary of the full json structure. ")

  example = {
    'data':[
      {
        'title':data['data'][0]['title'],
        'paragraphs':[
          {
            "context":"paragraph about Beyoncé",
            "qas":[
              {
                'question':'question about Beyoncé',
                'id': 'unique id',
                'answers': [
                  {
                    'text':'answer about Beyoncé',
                    'answer_start': 0
                  },
                ],
                'is_impossible': False
              }
            ]
          }
        ]
      }
    ],
    'version':"2.0"
  }
  st.json(example)

  st.markdown("You can check out the full first article in all it's json glory.")
  if st.checkbox("Show me Beyoncé!"):
    st.json(data['data'][0])


def print_summary_stats(stats, dataset):
  st.markdown(f"### {dataset} Summary")
  st.write("Total number of articles:", len(stats['Paragraphs per article']))
  st.write("Total number of questions:", np.sum(stats['Questions per paragraph']))
  st.write("Average number of paragraphs per article:", 
            round(np.mean(stats['Paragraphs per article'])))

def plot_selection(stats, SELECTION, dataset=None):
  if 'squad' in dataset.lower():
    color = COLORS[0]
  else:
    color = COLORS[1]

  COL_NAME = SELECTION.split(" ")[-1]+'s'
  hist = np.histogram(stats[SELECTION])
  df = pd.DataFrame(hist).T.dropna()
  df.rename(columns={0:COL_NAME, 1:SELECTION}, inplace=True)
  
  chart = alt.Chart(df).mark_bar(color=color).encode(
      x=SELECTION, 
      y=COL_NAME,
    )
  if dataset:
    st.subheader(dataset)
  altair_chart = st.altair_chart(chart, use_container_width=True)


# ----------- START THE APP -----------
st.title("QA Data Visualizer")
st.write("Question answering applications seem to work like magic, providing \
          answers when given raw blocks of text and a corresponding question. \
          However, training and evaluating these systems requires highly \
          structured data. This app explores that structure and allows you to \
          compare various aspects of different training datasets.")

st.markdown("The training and evaluation scripts in this repo are based on \
              those implemented by the HuggingFace team in the [transformers](https://huggingface.co/transformers/) \
              Python library. These scripts require data in JSON format with \
              a specific schema and keywords. The minimum required schema is \
              shown below (you can collapse it by clicking the arrows).")


st.json({
    'data':[
      {
        'paragraphs':[
          {
            "context":"Words in ",
            "qas":[
              {
                'question':'question text',
                'id': 'unique id',
                'answers': [
                  {
                    'text':'answer text',
                    'answer_start': 0
                  },
                ],
                'is_impossible': False
              }
            ]
          }
        ]
      }
    ],
  })

st.markdown("QA datasets can have minor variations on this theme; details \
            can be explored for the SQuAD2.0 dataset as an optional tutorial below.")

DATASETS = st.multiselect("Choose one or more datasets to explore", 
                            ['SQuAD2.0','Medical Research'])

### Load data and offer in depth data tutorials
tutorials = {}
datadict = {}
statsdict = {}
for dataset in DATASETS:
  data, stats = load_data(dataset, "train")
  datadict[dataset] = data
  statsdict[dataset] = stats
  if dataset == 'SQuAD2.0':
    tutorials[dataset] = st.checkbox(f"Walk me through the {dataset} JSON schema tutorial.") 
  
### Show selected tutorials if desired
for dataset,to_run in tutorials.items():
  if to_run and (dataset == 'SQuAD2.0'): 
    st.header("Explore the JSON Schema")
    run_tutorial(dataset, datadict[dataset]) 

if DATASETS:
  st.header("Global Statistics")
  st.markdown("While the basic schema must be the same for all datasets, \
          there is still room for variation. For example, articles need not \
          be broken down into paragraphs. In fact, this is precisely how the \
          Medical Research dataset is structured as you might notice \
          in the summary below.")

  for dataset, stats in statsdict.items():
    print_summary_stats(stats, dataset)

  st.header("Distributions")
  st.markdown("Below you can select and view various distributions for your \
              chosen dataset(s). How many questions are associated with each \
              paragraph (or article)? How many tokens (words) are in a \
              paragraph (context)?")
  SELECTION = st.selectbox('Select some data', [k for k,v in stats.items()])
  for dataset, stats in statsdict.items():
    plot_selection(stats, SELECTION, dataset=dataset)
