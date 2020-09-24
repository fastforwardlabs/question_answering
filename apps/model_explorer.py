import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import json

from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

from qa.data.processing import SquadLikeProcessor


# Plotting defaults
plt.style.use('fivethirtyeight')
COLORS = ["#00828c", "#ff8300"]

PREDICTION_DIR = "/Users/mbeck/Projects/cdsw-qa/question_answering/data/predictions/"
MODEL_DIR = "/Users/mbeck/Projects/cdsw-qa/question_answering/models/"
DATA_DIR = "/Users/mbeck/Projects/ff14/data/raw/squad/"

PREDICTION_OPTIONS = {
    "BERT":         "deepset-bert-base-cased-squad2",
    "RoBERTa":      "my-roberta-base-squad2", 
    "DistilBERT":   "twmkn9-distilbert-base-uncased-squad2",
    "MiniLM":       "deepset-minilm-uncased-squad2/",
    "XLM-RoBERTa":  "deepset-xlm-roberta-large-squad2",   
}

PLOT_OPTIONS = {
    "Exact Match": 'exact',
    "F1": 'f1',
    "Exact Match (answerable questions)": 'HasAns_exact',
    "F1 (answerable questions)": 'HasAns_f1',
    "Exact Match (unanswerable questions)": "NoAns_exact",
    "F1 (unanswerable questions)": 'NoAns_f1'
}

MODEL_OPTIONS = {
    "BERT":         "deepset/bert-base-cased-squad2",
    "RoBERTa":      MODEL_DIR+"my-roberta-base-squad2", # local model
    "DistilBERT":   "twmkn9/distilbert-base-uncased-squad2",
    "MiniLM":       "deepset/minilm-uncased-squad2",
    "XLM-RoBERTa":  "deepset/xlm-roberta-large-squad2"
    }

EXAMPLE_OPTIONS = {
    "What theory least best describes gravity?": "5ad28035d7d075001a4297a6",
    "Rudyard Kipling was an influential spokesman for what?": "5730b6592461fd1900a9cfd2",
    "What nationality was Louis XIII originally?": "5ad24b60d7d075001a428be6",
    "Who allegedly haunted the gate?": "57108ee6a58dae1900cd6a1c",
    "What do rapid concentrated sources of oxygen promote?": "5ad2678ad7d075001a42922c",
    "Why do people chose civil disobedience to protest?": "5728d4c03acd2414000dffa1",
    "I'm feeling lucky": "I'm feeling lucky"
}

def load_results(data_dir):
    data = json.load(open(data_dir+"/results_.json", "r"))
    return pd.DataFrame(data, index=[0])

def load_all_results(prediction_dict):
    dfs = []
    for model_name, data_dir in prediction_dict.items():
        results = load_results(PREDICTION_DIR+data_dir)
        results['model'] = model_name
        dfs.append(results)
    df = pd.concat(dfs, ignore_index=True)
    df.set_index('model', inplace=True)
    return df

def load_predictions(data_dir):
    preds = json.load(open(data_dir+"/predictions_.json", 'r'))
    return pd.Series(preds)

def load_all_predictions(prediction_dict):
    dfs = []
    for model_name, data_dir in prediction_dict.items():
        predictions = load_predictions(PREDICTION_DIR+data_dir)
        series = pd.Series(data=predictions, name=model_name)
        dfs.append(pd.DataFrame(series))
    df = pd.concat(dfs, axis=1)
    return df    

def plot_results(selection):
    column = PLOT_OPTIONS[selection]

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot()
    ax2 = ax.twinx()
    ax2.grid(False)

    xticks = np.arange(len(all_results.index))
    yticks = np.linspace(0, 90, 10)
    yticks2 = np.around(np.linspace(0, 0.03, 10), decimals=3)
    
    width = .35

    ax.bar(xticks-width/2, all_results[column], width=width, color=COLORS[0], label='thing')
    ax2.bar(xticks+width/2, all_results['avg_time_per_example'], width=width, color=COLORS[1], label='time')

    if "exact" in column:
        ax.set_ylabel('Exact Match')
    else:
        ax.set_ylabel("F1 Score")
    ax2.set_ylabel("Inference time [sec]")

    ax.set_yticks(yticks)
    ax2.set_yticks(yticks2)
    ax.set_xticks(xticks)
    ax.set_xticklabels(all_results.index)

    fig.tight_layout()
 
    st.pyplot(fig)

@st.cache(allow_output_mutation=True)
def load_qa_model(model_choice):
    model_name_or_path = MODEL_OPTIONS[model_choice]
    return pipeline('question-answering', model=model_name_or_path, tokenizer=model_name_or_path)

@st.cache(allow_output_mutation=True)
def load_squad_validation_set():
    processor = SquadLikeProcessor()
    squad_validation_set = processor.get_dev_examples(DATA_DIR, "dev-v2.0.json")

    qid_to_index = {example.qas_id: i for i, example in enumerate(squad_validation_set)}
    index_to_qid = {i:example.qas_id for i, example in enumerate(squad_validation_set)}
    #answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if has_answer]
    #no_answer_qids = [qas_id for qas_id, has_answer in qid_to_has_answer.items() if not has_answer]
    return squad_validation_set, qid_to_index, index_to_qid

def load_squad_example(idx, squad_validation_set):
    example = squad_validation_set[idx]
    question = example.question_text
    context = example.context_text
    answers = [answer['text'] for answer in example.answers]
    return question, context, answers


st.title("QA Model Explorer")
st.markdown("What do I want to show here? ")

st.header("Comparing model performance: Quantitative")
st.markdown("Which model should you choose for your application? That will \
    depend on several factors including the performance of the model on a \
    validation set, the inference time of the model, and memory constraints, among other considerations. \
    Here we'll explore the first two: validation results and inference time.")
       
st.markdown("We evaluated several QA models against the SQuAD2.0 validation set. \
    The SQuAD2.0 validation set contains nearly 12,000 \
    examples; about half of them are _answerable_ -- the answer can be found \
    in the context. The other half are impossible to answer, given the context.\
    (Read more about the purpose of unanswerable questions in \
    [THIS BLOG I WROTE](https://qa.fastforwardlabs.com/no%20answer/null%20threshold/bert/distilbert/exact%20match/f1/robust%20predictions/2020/06/09/Evaluating_BERT_on_SQuAD.html) \
    You can explore results for all examples, just the _answerable_ ones, or\
    just the _unanswerable_ ones.")

all_results = load_all_results(PREDICTION_OPTIONS)

plot_selection = st.selectbox("Select a quantity to view", 
                                [k for k,v in PLOT_OPTIONS.items()])

plot_results(plot_selection)

st.markdown("#### What do these quantities mean?")
st.markdown("F1 score and Exact Match are widely used metrics for question \
    answering tasks. They are computed by comparing the characters of the \
    model's predicted answer to the ground truth answer for each question and\
    taking an average over all questions (or just the _answerable_ or \
    _unanswerable_ questions).")
st.latex(r'''
    \begin{aligned}
        \mathrm{F1}& = \frac{2}{\mathrm{recall^{-1}} + \mathrm{precision^{-1}}} 
                     = 2 * \frac{\mathrm{precision} * \mathrm{recall}}{\mathrm{precision} + \mathrm{recall}} \\ \\
        \mathrm{EM}& = \begin{cases}
                            1 &\text{if } \text{\textbf{perfect} match} \\
                            0 &\text{otherwise}
                        \end{cases}
    \end{aligned}
''')


st.header("Comparing model performance: Qualitative")
st.markdown("Select a model and some canned question/context pairs and see for yourself")

model_predictions = load_all_predictions(PREDICTION_OPTIONS)
squad_validation_set, qid_to_idx, idx_to_qid = load_squad_validation_set()

question_selection = st.selectbox("Choose an example", 
                                  [k for k,v in EXAMPLE_OPTIONS.items()])
qid = EXAMPLE_OPTIONS[question_selection]

if qid == "I'm feeling lucky":
    idx = np.random.randint(0,len(squad_validation_set)-1, 1)[0]
    question, context, answers = load_squad_example(idx, squad_validation_set)
else:
    idx = qid_to_idx[qid]
    question, context, answers = load_squad_example(idx, squad_validation_set) 

st.markdown("### Question")
st.markdown(question)
st.markdown("### Context")
st.markdown(context)

st.markdown("### Model Predictions")
# Model predictions are indexed on the question id (qid)
qid = idx_to_qid[idx]
st.table(model_predictions.loc[qid].T)

st.markdown("### Ground Truth Answer(s)")
if answers:
    st.write(list(set(answers))) 
else:
    st.write("This question is impossible to answer given the context!")                             

#model_selection = st.selectbox("Choose a model to load", 
#                                [k for k,v in MODEL_OPTIONS.items()])
#qa = load_qa_model(model_selection)

