import streamlit as st
import wikipedia as wiki
import pandas as pd
from PIL import Image
import requests
from io import BytesIO
from transformers import (
    pipeline, 
    AutoModelForQuestionAnswering,
    AutoTokenizer
)

MODEL_OPTIONS = {
    "BERT":         "deepset/bert-base-cased-squad2",
    "RoBERTa":      "mbeck/roberta-base-squad2", 
    "DistilBERT":   "twmkn9/distilbert-base-uncased-squad2",
    "MiniLM":       "deepset/minilm-uncased-squad2",
    "XLM-RoBERTa":  "deepset/xlm-roberta-large-squad2"
    }

@st.cache(allow_output_mutation=True)
def load_model(model_choice):
    model_name = MODEL_OPTIONS[model_choice]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline('question-answering', model=model_name, tokenizer=tokenizer)
    return model

def highlight_text(segment, context, full_text=False):
    if segment not in context:
        return 
    
    length = len(segment)

    if full_text: 
        # find the section the answer was found in and display only that section
        chunks = context.split("==") 
        for chunk in chunks:
            if segment in chunk:
                idx = chunk.index(segment)
                chunk1 = chunk[:idx]
                chunk2 = chunk[idx:idx+length]
                chunk3 = chunk[idx+length:]
                break
    else:
        idx = context.index(segment)
        chunk1 = context[:idx]
        chunk2 = context[idx:idx+length]
        chunk3 = context[idx+length:]

    new_context = chunk1 + '<span style="background-color: #FFFF00"> **' + chunk2 + '** </span>' + chunk3
    return new_context

def make_url(segment, url):
    new_segment = f'<a target="_blank" href="{url}">{segment}</a>'
    return new_segment

# ------ SIDEBAR SELECTIONS ------ 
model_choice = st.sidebar.selectbox(
    "Choose a Transformer model:",
    list(MODEL_OPTIONS.keys())
)

number_of_pages = st.sidebar.slider(
    "How many Wikipedia pages should be displayed?", 1, 5, 1)

number_of_answers = st.sidebar.slider(
    "How many answers should the model suggest for each Wikipedia page?", 1,5,1)

use_full_text = st.sidebar.checkbox(
    "Use full text rather than summaries of Wikipedia pages (slower but better answers)"
)

# ------ BEGIN APP ------ 
st.title("Question Answering with ")
response = requests.get("https://upload.wikimedia.org/wikipedia/commons/5/53/Wikipedia-logo-en-big.png")
st.image(Image.open(BytesIO(response.content)), width=400)

st.markdown("This app demonstrates how to build a simple question answering system on Wikipedia. \
    You can choose one of five Transformer models that have already been trained for extractive \
    question answering. When selecting a model, if that model has never been called before, its \
    weights will be downloaded before it can be used for question answering. For large \
    models this can take quite some time. Once cached, switching between models will be seemless.")

st.markdown("A question entered below will first be used in Wikipedia's default search engine, \
    resulting in a ranked list of Wikipedia pages that are most related to the question. \
    The question and each Wikipedia page are then sent to the QA model, which returns answers \
    extracted from the text. These answers are displayed below a snippet of the Wikipedia article. \
    By default, only the page summary is used when searching for answers to questions. \
    This saves time since many Wikipedia pages are very long and QA models can be very slow. \
    However you can choose to use the full text of the article in the sidebar. This usually provides \
    better results, though it does take longer to process.")

# ------ LOAD QA MODEL ------ 
reader = load_model(model_choice)

# ------ GET QUESTION ------ 
st.markdown("## Ask a question")
query = st.text_input('Enter text here', 'Why is the sky blue?')
st.markdown(f"## Displaying the top {number_of_pages} results:")

# ------ SEARCH ENGINE (RETRIEVER) ------ 
results = wiki.search(query, results=number_of_pages)

# ------ ANSWER EXTRACTION (READER) ------ 
for i, result in enumerate(results):
    wiki_page = wiki.page(result, auto_suggest=False)

    # display the Wiki title as a URL
    title_url = make_url(result, wiki_page.url)
    st.markdown("### "+ str(i+1)+') '+title_url, unsafe_allow_html=True)

    # grab text for answer extraction
    if use_full_text:
        context = wiki_page.content
    else:
        context = wiki_page.summary
        
    # extract answers
    inputs = {'question':query, 'context':context}
    answers = reader(inputs, **{'topk':number_of_answers})
    try: 
        answerdf = pd.DataFrame(answers)
    except: 
        answerdf = pd.DataFrame(answers, index=[0]) 

    # display results
    hilite_context = highlight_text(answerdf['answer'][0], context, full_text=use_full_text)
    st.markdown(hilite_context, unsafe_allow_html=True)

    answerdf.drop(columns=['start', 'end'], inplace=True)
    st.table(answerdf)