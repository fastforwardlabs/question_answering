# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

from PIL import Image
from rank_bm25 import BM25Okapi
import wikipedia as wiki
import pandas as pd
import streamlit as st

from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

from qa.utils import absolute_path


MODEL_OPTIONS = {
    "BERT": "deepset/bert-base-cased-squad2",
    "RoBERTa": "mbeck/roberta-base-squad2",
    "DistilBERT": "twmkn9/distilbert-base-uncased-squad2",
    "MiniLM": "deepset/minilm-uncased-squad2",
    "XLM-RoBERTa": "deepset/xlm-roberta-large-squad2",
}

CONTEXT_OPTIONS = {
    "Wikipedia summary paragraph": "summary",
    "Full Wikipedia article": "full",
    "Use RelSnip to identify most relevant sections": "relsnip",
}


@st.cache(allow_output_mutation=True)
def load_model(model_choice):
    model_name = MODEL_OPTIONS[model_choice]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = pipeline("question-answering", model=model_name, tokenizer=tokenizer)
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
                chunk2 = chunk[idx : idx + length]
                chunk3 = chunk[idx + length :]
                break
    else:
        idx = context.index(segment)
        chunk1 = context[:idx]
        chunk2 = context[idx : idx + length]
        chunk3 = context[idx + length :]

    new_context = (
        chunk1
        + '<span style="background-color: #FFFF00"> **'
        + chunk2
        + "** </span>"
        + chunk3
    )
    return new_context


def relsnip(context, num_fragments=5):
    # Wiki section headings are wrapped with "==", (e.g., == Color ==)
    # split the context by article sections
    chunks = context.split("\n== ")

    # Remove sections that won't contain an answer
    chunks_cleaned = list()
    for chunk in chunks:
        subchunks = chunk.split(" ==")
        if subchunks[0] in [
            "See also",
            "References",
            "Further reading",
            "External links",
        ]:
            continue
        chunks_cleaned.append(chunk)

    # tokenize each chunk and pass to BM25 search algorithm
    tokenized_chunks = [chunk.split(" ") for chunk in chunks_cleaned]
    bm25 = BM25Okapi(tokenized_chunks)

    # tokenize the query and score each chunk
    tokenized_query = query.split(" ")
    chunk_scores = bm25.get_scores(tokenized_query)

    # sort the chunks by their BM25 score
    sorted_chunks = sorted([c for s, c in zip(chunk_scores, chunks)], reverse=True)

    # select the num_fragments highest scoring chunks
    short_context = ""
    for chunk in sorted_chunks[:num_fragments]:
        short_context = short_context + chunk

    return short_context


def make_url(segment, url):
    new_segment = f'<a target="_blank" href="{url}">{segment}</a>'
    return new_segment


# ------ SIDEBAR SELECTIONS ------
image = Image.open(absolute_path("images", "cloudera-fast-forward.png"))
st.sidebar.image(image, use_column_width=True)

st.sidebar.markdown(
    "This app demonstrates a simple question answering system on Wikipedia. \
    The question is first used in Wikipedia's default search engine, \
    resulting in a ranked list of relevant Wikipedia pages. \
    The question and each Wikipedia page are then sent to the QA model, which returns answers \
    extracted from the text."
)

model_choice = st.sidebar.selectbox(
    "Choose a Transformer model:", list(MODEL_OPTIONS.keys())
)

number_of_pages = st.sidebar.slider(
    "How many Wikipedia pages should be displayed?", 1, 5, 1
)

number_of_answers = st.sidebar.slider(
    "How many answers should the model suggest for each Wikipedia page?", 1, 5, 1
)

st.sidebar.text("")
st.sidebar.markdown(
    "By default, the QA Model will only process the Wikipedia **summary** for answers. \
    This saves time since Wikipedia pages are long and QA models are *slow*. \
    Here, you can opt to use the **full text** of the article, or you can \
    choose **RelSnip**, which uses BM25 to identify the most relevant sections \
    of Wikipedia pages."
)

context_choice = st.sidebar.selectbox(
    "Choose which part of the Wikipedia page(s) to process:",
    list(CONTEXT_OPTIONS.keys()),
)
context_selection = CONTEXT_OPTIONS[context_choice]
if context_selection == "relsnip":
    num_sections = st.sidebar.slider(
        "How many sections should RelSnip identify?", 3, 7, 5
    )

st.sidebar.markdown(
    "**NOTE: Including more text often results in a better answer, but longer inference times.**"
)

# ------ BEGIN APP ------
st.title("Question Answering with ")
image = absolute_path("images/669px-Wikipedia-logo-v2-en.svg.png")
st.image(Image.open(image), width=400)

# ------ LOAD QA MODEL ------
reader = load_model(model_choice)

# ------ GET QUESTION ------
st.markdown("## Ask a question")
query = st.text_input("Enter text here", "Why is the sky blue?")
st.markdown(f"## Displaying the top {number_of_pages} results:")

# ------ SEARCH ENGINE (RETRIEVER) ------
results = wiki.search(query, results=number_of_pages)

# ------ ANSWER EXTRACTION (READER) ------
for i, result in enumerate(results):
    wiki_page = wiki.page(result, auto_suggest=False)

    # display the Wiki title as a URL
    title_url = make_url(result, wiki_page.url)
    st.markdown("### " + str(i + 1) + ") " + title_url, unsafe_allow_html=True)

    use_full_text = True
    # grab text for answer extraction
    if context_selection == "full":
        context = wiki_page.content
    elif context_selection == "relsnip":
        context = wiki_page.content
        context = relsnip(context, num_sections)
    else:
        context = wiki_page.summary
        use_full_text = False

    # extract answers
    inputs = {"question": query, "context": context}
    answers = reader(inputs, **{"topk": number_of_answers})
    try:
        answerdf = pd.DataFrame(answers)
    except:
        answerdf = pd.DataFrame(answers, index=[0])

    # display results
    hilite_context = highlight_text(
        answerdf["answer"][0], context, full_text=use_full_text
    )
    st.markdown(hilite_context, unsafe_allow_html=True)

    answerdf.drop(columns=["start", "end"], inplace=True)
    st.table(answerdf)
