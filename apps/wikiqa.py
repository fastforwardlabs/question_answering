import streamlit as st
import wikipedia as wiki
from transformers import pipeline


MODEL_OPTIONS = {
    "BERT":         "deepset/bert-base-cased-squad2",
    "RoBERTa":      "mbeck/roberta-base-squad2", 
    "DistilBERT":   "twmkn9/distilbert-base-uncased-squad2",
    "MiniLM":       "deepset/minilm-uncased-squad2",
    "XLM-RoBERTa":  "deepset/xlm-roberta-large-squad2"
    }

@st.cache(allow_output_mutation=True)
def load_model(model_choice):
    model = MODEL_OPTIONS[model_choice]
    return pipeline('question-answering', model=model, tokenizer=model)



model_choice = st.sidebar.selectbox(
    "Choose a Transformer model:",
    list(MODEL_OPTIONS.keys())
)

reader = load_model(model_choice)

st.title("Basic QA with Wikipedia")

question = st.text_input('Ask a Question', 'Why is the sky blue?')

results = wiki.search(question)

for result in results:
  st.markdown(wiki.page(result))

page = wiki.page(results[0])
st.write(f"Top result: {page}")

topk = st.slider("Return top k answers", 1,5,1)

inputs = {'question':question, 'context':page.content}
answers = reader(inputs, **{'topk':topk})

answerdf = pd.DataFrame(answers)
st.table(answerdf)

#st.write("Answer:", reader(inputs, **{'topk':topk}))

st.markdown('###### Found from this Wikipedia page:')
st.markdown(f'<a target="_blank" href="{page.url}">{page.title}</a>',
            unsafe_allow_html=True)

