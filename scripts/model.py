from transformers import pipeline

"""
MODEL KWARGS
kwargs.setdefault("topk", 1)
kwargs.setdefault("doc_stride", 128)
kwargs.setdefault("max_answer_len", 15)
kwargs.setdefault("max_seq_len", 384)
kwargs.setdefault("max_question_len", 64)
kwargs.setdefault("handle_impossible_answer", False)
"""


DEFAULT_MODEL = "deepset/bert-base-cased-squad2"
DEVICE = -1 # cpu only
    
qa_model = pipeline('question-answering', 
                    model=DEFAULT_MODEL, 
                    tokenizer=DEFAULT_MODEL,
                    device=DEVICE)
  
  
def question_answering(args):
  """
  Model API for use with MLViz must adhere to the following schema:
  
  INPUT: 
  {
    "version": "1.0",
    "data": {
      "colnames": ["question_text", "context_text"],
      "coltypes": ["STRING", "STRING"],
      "rows": [
        ["Do retrievers shed?", "Golden retrievers are well suited to residency in suburban or country environments. They shed copiously, particularly at the change of seasons, and require fairly regular grooming. The Golden Retriever was originally bred in Scotland in the mid-19th century."],
      ]
    }
  }

  OUTPUT:
  {
    "version": "1.0",
    "data": {
      "colnames": ["answer_text", "answer_score"],
      "coltypes": ["STRING", "REAL"],
      "rows": [
        ["They shed copiously,", 0.7170811319842585], 
      ]
    }
  }
  """
  
  rows = args.get("data").get("rows")
  
  result = {
    "colnames": ['answer_text', 'answer_score'],
    "coltypes": ['STRING', 'REAL']
  }
  
  outRows = []
  for row in rows:
  
    inputs = {"question": row[0], "context": row[1]}
    
    # prediction: {'answer': str, 'score": float, "start": int, "end": int}
    prediction = qa_model.predict(inputs)

    outRows.append([prediction['answer'], prediction['score']])
    
  result['rows'] = outRows
  
  return {
    "version": "1.0",
    "data": result
  }


# Test the model itself
context = """
The Golden Retriever is a medium-large gun dog that was bred to retrieve 
shot waterfowl, such as ducks and upland game birds, during hunting and 
shooting parties.[3] The name "retriever" refers to the breed's ability 
to retrieve shot game undamaged due to their soft mouth. Golden 
retrievers have an instinctive love of water, and are easy to train to 
basic or advanced obedience standards. They are a long-coated breed, with 
a dense inner coat that provides them with adequate warmth in the 
outdoors, and an outer coat that lies flat against their bodies and 
repels water. Golden retrievers are well suited to residency in suburban 
or country environments.[4] They shed copiously, particularly at the 
change of seasons, and require fairly regular grooming. The Golden 
Retriever was originally bred in Scotland in the mid-19th century.
"""
inputs = {"question":"Do retrievers shed?", 
          "context": context
}
prediction = qa_model.predict(inputs)
print(prediction)

# Test the function
question_answering(
  {
    "version": "1.0",
    "data": {
      "colnames": ["question_text", "context_text"],
      "coltypes": ["STRING", "STRING"],
      "rows": [
        ["Do retrievers shed?", 
         """
The Golden Retriever is a medium-large gun dog that was bred to retrieve 
shot waterfowl, such as ducks and upland game birds, during hunting and 
shooting parties.[3] The name "retriever" refers to the breed's ability 
to retrieve shot game undamaged due to their soft mouth. Golden 
retrievers have an instinctive love of water, and are easy to train to 
basic or advanced obedience standards. They are a long-coated breed, with 
a dense inner coat that provides them with adequate warmth in the 
outdoors, and an outer coat that lies flat against their bodies and 
repels water. Golden retrievers are well suited to residency in suburban 
or country environments.[4] They shed copiously, particularly at the 
change of seasons, and require fairly regular grooming. The Golden 
Retriever was originally bred in Scotland in the mid-19th century.
"""
        ]
      ]
    }
  }
)