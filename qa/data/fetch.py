# helper functions for downloading various QA datasets

def download_squad(version=None):
    if version == 1:
        !wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
        !wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
    elif version == 2:
        !wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
        !wget -P $SQUAD_DIR https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
    else:
        print("Please specificy SQuAD version number and try again.")
        

def download_covidQA():
    !wget -P /home/cdsw/data/covidQA https://github.com/deepset-ai/COVID-QA/raw/master/data/question-answering/COVID-QA.json


def main():
    GET_SQUAD = False
    GET_COVID = True
    
    if GET_SQUAD:
        download_squad(version=2)
    if GET_COVID:
        download_covidQA()

        