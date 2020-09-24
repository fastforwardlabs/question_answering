# helper functions for downloading various QA datasets
import urllib.request

def download_squad(version=2):
    if version not in [1,2]:
        print("Please specificy SQuAD version number.")
        return 
    
    url_base = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
    data_dir = "/home/cdsw/data/squad/"
      
    if version == 1:
        print("Downloading SQuAD1.1 training and development sets...")
        train = "train-v1.1.json"
        dev = "dev-v1.1.json"
        
    elif version == 2:
        print("Downloading SQuAD2.0 training and development sets...")
        train = "train-v2.0.json"
        dev = "dev-v2.0.json"
        
    urllib.request.urlretrieve(url_base+train, data_dir+train)
    urllib.request.urlretrieve(url_base+dev, data_dir+dev)
    return

def download_covidQA():
    print("Downloading COVID-QA dataset...")
    
    url = "https://github.com/deepset-ai/COVID-QA/raw/master/data/question-answering/COVID-QA.json"
    output_filename = "/home/cdsw/data/covidQA/COVID-QA.json"
    urllib.request.urlretrieve(url, output_filename)
    return

def main():
    GET_SQUAD = True
    GET_COVID = False
    
    if GET_SQUAD:
        download_squad(version=2)
    if GET_COVID:
        download_covidQA()

if __name__ == "__main__":
    main()