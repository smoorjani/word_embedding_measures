import glob
import re
import json
import nltk
from nltk import WordPunctTokenizer
from torch import chunk

def preprocess_text(document: str, stemmer: nltk.stem.WordNetLemmatizer, en_stop: set) -> str:
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(document))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    tokens = document.split()
    tokens = [stemmer.lemmatize(word) for word in tokens]
    tokens = [word for word in tokens if word not in en_stop]
    tokens = [word for word in tokens if len(word) > 3]

    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def get_word_tokenized_corpus(abstracts: list, stemmer: nltk.stem.WordNetLemmatizer, en_stop: set) -> list:
    final_corpus = [preprocess_text(abstract, stemmer, en_stop) for abstract in abstracts if abstract.strip() !='']
    word_punctuation_tokenizer = WordPunctTokenizer()
    return [word_punctuation_tokenizer.tokenize(sent) for sent in final_corpus]

'''
Interacting with Data Dicts
'''
def get_data_property(data: list, property: str = "abstract") -> list:
    """Gets a specific property from a dataset of JSONs

    Args:
        data (list): list of json objects
        property (str, optional): property to extract. Can use "abstract" or "n_citation". Defaults to "abstract".

    Returns:
        list: list of the requested property
    """
    return [d[property] for d in data]

def get_data_chunks(abstract: str, T: int = 20) -> list:
    tokens = abstract.split(" ")
    chunk_len = len(tokens) / T
    
    chunks = []
    for i in range(T):
        min_idx = int(i * chunk_len)
        max_idx = int(min(len(tokens), (i+1) * chunk_len))
        chunk = tokens[min_idx:max_idx]
        chunks.append(chunk)

    return chunks

'''
Loading Data from Files
'''
def load_datafile(filename: str, limit: int = -1) -> list:
    with open(filename, 'r') as f:
        data = f.readlines()

    data_dicts = [json.loads(d) for d in data]
    valid_data_dicts = [d for d in data_dicts if 'abstract' in d and 'n_citation' in d]
    if limit > 0:
        return valid_data_dicts[:limit]
    return valid_data_dicts

def load_jsondata(args):
    data = None
    proj_dir = args.proj_dir
    if proj_dir + args.data_file[-1] == '/':
        data = []
        for filename in glob.glob(proj_dir + args.data_file + '*.json'):
            data.extend(load_datafile(filename))
    else:
        data = load_datafile(proj_dir + args.data_file, limit=args.limit)
    return data

def load_txtdata(args):
    with open(args.proj_dir + args.data_file, 'r', encoding="utf8") as f:
        data = f.readlines()
    data_dicts = [{'abstract': sent, 'n_citation': 1} for sent in data]
    return data_dicts

def load_data(args):
    data = None
    if '.json' in args.data_file or args.data_file[-1] == '/':
        data = load_jsondata(args)
    else:
        data = load_txtdata(args)
    return data

