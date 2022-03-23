from email.policy import strict
import glob
import re
import json
import nltk
import xml.etree.ElementTree as ET
from nltk import WordPunctTokenizer

ABSTRACT = 'abstract'
N_CITATION = 'n_citation'
BASE_PROPERTY_LIST = [ABSTRACT, N_CITATION]


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
def get_data_property(data: list, prop: str = ABSTRACT) -> list:
    """Gets a specific property from a dataset of JSONs

    Args:
        data (list): list of json objects
        property (str, optional): property to extract. Can use "abstract" or "n_citation". Defaults to "abstract".

    Returns:
        list: list of the requested property
    """
    try:
        return [d[prop] for d in data]
    except KeyError:
        print('Missing key!')
        properties = []
        for d in data:
            if prop in d:
                properties.append(d[prop])
            else:
                properties.append(None)
        return properties

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
def load_datafile(filename: str, limit: int = -1, strict_loading_list=BASE_PROPERTY_LIST) -> list:
    with open(filename, 'r') as f:
        data = f.readlines()

    data_dicts = [json.loads(d) for d in data]

    valid_data_dicts = []
    for d in data_dicts:
        # checks to see that all required properties are in the data
        missing_property = False
        for p in strict_loading_list:
            if p not in d:
                missing_property = True
        
        if not missing_property:
            valid_data_dicts.append(d)


    if limit > 0:
        return valid_data_dicts[:limit]
    return valid_data_dicts

def load_jsondata(args, strict_loading_list=BASE_PROPERTY_LIST):
    data = None
    proj_dir = args.proj_dir
    if proj_dir + args.data_file[-1] == '/':
        data = []
        for filename in glob.glob(proj_dir + args.data_file + '*.json'):
            data.extend(load_datafile(filename, strict_loading_list=strict_loading_list))
    else:
        data = load_datafile(proj_dir + args.data_file, limit=args.limit, strict_loading_list=strict_loading_list)
    return data

def load_txtdata(args, strict_loading_list=BASE_PROPERTY_LIST):
    with open(args.proj_dir + args.data_file, 'r', encoding="utf8") as f:
        data = f.readlines()

    if ABSTRACT in strict_loading_list and N_CITATION in strict_loading_list:
        data_dicts = [{ABSTRACT: sent, N_CITATION: 1} for sent in data]
    else:
        raise NotImplementedError('This feature is not ready. Please adapt the source code to your use case!')

    return data_dicts

def load_data(args):
    strict_loading_list = args.strict_loading_list if args.strict_loading_list else BASE_PROPERTY_LIST

    data = None
    if '.xml' in args.data_file or (args.data_file[-1] == '/' and args.data_file_type == 'xml'):
        print('Loading XML...')
        data = get_persuasive_pairs_xml(args.data_file)
    elif '.json' in args.data_file or (args.data_file[-1] == '/' and args.data_file_type == 'json'):
        print('Loading JSON...')
        data = load_jsondata(args, strict_loading_list=strict_loading_list)
    else:
        print('Loading TXT...')
        data = load_txtdata(args, strict_loading_list=strict_loading_list)
    return data

'''
Loading Data from 16k persuasiveness dataset and IMDB Corpus
'''
def get_persuasive_pairs_xml(directory: str = '../persuasive_classifier/16k_persuasiveness/data/UKPConvArg1Strict-XML/'):
    '''
    Extracts and compiles the 16k persuasiveness pairs from a directory containing XML files

    Args:
        directory (str): directory to 16k persuasive pairs folder with XML files
    Returns:
        a list of dictionaries with both sentences and the label
    '''
    data = []

    for filename in glob.glob(directory + '*.xml'):
        root = ET.parse(filename).getroot()

        argument_pairs = [type_tag for type_tag in root.findall(
            'annotatedArgumentPair')]

        for argument_pair in argument_pairs:
            sentence_a = argument_pair.find('arg1/text').text
            sentence_b = argument_pair.find('arg2/text').text

            labels = [type_tag.find('value').text for type_tag in argument_pair.findall(
                'mTurkAssignments/mTurkAssignment')]

            label = max(labels, key=labels.count)
            confidence = labels.count(label)/len(labels)

            # labels should be 0 and 1 if no equal arguments
            label = int(label[-1]) - 1 if 'equal' not in label else 0
            if not label:
                continue

            # row = {'label': label, 'sentence_a': sentence_a, 'sentence_b': sentence_b, 'confidence': confidence}
            sentence = sentence_a if not label else sentence_b
            row = {ABSTRACT: sentence, N_CITATION: confidence}
            data.append(row)

    return data
