from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# uncomment these lines to download the NLTK packages
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('omw-1.4')

import argparse
import gensim
import numpy as np
from os.path import exists
from typing import List, Dict
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from utils.data import load_data, get_word_tokenized_corpus, get_data_property, get_data_chunks, ABSTRACT, N_CITATION
from utils.embeddings import train_fasttext_embedding, pretrain_fasttext_embedding, get_chunk_embeddings, save_fasttext, load_fasttext
from utils.features import get_features
from utils.controls import get_controls

def setup_args() -> argparse.Namespace:
    """Gets the command-line arguments used. Feel free to add your own here.

    Returns:
        argparse.Namespace: argparse object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='fasttext_model/wiki-news-300d-1M.vec', help="File to load model from.")
    parser.add_argument("--train_model", action='store_true', help="Whether to train existing model on data.")

    parser.add_argument("--data_file", type=str, default='data/dblp-ref-0.json', help="File/directory to load data from.")
    parser.add_argument("--data_file_type", type=str, help="If a directory is provided as the datafile, provide type to read")
    parser.add_argument("--chunk_embs_file", type=str, default='data/chunk_embs.txt', help="File to save/load chunks from.")
    parser.add_argument("--proj_dir", type=str, default='./word_embedding_measures/', help="Directory storing all data/models.")

    parser.add_argument("--limit", type=int, default=30000, help="Number of documents to train on")
    # TODO: allow user to create chunks based on length (e.g. I want a chunk to be a sentence or I want a chunk to be 5 tokens)
    parser.add_argument("--T", type=int, default=20, help="Number of chunks to make")
    args = parser.parse_args()

    if args.data_file[-1] == '/':
        assert args.data_file_type is not None
    return args
    

def setup(args: argparse.Namespace) -> tuple:
    """Sets up the FastText model (either training or just loading) and 
    loads in and preprocesses the data.

    Args:
        args(argparse.Namespace): argparse object

    Returns:
        tuple[gensim.models.fasttext.FastText, List[Dict]]: returns FastText model and data
    """
    proj_dir = args.proj_dir
    stemmer = WordNetLemmatizer()
    en_stop = set(stopwords.words('english'))

    print('Loading data...')
    data = load_data(args)
    documents = get_data_property(data, ABSTRACT)

    ft_model = None
    if exists(proj_dir + args.model_name):
        print('Retrieving model...')
        ft_model = load_fasttext(proj_dir + args.model_name)

        if args.train_model:
            print('Training model...')
            tokenized_data = get_word_tokenized_corpus(documents, stemmer, en_stop)
            ft_model = train_fasttext_embedding(ft_model, tokenized_data)
            save_fasttext(ft_model, proj_dir + args.model_name.replace('.bin', '-trained.bin'))
    else:
        # tokenize data and train fasttext model
        print('Training model...')
        tokenized_data = get_word_tokenized_corpus(documents, stemmer, en_stop)
        ft_model = pretrain_fasttext_embedding(tokenized_data) 
        save_fasttext(ft_model, proj_dir + args.model_name)

    return ft_model, data


def setup_chunk_embeddings(args: argparse.Namespace, ft_model: gensim.models.fasttext.FastText, documents: List[str]) -> np.ndarray:
    """_summary_

    Args:
        args (argparse.Namespace): argparse object
        ft_model (gensim.models.fasttext.FastText): fasttext model
        documents (List[str]): list of documents

    Returns:
        np.ndarray: returns numpy array of chunk embeddings
    """

    if exists(args.proj_dir + args.chunk_embs_file):
        print('Loading chunk embeddings...')
        with open(args.proj_dir + args.chunk_embs_file, 'r+') as f:
            shape = tuple(map(int, f.readline()[1:].split(',')))
            limit = min(shape[0], args.limit)
            chunk_embs = np.loadtxt(f, skiprows=0, delimiter=',', max_rows=limit * args.T).reshape(limit, shape[1], shape[2])
    else:
        # should be of size [# documents, T, len(document)/T]
        print('Chunking documents...')
        chunks = [get_data_chunks(document, T=args.T) for document in documents]
        chunk_embs = np.array([get_chunk_embeddings(ft_model, chunk) for chunk in chunks])

        header = ','.join(map(str, chunk_embs.shape))
        np.savetxt(args.proj_dir + args.chunk_embs_file, chunk_embs.reshape(-1, chunk_embs.shape[-1]), header=header, delimiter=',')

    return chunk_embs

if __name__ == "__main__":
    args = setup_args()

    # additional properties I want from the data
    strict_loading_list = [ABSTRACT, N_CITATION, 'year', 'venue']
    args.strict_loading_list = strict_loading_list

    if exists(args.proj_dir + args.chunk_embs_file):
        print('Found existing chunks, loading data...')
        data = load_data(args)
        chunk_embs = setup_chunk_embeddings(args, None, None)
    else:
        print('No existing chunks, calling setup...')
        ft_model, data = setup(args)
        documents = get_data_property(data, ABSTRACT)
        chunk_embs = setup_chunk_embeddings(args, ft_model, documents)

    # this is the dependent variable we will do regression on
    # you should change this to whatever metric you want
    citation_counts = get_data_property(data, N_CITATION)
    # controls are very important for the regression and should be changed
    # based on your own use case
    controls = get_controls(data)

    print('Getting features...')
    features = [get_features(chunk_emb) for chunk_emb in chunk_embs]
    features_dict = defaultdict(list)

    # move features into a dict that stores lists for every feature
    for d in features:
        for k, v in d.items():
            features_dict[k].append(v)

    '''
    for key, value in features_dict.items():
        print(key)
        print(value)
        print('-'*100)
    '''

    # if there is a missing value at the end of distances (sometimes there are only T-1 chunks)
    for i, l in enumerate(features_dict['distances']):
        for j in range(args.T - 1 - len(l)):
            l.append(np.nan)
            
    avg_distances = np.nanmean(np.array(features_dict['distances']), axis=0, dtype='float32')
    # plots the speed over the chunks
    plt.plot(list(range(args.T-1)), avg_distances)
    plt.savefig('distances.png') 

    print('Getting coefficients...')
    for key, value in features_dict.items():
        if key == 'distances':
            continue
        
        # remove rows with invalid values
        nan_vals = np.argwhere(np.isnan(value))

        controls = np.delete(controls, nan_vals, axis=0)
        non_nan_citations = np.log(1 + np.delete(np.array(citation_counts), nan_vals))
        non_nan_vals = np.log(np.delete(value, nan_vals)).reshape(-1, 1)
        dependent_vars = np.concatenate((non_nan_vals, controls), axis=1)

        clf = LinearRegression()
        clf.fit(dependent_vars, non_nan_citations)
        print(f'{key} coeff {clf.coef_[0]} mean {np.mean(non_nan_vals)}')

