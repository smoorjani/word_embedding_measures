import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('omw-1.4')

import argparse
import numpy as np
from os.path import exists
from collections import defaultdict
from sklearn.linear_model import Lasso

from utils.data import load_data, get_word_tokenized_corpus, get_data_property, get_data_chunks
from utils.embeddings import train_fasttext_embedding, pretrain_fasttext_embedding, get_chunk_embeddings, save_fasttext, load_fasttext
from utils.features import get_speed, get_volume, get_circuitousness, get_all_features

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='fasttext_model/wiki-news-300d-1M.vec', help="File to load model from.")
    parser.add_argument("--train_model", action='store_true', help="Whether to train existing model on data.")

    parser.add_argument("--data_file", type=str, default='data/dblp-ref-0.json', help="File to load data from.")
    parser.add_argument("--chunk_embs_file", type=str, default='data/chunk_embs.txt', help="File to save/load chunks from.")
    parser.add_argument("--proj_dir", type=str, default='./saved/', help="Directory storing all data/models.")

    parser.add_argument("--limit", type=int, default=30000, help="Number of abstracts to train on")
    # TODO: allow user to create chunks based on length (e.g. I want a chunk to be a sentence or I want a chunk to be 5 tokens)
    parser.add_argument("--T", type=int, default=20, help="Number of chunks to make")
    args = parser.parse_args()
    return args
    

def setup(args):
    proj_dir = args.proj_dir
    stemmer = WordNetLemmatizer()
    en_stop = set(stopwords.words('english'))

    print('Loading data...')
    data = load_data(args)
    abstracts = get_data_property(data, "abstract")
    citation_counts = get_data_property(data, "n_citation")

    ft_model = None
    if exists(proj_dir + args.model_name):
        print('Retrieving model...')
        ft_model = load_fasttext(proj_dir + args.model_name)

        if args.train_model:
            print('Training model...')
            tokenized_data = get_word_tokenized_corpus(abstracts, stemmer, en_stop)
            ft_model = train_fasttext_embedding(ft_model, tokenized_data)
            save_fasttext(ft_model, proj_dir + args.model_name.replace('.', '-trained.'))
    else:
        # tokenize data and train fasttext model
        print('Training model...')
        tokenized_data = get_word_tokenized_corpus(abstracts, stemmer, en_stop)
        ft_model = pretrain_fasttext_embedding(tokenized_data) 
        save_fasttext(ft_model, proj_dir + args.model_name)

    return ft_model, abstracts, citation_counts


def setup_chunk_embeddings(args, ft_model, abstracts):

    if exists(args.proj_dir + args.chunk_embs_file):
        print('Loading chunk embeddings...')
        with open(args.proj_dir + args.chunk_embs_file, 'r+') as f:
            shape = tuple(map(int, f.readline()[1:].split(',')))
            chunk_embs = np.loadtxt(f, skiprows=0, delimiter=',', max_rows=args.limit * args.T).reshape(args.limit, shape[1], shape[2])
    else:
        # should be of size [# abstracts, T, len(abstract)/T]
        print('Chunking abstracts...')
        chunks = [get_data_chunks(abstract, T=args.T) for abstract in abstracts]
        chunk_embs = np.array([get_chunk_embeddings(ft_model, chunk) for chunk in chunks])

        header = ','.join(map(str, chunk_embs.shape))
        np.savetxt(args.proj_dir + 'test.txt', chunk_embs.reshape(-1, chunk_embs.shape[-1]), header=header, delimiter=',')

    return chunk_embs

if __name__ == "__main__":
    args = setup_args()
    if exists(args.proj_dir + args.chunk_embs_file):
        print('Found existing chunks, loading data...')
        data = load_data(args)
        abstracts = get_data_property(data, "abstract")
        citation_counts = get_data_property(data, "n_citation")
        chunk_embs = setup_chunk_embeddings(args, None, None)
    else:
        print('No existing chunks, calling setup...')
        ft_model, abstracts, citation_counts = setup(args)
        chunk_embs = setup_chunk_embeddings(args, ft_model, abstracts)
    
    print('Getting features...')
    features = [get_all_features(chunk_emb) for chunk_emb in chunk_embs]
    features_dict = defaultdict(list)

    for d in features:
        for k, v in d.items():
            features_dict[k].append(v)

    print('Getting coefficients...')
    for key, value in features_dict.items():
        clf = Lasso(alpha=0.1)
        nan_vals = np.argwhere(np.isnan(value))

        non_nan_citations = np.delete(np.array(citation_counts), nan_vals).reshape(-1, 1)
        non_nan_vals = np.delete(value, nan_vals)

        print(non_nan_citations, non_nan_vals)

        clf.fit(non_nan_citations, non_nan_vals)
        print(f'{key} coeff {clf.coef_}')

