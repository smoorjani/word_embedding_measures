import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')

import glob
import argparse
import numpy as np
from os.path import exists
from sklearn.linear_model import Lasso

from utils.data import load_data, get_word_tokenized_corpus, get_data_property, get_data_chunks
from utils.embeddings import load_fasttext_embedding, get_chunk_embeddings, save_fasttext, load_fasttext
from utils.features import get_speed, get_volume, get_circuitousness

def setup_args(parser):
    parser.add_argument("--model_name", type=str, default='/jet/home/moorjani/word_embedding_measures/ft_model.model', help="File to load data from.")
    parser.add_argument("--data_file", type=str, default='data/dblp-ref-0.json', help="File to load data from.")
    parser.add_argument("--chunk_embs_file", type=str, default='data/chunk_embs.txt', help="File to save/load chunks from.")
    parser.add_argument("--limit", type=int, default=30000, help="Number of abstracts to train on")
    # TODO: allow user to create chunks based on length (e.g. I want a chunk to be a sentence or I want a chunk to be 5 tokens)
    parser.add_argument("--T", type=int, default=20, help="Number of chunks to make")
    

def setup():
    parser = argparse.ArgumentParser(description='Process some integers.')
    setup_args(parser)
    args = vars(parser.parse_args())

    stemmer = WordNetLemmatizer()
    en_stop = set(stopwords.words('english'))

    print('Loading data...')
    data = None
    if args['data_file'][-1] == '/':
        data = []
        for filename in glob.glob(args['data_file'] + '*.json'):
            data.extend(load_data(filename))
    else:
        data = load_data(args['data_file'], limit=args['limit'])

    abstracts = get_data_property(data, "abstract")
    citation_counts = get_data_property(data, "n_citation")

    ft_model = None
    if exists(args['model_name']):
        print('Retrieving model...')
        ft_model = load_fasttext(args['model_name'])
    else:
        # tokenize data and train fasttext model
        print('Training model...')
        tokenized_data = get_word_tokenized_corpus(abstracts, stemmer, en_stop)
        ft_model = load_fasttext_embedding(tokenized_data) 
        save_fasttext(ft_model, args['model_name'])

    return args, ft_model, abstracts, citation_counts


def setup_chunk_embeddings(args, ft_model, abstracts):

    if exists(args['chunk_embs_file']):
        print('Loading chunk embeddings...')
        with open(args['chunk_embs_file'], 'r+') as f:
            shape = tuple(map(int, f.readline()[1:].split(',')))
            chunk_embs = np.loadtxt(f, skiprows=0, delimiter=',').reshape(shape)
    else:
        # should be of size [# abstracts, T, len(abstract)/T]
        print('Chunking abstracts...')
        chunks = [get_data_chunks(abstract, T=args['T']) for abstract in abstracts]
        chunk_embs = np.array([get_chunk_embeddings(ft_model, chunk) for chunk in chunks])

        header = ','.join(map(str, chunk_embs.shape))
        np.savetxt('test.txt', chunk_embs.reshape(-1, chunk_embs.shape[-1]), header=header, delimiter=',')

    return chunk_embs

if __name__ == "__main__":
    args, ft_model, abstracts, citation_counts = setup()
    chunk_embs = setup_chunk_embeddings(args, ft_model, abstracts)
    
    print('Getting features...')
    features = {}
    features['speed'] = [get_speed(chunk_emb)[-1] for chunk_emb in chunk_embs]
    # TODO: check formulation to make sure epsilon parameter is OK
    features['circuitousness'] = [get_circuitousness(chunk_emb) for chunk_emb in chunk_embs]
    # TODO: normalize the values/cross-check with their description of implementation
    features['volume'] = [get_volume(chunk_emb) for chunk_emb in chunk_embs]

    print('Getting coefficients...')
    for key, value in features.items():
        clf = Lasso(alpha=0.1)
        clf.fit(np.array(citation_counts).reshape(-1, 1),value)
        print(f'{key} coeff {clf.coef_}')



