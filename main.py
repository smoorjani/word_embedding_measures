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
from sklearn.linear_model import Lasso

from utils.data import load_data, get_word_tokenized_corpus, get_data_property, get_data_chunks
from utils.embeddings import load_fasttext_embedding, get_chunk_embeddings
from utils.features import get_speed, get_volume, get_circuitousness

def setup_args(parser):
    # TODO: allow multiple files to be used. Maybe use a data directory.
    parser.add_argument("--data_file", type=str, default='data/dblp-ref-0.json', help="File to load data from.")
    # TODO: allow user to create chunks based on length (e.g. I want a chunk to be a sentence or I want a chunk to be 5 tokens)
    parser.add_argument("--T", type=int, default=20, help="Number of chunks to make")

if __name__ == "__main__":
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
        data = load_data(args['data_file'])

    abstracts = get_data_property(data, "abstract")
    citation_counts = get_data_property(data, "n_citation")

    # tokenize data and train fasttext model
    print('Training model...')
    tokenized_data = get_word_tokenized_corpus(abstracts, stemmer, en_stop)
    ft_model = load_fasttext_embedding(tokenized_data) 

    # should be of size [# abstracts, T, len(abstract)/T]
    print('Chunking abstracts...')
    chunks = [get_data_chunks(abstract, T=args['T']) for abstract in abstracts]
    chunk_embs = [get_chunk_embeddings(ft_model, chunk) for chunk in chunks]

    # TODO: get features like speed, volume, and circuitousness
    print('Getting features...')
    features = {}
    features['speed'] = [get_speed(chunk_emb)[-1] for chunk_emb in chunk_embs]
    features['volume'] = [get_volume(chunk_emb) for chunk_emb in chunk_embs]
    features['circuitousness'] = [get_circuitousness(chunk_emb) for chunk_emb in chunk_embs]

    for key, value in features.items():
        clf = Lasso(alpha=0.1)
        clf.fit(np.array(citation_counts).reshape(-1, 1),value)
        print(f'{key} coeff {clf.coef_}')



