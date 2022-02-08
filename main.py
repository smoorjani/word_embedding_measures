from pkgutil import get_data
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
import argparse

from utils.data import load_data, get_word_tokenized_corpus, get_data_property, get_data_chunks
from utils.embeddings import load_fasttext_embedding, get_chunk_embeddings
from utils.features import get_speed, get_volume, get_circuitousness

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    args = parser.parse_args()

    stemmer = WordNetLemmatizer()
    en_stop = set(stopwords.words('english'))

    data = load_data(args['data_file'])
    abstracts = get_data_property(data, "abstract")
    citation_counts = get_data_property(data, "n_citation")

    # tokenize data and train fasttext model
    tokenized_data = get_word_tokenized_corpus(abstracts, stemmer, en_stop)
    ft_model = load_fasttext_embedding(tokenized_data) 

    # should be of size [# abstracts, T, len(abstract)/T]
    chunks = [get_data_chunks(abstract, T=args['T']) for abstract in abstracts]
    chunk_embs = [get_chunk_embeddings(ft_model, chunk) for chunk in chunks]

    # TODO: get features like speed, volume, and circuitousness
    speeds = [get_speed(chunk_emb) for chunk_emb in chunk_embs]
    volumes = [get_volume(chunk_emb) for chunk_emb in chunk_embs]
    circuitousness = [get_circuitousness(chunk_emb) for chunk_emb in chunk_embs]



def setup_args(parser):
    # TODO: allow multiple files to be used. Maybe use a data directory.
    parser.add_argument("--data_file", type=str, default='data/dblp-ref-0.json', help="File to load data from.")
    # TODO: allow user to create chunks based on length (e.g. I want a chunk to be a sentence or I want a chunk to be 5 tokens)
    parser.add_argument("--T", type=int, default=20, help="Number of chunks to make")