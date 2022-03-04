import io
import numpy as np

import gensim
from gensim.test.utils import get_tmpfile, datapath
from gensim.models.fasttext import FastText, load_facebook_vectors, load_facebook_model 
from gensim.models.keyedvectors import load_word2vec_format

def pretrain_fasttext_embedding(word_tokenized_corpus: list, embedding_size: int = 300, window_size: int = 40, min_word: int = 3, down_sampling: float = 1e-2) -> gensim.models.fasttext.FastText:
    """Pretrains and returns a FastText model

    Args:
        word_tokenized_corpus (list): list of tokenized sentences
        embedding_size (int, optional): size of embedding vector. Defaults to 300.
        window_size (int, optional): number of words to use before/after as context. Defaults to 40.
        min_word (int, optional): minimum frequency of word to include. Defaults to 3.
        down_sampling (float, optional): most frequently occuring word will be downsampled by. Defaults to 1e-2.

    Returns:
        gensim.models.fasttext.FastText: trained FastText model
    """
    ft_model = FastText(word_tokenized_corpus,
                      vector_size=embedding_size,
                      window=window_size,
                      min_count=min_word,
                      sample=down_sampling,
                      sg=1,
                      epochs=100)

    return ft_model


def train_fasttext_embedding(ft_model, word_tokenized_corpus: list, epochs: int = 5) -> gensim.models.fasttext.FastText:
    """Trains and returns a FastText model

    Args:
        ft_model: model to trian
        word_tokenized_corpus (list): list of tokenized sentences
        epochs (int): number of epochs to trian for
    Returns:
        gensim.models.fasttext.FastText: trained FastText model
    """
    ft_model.build_vocab(word_tokenized_corpus, update=True)
    ft_model.train(corpus_iterable=word_tokenized_corpus,
                   total_examples=len(word_tokenized_corpus),
                   epochs=epochs)

    return ft_model


def save_fasttext(ft_model, filename):
    # fname = get_tmpfile(filename)
    ft_model.save(filename)


def load_fasttext(filename):
    if '.bin' in filename:
        return load_facebook_model(filename)
    elif '.vec' in filename:
        return load_word2vec_format(filename)
    else:
        return FastText.load(filename)


def get_chunk_embeddings(ft_model: gensim.models.fasttext.FastText, chunks: list) -> list:
    avg_embs = []
    for chunk in chunks:
        avg_emb = np.zeros((300,))
        if len(chunk):
            embs = []
            for token in chunk:
                try:
                    try:
                        embs.append(np.array(ft_model.wv[token]))
                    except AttributeError:
                        embs.append(np.array(ft_model[token]))
                except KeyError:
                    print(f'{token} not in vocab!')
                    continue
            embs = np.stack(embs)
            avg_emb = np.average(embs, axis=0)
        avg_embs.append(avg_emb)
    return avg_embs
    
