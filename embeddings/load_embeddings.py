import numpy as np
from gensim.models import KeyedVectors

def load_embeddings(EMBEDDING_FILE, EMBEDDING_DIM):
  ########################################
  ## index word vectors
  ########################################
  print('Indexing word vectors')
  isbin = EMBEDDING_FILE.endswith(".bin")
  word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=isbin)
  print('Found %s word vectors of word2vec' % len(word2vec.vocab))
  return word2vec

def prepare_embedding(MAX_NB_WORDS, EMBEDDING_DIM, word_index, word2vec) :
  nb_words = min(MAX_NB_WORDS, len(word_index))+1
  embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
  for word, i in word_index.items():
    if word in word2vec.vocab:
      embedding_matrix[i] = word2vec.word_vec(word)
  print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
  return embedding_matrix
