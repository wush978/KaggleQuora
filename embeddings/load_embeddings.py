from gensim.models import KeyedVectors

def load_embeddings(EMBEDDING_FILE, EMBEDDING_DIM):
  ########################################
  ## index word vectors
  ########################################
  print('Indexing word vectors')
  word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=False)
  print('Found %s word vectors of word2vec' % len(word2vec.vocab))
  return word2vec
