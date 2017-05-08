from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def tokenize(MAX_NB_WORDS, *objs) :
  texts = []
  for key in ["t1", "t2"] :
    for obj in objs :
      if key in obj :
        texts += obj[key]
  tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
  tokenizer.fit_on_texts(texts)
  return tokenizer

def get_word_index(tokenizer, *objs) :
  for key in ["t1", "t2"] :
    for obj in objs :
      if key in obj :
        obj[key.replace("t", "s")] = tokenizer.texts_to_sequences(obj[key])
  word_index = tokenizer.word_index
  print('Found %s unique tokens' % len(word_index))
  return word_index

def get_pad_sequences(MAX_SEQUENCE_LENGTH, *objs) :
  for key in ["s1", "s2"] :
    for obj in objs :
      if key in obj :
        obj[key.replace("s", "d")] = pad_sequences(obj[key], maxlen = MAX_SEQUENCE_LENGTH)
  return objs
