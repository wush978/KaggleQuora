'''
Single model may achieve LB scores at around 0.29+ ~ 0.30+
Average ensembles can easily get 0.28+ or less
Don't need to be an expert of feature engineering
All you need is a GPU!!!!!!!

The code is tested on Keras 2.0.0 using Tensorflow backend, and Python 2.7

According to experiments by kagglers, Theano backend with GPU may give bad LB scores while
        the val_loss seems to be fine, so try Tensorflow backend first please
'''

########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from data.text_to_wordlist import text_to_wordlist
from sklearn.metrics import log_loss

import sys
import importlib
importlib.reload(sys)

########################################
## set directories and parameters
########################################
def main(EMBEDDING_FILE, TRAIN_DATA_FILE, VALID_DATA_FILE, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EMBEDDING_DIM, VALIDATION_SPLIT, BATCH_SIZE) :
  # EMBEDDING_FILE = BASE_DIR + 'GoogleNews-vectors-negative300.bin'
  # TRAIN_DATA_FILE = BASE_DIR + 'train.csv'
  # TEST_DATA_FILE = BASE_DIR + 'test.csv'
  # MAX_SEQUENCE_LENGTH = 30
  # MAX_NB_WORDS = 200000
  # EMBEDDING_DIM = 300
  # VALIDATION_SPLIT = 0.1
  num_lstm = np.random.randint(175, 275)
  num_dense = np.random.randint(100, 150)
  rate_drop_lstm = 0.15 + np.random.rand() * 0.25
  rate_drop_dense = 0.15 + np.random.rand() * 0.25
  act = 'relu'
  re_weight = True # whether to re-weight classes to fit the 17.5% share in test set
  STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
          rate_drop_dense)
  ########################################
  ## index word vectors
  ########################################
  print('Indexing word vectors')
  word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, \
        binary=False)
  print('Found %s word vectors of word2vec' % len(word2vec.vocab))
  ########################################
  ## process texts in datasets
  ########################################
  print('Processing text dataset')
  # The function "text_to_wordlist" is from
  # https://www.kaggle.com/currie32/quora-question-pairs/the-importance-of-cleaning-text
  texts_1 = []
  texts_2 = []
  labels = []
  with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
      texts_1.append(text_to_wordlist(values[4]))
      texts_2.append(text_to_wordlist(values[5]))
      labels.append(int(values[6]))
  print('Found %s texts in training dataset' % len(texts_1))
  valid_texts_1 = []
  valid_texts_2 = []
  valid_labels = []
  with codecs.open(VALID_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
      valid_texts_1.append(text_to_wordlist(values[4]))
      valid_texts_2.append(text_to_wordlist(values[5]))
      valid_labels.append(int(values[6]))
  print('Found %s texts in valid dataset' % len(valid_texts_1))
  tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
  tokenizer.fit_on_texts(texts_1 + texts_2 + valid_texts_1 + valid_texts_2)
  sequences_1 = tokenizer.texts_to_sequences(texts_1)
  sequences_2 = tokenizer.texts_to_sequences(texts_2)
  valid_sequences_1 = tokenizer.texts_to_sequences(valid_texts_1)
  valid_sequences_2 = tokenizer.texts_to_sequences(valid_texts_2)
  word_index = tokenizer.word_index
  print('Found %s unique tokens' % len(word_index))
  data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
  data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
  labels = np.array(labels)
  print('Shape of data tensor:', data_1.shape)
  print('Shape of label tensor:', labels.shape)
  valid_data_1 = pad_sequences(valid_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
  valid_data_2 = pad_sequences(valid_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
  ########################################
  ## prepare embeddings
  ########################################
  print('Preparing embedding matrix')
  nb_words = min(MAX_NB_WORDS, len(word_index))+1
  embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
  for word, i in word_index.items():
    if word in word2vec.vocab:
      embedding_matrix[i] = word2vec.word_vec(word)
  print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
  
  ########################################
  ## sample train/validation data
  ########################################
  #np.random.seed(1234)
  perm = np.random.permutation(len(data_1))
  idx_train = perm[:int(len(data_1)*(1-VALIDATION_SPLIT))]
  idx_val = perm[int(len(data_1)*(1-VALIDATION_SPLIT)):]
  
  data_1_train = np.vstack((data_1[idx_train], data_2[idx_train]))
  data_2_train = np.vstack((data_2[idx_train], data_1[idx_train]))
  labels_train = np.concatenate((labels[idx_train], labels[idx_train]))
  
  data_1_val = np.vstack((data_1[idx_val], data_2[idx_val]))
  data_2_val = np.vstack((data_2[idx_val], data_1[idx_val]))
  labels_val = np.concatenate((labels[idx_val], labels[idx_val]))
  
  weight_val = np.ones(len(labels_val))
  if re_weight:
    weight_val *= 0.472001959
    weight_val[labels_val==0] = 1.309028344
  ########################################
  ## define the model structure
  ########################################
  embedding_layer = Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False)
  lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)
  sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
  embedded_sequences_1 = embedding_layer(sequence_1_input)
  x1 = lstm_layer(embedded_sequences_1)
  sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
  embedded_sequences_2 = embedding_layer(sequence_2_input)
  y1 = lstm_layer(embedded_sequences_2)
  merged = concatenate([x1, y1])
  merged = Dropout(rate_drop_dense)(merged)
  merged = BatchNormalization()(merged)
  merged = Dense(num_dense, activation=act)(merged)
  merged = Dropout(rate_drop_dense)(merged)
  merged = BatchNormalization()(merged)
  preds = Dense(1, activation='sigmoid')(merged)
  
  ########################################
  ## add class weight
  ########################################
  if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
  else:
    class_weight = None
  ########################################
  ## train the model
  ########################################
  model = Model(inputs=[sequence_1_input, sequence_2_input], \
          outputs=preds)
  model.compile(loss='binary_crossentropy',
          optimizer='nadam',
          metrics=['acc'])
  #model.summary()
  print(STAMP)
  early_stopping =EarlyStopping(monitor='val_loss', patience=3)
  bst_model_path = STAMP + '.h5'
  model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
  hist = model.fit([data_1_train, data_2_train], labels_train, \
        validation_data=([data_1_val, data_2_val], labels_val, weight_val), \
        epochs=200, batch_size=BATCH_SIZE, shuffle=True, \
        class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])
  model.load_weights(bst_model_path)
  bst_val_score = min(hist.history['val_loss'])
  ########################################
  ## make the submission
  ########################################
  # print('Start making the submission before fine-tuning')
  # preds = model.predict([test_data_1, test_data_2], batch_size=8192, verbose=1)
  # preds += model.predict([test_data_2, test_data_1], batch_size=8192, verbose=1)
  # preds /= 2
  # submission = pd.DataFrame({'test_id':test_ids, 'is_duplicate':preds.ravel()})
  # submission.to_csv('%.4f_'%(bst_val_score)+STAMP+'.csv', index=False)
  preds = model.predict([valid_data_1, valid_data_2], batch_size=BATCH_SIZE, verbose=1)
  preds += model.predict([valid_data_2, valid_data_1], batch_size=BATCH_SIZE, verbose=1)
  preds /= 2
  print("Result: ")
  print(log_loss(valid_labels, preds))

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description = 'LSTM + WordEnbedding')
  # EMBEDDING_FILE, TRAIN_DATA_FILE, VALID_DATA_FILE, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EMBEDDING_DIM, VALIDATION_SPLIT
  parser.add_argument('EMBEDDING_FILE')
  parser.add_argument('TRAIN_DATA_FILE')
  parser.add_argument('VALID_DATA_FILE')
  parser.add_argument('MAX_SEQUENCE_LENGTH', type = int)
  parser.add_argument('MAX_NB_WORDS', type = int)
  parser.add_argument('EMBEDDING_DIM', type = int)
  parser.add_argument('VALIDATION_SPLIT', nargs = "?", default = 0.1, type = float)
  parser.add_argument('BATCH_SIZE', type = int)
  args = parser.parse_args()
  main(args.EMBEDDING_FILE, args.TRAIN_DATA_FILE, args.VALID_DATA_FILE, args.MAX_SEQUENCE_LENGTH, args.MAX_NB_WORDS, args.EMBEDDING_DIM, args.VALIDATION_SPLIT, args.BATCH_SIZE)
