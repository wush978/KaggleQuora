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
import gzip
import numpy as np
import pandas as pd

from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from data.load import load_train, load_test
from embeddings.load_embeddings import load_embeddings, prepare_embedding
from models.text_preprocess import tokenize, get_word_index, get_pad_sequences
from models.cv import get_cv_index
from sklearn.metrics import log_loss

import sys
import importlib
importlib.reload(sys)

import pdb

########################################
## set directories and parameters
########################################
def main(EMBEDDING_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EMBEDDING_DIM, BATCH_SIZE, CV_SIZE, VALID_RATIO, NLAYERS) :
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
  STAMP = 'tmp/lstm-%d-layers_%d_%d_%.2f_%.2f'%(NLAYERS, num_lstm, num_dense, rate_drop_lstm, \
          rate_drop_dense)
  word2vec = load_embeddings(EMBEDDING_FILE, EMBEDDING_DIM)
  ########################################
  ## process texts in datasets
  ########################################
  print('Processing text dataset')
  train = load_train(TRAIN_DATA_FILE)
  has_test = os.path.isfile(TEST_DATA_FILE)
  if has_test :
    test = load_test(TEST_DATA_FILE)
  else :
    test = {}
  tokenizer = tokenize(MAX_NB_WORDS, train, test)
  word_index = get_word_index(tokenizer, train, test)
  get_pad_sequences(MAX_SEQUENCE_LENGTH, train, test)
  if "label" in train :
    train["label"] = np.array(train["label"])
  if "label" in test :
    test["label"] = np.array(test["label"])
  print('Shape of data tensor:', train["d1"].shape)
  print('Shape of label tensor:', train["label"].shape)
  ########################################
  ## prepare embeddings
  ########################################
  print('Preparing embedding matrix')
  nb_words = min(MAX_NB_WORDS, len(word_index))+1
  embedding_matrix = prepare_embedding(MAX_NB_WORDS, EMBEDDING_DIM, word_index, word2vec)
  ########################################
  ## sample train/validation data
  ########################################
  #np.random.seed(1234)
  sample_size = len(train["d1"])
  cv_indexes = get_cv_index(sample_size, CV_SIZE, VALID_RATIO, has_test)
  train["data"] = []
  cv_index_index = 1
  def get_element(idx, idx_train, idx_valid, idx_test, data_idx, data1, data2, label) :
    return {
      "idx" : idx,
      "idx_train" : data_idx[idx_train],
      "idx_valid" : data_idx[idx_valid],
      "idx_test" : data_idx[idx_test] if idx_test is not None else None,
      "d1_train" : np.vstack((data1[idx_train], data2[idx_train])),
      "d2_train" : np.vstack((data2[idx_train], data1[idx_train])),
      "label_train" : np.concatenate((label[idx_train], label[idx_train])),
      "d1_valid" : np.vstack((data1[idx_valid], data2[idx_valid])),
      "d2_valid" : np.vstack((data2[idx_valid], data1[idx_valid])),
      "label_valid" : np.concatenate((label[idx_valid], label[idx_valid])),
      "weight_valid" : np.ones(2 * len(idx_valid)),
      "d1_test" : data1[idx_test] if idx_test is not None else None,
      "d2_test" : data2[idx_test] if idx_test is not None else None,
      "label_test" : label[idx_test] if idx_test is not None else None
    }
  for cv_index in cv_indexes :
    element = get_element(cv_index_index, cv_index["idx_train"], cv_index["idx_valid"], cv_index["idx_test"], 
      train["idx"], train["d1"], train["d2"], train["label"])
    if re_weight:
      element["weight_valid"] *= 0.472001959
      element["weight_valid"][element["label_valid"]==0] = 1.309028344
    if cv_index["idx_test"] is None :
      element["idx"] = 0
      element["d1_test"] = test["d1"]
      element["d2_test"] = test["d2"]
    train["data"].append(element)
    cv_index_index += 1
  ########################################
  ## define the model structure
  ########################################
  def get_model() :
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
    for layer in range(NLAYERS) :
      merged = Dense(num_dense, activation=act)(merged)
      merged = Dropout(rate_drop_dense)(merged)
      merged = BatchNormalization()(merged)
    preds = Dense(1, activation='sigmoid')(merged)
    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], 
            outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer='nadam',
            metrics=['acc'])
  #model.summary()
    early_stopping =EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = ("%s-%d.h5") % (STAMP, element["idx"])
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)
    return {
      "model" : model, 
      "callbacks" : [early_stopping, model_checkpoint], 
      "output" : bst_model_path
    }
  ########################################
  ## add class weight
  ########################################
  if re_weight:
    class_weight = {0: 1.309028344, 1: 0.472001959}
  else:
    class_weight = None
  for element in train["data"] :
    print(("fold: %d") % element["idx"])
    model = get_model()
    hist = model["model"].fit([element["d1_train"], element["d2_train"]], element["label_train"], 
          validation_data=([element["d1_valid"], element["d2_valid"]], element["label_valid"], element["weight_valid"]), 
          epochs=200, batch_size=BATCH_SIZE, shuffle=True, 
          class_weight=class_weight, callbacks=model["callbacks"])
    model["model"].load_weights(model["output"])
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
    preds = model["model"].predict([element["d1_test"], element["d2_test"]], batch_size=BATCH_SIZE, verbose=1)
    preds += model["model"].predict([element["d2_test"], element["d1_test"]], batch_size=BATCH_SIZE, verbose=1)
    preds /= 2
    if element["label_test"] is not None :
      print("Result: ")
      print(log_loss(element["label_test"], preds))
    STAMP_result = STAMP.replace("tmp/", "result/")
    pd.DataFrame({"idx" : element["idx_test"], "preds" : np.squeeze(preds)}).to_csv(("%s-%d.csv.gz") % (STAMP_result, element["idx"]), compression = "gzip", index = False)
    with gzip.open("%s-%d.json.gz" % (STAMP_result, element["idx"]), "wb") as json_file:
      json_file.write(model["model"].to_json().encode())
    model["model"].save_weights("%s-%d.h5" % (STAMP_result, element["idx"]))

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description = 'LSTM + WordEnbedding')
  # EMBEDDING_FILE, TRAIN_DATA_FILE, VALID_DATA_FILE, MAX_SEQUENCE_LENGTH, MAX_NB_WORDS, EMBEDDING_DIM, VALIDATION_SPLIT
  parser.add_argument('EMBEDDING_FILE')
  parser.add_argument('TRAIN_DATA_FILE')
  parser.add_argument('MAX_SEQUENCE_LENGTH', type = int)
  parser.add_argument('MAX_NB_WORDS', type = int)
  parser.add_argument('EMBEDDING_DIM', type = int)
  parser.add_argument('BATCH_SIZE', type = int)
  parser.add_argument('--TEST_DATA_FILE', default = "")
  parser.add_argument('--CV_SIZE', default = 2, type = int)
  parser.add_argument('--VALID_RATIO', default = 0.1, type = float)
  parser.add_argument('--NLAYERS', default = 2, type = int)
  args = parser.parse_args()
  main(args.EMBEDDING_FILE, args.TRAIN_DATA_FILE, args.TEST_DATA_FILE, args.MAX_SEQUENCE_LENGTH, args.MAX_NB_WORDS, args.EMBEDDING_DIM, args.BATCH_SIZE, args.CV_SIZE, args.VALID_RATIO, args.NLAYERS)
