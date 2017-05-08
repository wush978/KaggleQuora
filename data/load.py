import csv
import codecs
import numpy as np
from data.text_to_wordlist import text_to_wordlist

def load_train(TRAIN_DATA_FILE) :
  idx = []
  texts_1 = []
  texts_2 = []
  labels = []
  with codecs.open(TRAIN_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
      idx.append(values[0])
      texts_1.append(text_to_wordlist(values[4]))
      texts_2.append(text_to_wordlist(values[5]))
      labels.append(int(values[6]))
  print('Found %s rows from file %s' % (len(texts_1), TRAIN_DATA_FILE))
  return {"idx" : np.array(idx, dtype = int), "t1" : texts_1, "t2" : texts_2, "label" : labels}

def load_test(TEST_DATA_FILE) :
  idx = []
  texts_1 = []
  texts_2 = []
  with codecs.open(TEST_DATA_FILE, encoding='utf-8') as f:
    reader = csv.reader(f, delimiter=',')
    header = next(reader)
    for values in reader:
      idx.append(values[0])
      texts_1.append(text_to_wordlist(values[1]))
      texts_2.append(text_to_wordlist(values[2]))
  print('Found %s rows from file %s' % (len(texts_1), TEST_DATA_FILE))
  return {"idx" : np.array(idx, dtype = int), "t1" : texts_1, "t2" : texts_2}
