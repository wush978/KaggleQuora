import csv
import codecs

def load_train(TRAIN_DATA_FILE):
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
  print('Found %s rows from file %s' % (len(texts_1), TRAIN_DATA_FILE))
  return {"t1" : texts_1, "t2" : texts_2, "label" : labels}
