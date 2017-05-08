import numpy as np
import pdb

def get_cv_index(size, CV_SIZE, VALID_RATIO, has_test) :
  perm = np.random.permutation(size)
  cv_indexes = np.array_split(perm, CV_SIZE)
  result = []
  for i in range(CV_SIZE) :
    idx_train = np.empty((0,), dtype = cv_indexes[0].dtype)
    idx_test = np.empty((0,), dtype = cv_indexes[0].dtype)
    for j in range(CV_SIZE) :
      if i != j :
        idx_train = np.concatenate((idx_train, cv_indexes[j]))
      else :
        idx_test = np.concatenate((idx_test, cv_indexes[j]))
    result.append({
      "idx_train" : idx_train[:int(len(idx_train) * (1 - VALID_RATIO))],
      "idx_valid" : idx_train[int(len(idx_train) * (1 - VALID_RATIO)):],
      "idx_test" : idx_test
    })
  if has_test : 
    idx_train = range(size)
    idx_test = None
    result.append({
      "idx_train" : idx_train[:int(len(idx_train) * (1 - VALID_RATIO))],
      "idx_valid" : idx_train[int(len(idx_train) * (1 - VALID_RATIO)):],
      "idx_test" : idx_test
    })
  return result
