.PHONY=valid

all :

valid : train-fold.R data/train.csv
	Rscript train-fold.R
