library(data.table)
library(dtplyr)
library(dplyr)
library(itertools)
df <- fread("data/train.csv")
setkey(df, "id")
folds <- df$id %>%
  sample() %>%
  split(rep(1:5, nrow(df) / 5))
izip(i = 1:5, fold = folds) %>%
  lapply(function(.) {
    .$i
    .$fold
    df.test <- df[.$fold]
    df.train <- df[setdiff(df$id, .$fold)]
    write.csv(df.train, file.path("data", sprintf("train-%d.csv", .$i)))
    write.csv(df.test, file.path("data", sprintf("valid-%d.csv", .$i)))
  })
gc()
