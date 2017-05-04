library(data.table)
argv <- commandArgs(TRUE)
dt <- fread(argv[1])
cat(dt[[argv[2]]], sep = "\n")
