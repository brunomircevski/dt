# rpart adapter: fit one protocol on a prepared dataset, print one JSON line.
#
#   Rscript rpart_fit.R data=DIR n_train=N n_test=M n_features=F warmup=ROWS \
#       eval=0|1 mode=fit|alpha maxdepth=D [alpha=A]
#
# warmup=ROWS: one untimed fit on the first ROWS training rows first (0 = none).
# Timed: the rpart() call, plus for mode=alpha pruning to `alpha` (in ./tree's
# units: a leaf's cost as a misclassification rate). No cross-validation
# (xval = 0). Loading and prediction are not timed; with eval=1 the training
# and test accuracy are measured after the timed fit.
#
# Gini; split down to single-row leaves (minsplit 2, minbucket 1, cp 0); no
# competitor or surrogate splits (rpart's defaults compute 4 and 5 per node,
# which no other tool here does).
suppressMessages(library(rpart))

args <- list()
for (arg in commandArgs(trailingOnly = TRUE)) {
  kv <- strsplit(arg, "=", fixed = TRUE)[[1]]
  args[[kv[1]]] <- paste(kv[-1], collapse = "=")
}
n_features <- as.integer(args$n_features)
mode <- args$mode

read_part <- function(part, rows) {
  con <- file(file.path(args$data, paste0(part, ".f32")), "rb")
  values <- readBin(con, "numeric", n = rows * n_features, size = 4, endian = "little")
  close(con)
  con <- file(file.path(args$data, paste0(part, ".y.i32")), "rb")
  labels <- readBin(con, "integer", n = rows, size = 4, endian = "little")
  close(con)
  frame <- as.data.frame(matrix(values, nrow = rows, byrow = TRUE))
  names(frame) <- paste0("f", seq_len(n_features) - 1)
  frame$class <- factor(labels)
  frame
}

control <- rpart.control(minsplit = 2, minbucket = 1, cp = 0, maxcompete = 0,
                         maxsurrogate = 0, usesurrogate = 0,
                         xval = 0,
                         maxdepth = as.integer(args$maxdepth))

fit <- function(frame) {
  model <- rpart(class ~ ., data = frame, method = "class", parms = list(split = "gini"),
                 control = control)
  if (mode == "alpha") {
    # cp is alpha relative to the root's risk (its misclassified rows):
    # alpha * n errors per leaf = cp * root risk.
    model <- prune(model, cp = as.numeric(args$alpha) * nrow(frame) / model$frame$dev[1])
  }
  model
}

train <- read_part("train", as.integer(args$n_train))
warmup <- as.integer(args$warmup)
if (warmup > 0) invisible(fit(train[seq_len(min(warmup, nrow(train))), ]))
start <- Sys.time()
model <- fit(train)
seconds <- as.numeric(difftime(Sys.time(), start, units = "secs"))

leaf <- model$frame$var == "<leaf>"
result <- sprintf(paste0('"train_seconds": %.9f, "nodes": %d, "leaves": %d, "depth": %d, ',
                         '"n_train_loaded": %d, "n_features_loaded": %d'),
                  seconds, nrow(model$frame), sum(leaf),
                  max(floor(log2(as.numeric(rownames(model$frame))))),
                  nrow(train), ncol(train) - 1L)
accuracy <- function(frame) {
  mean(as.character(predict(model, frame, type = "class")) == as.character(frame$class))
}
if (args$eval == "1") {
  result <- paste0(result, sprintf(', "train_accuracy": %.9f', accuracy(train)))
  test <- read_part("test", as.integer(args$n_test))
  result <- paste0(result, sprintf(', "test_accuracy": %.9f', accuracy(test)))
}
cat("{", result, "}\n", sep = "")
