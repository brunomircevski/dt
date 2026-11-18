# rpart adapter: fit one protocol on a prepared dataset, print one JSON line.
#
#   Rscript rpart_fit.R train=STEM test=STEM n_train=N n_test=M n_features=F \
#       eval=0|1 mode=fit|alpha maxdepth=D [alpha=A]
#
# One fresh process fits exactly one tree: no warm-up fit before it.
# read_seconds: reading the training rows into a data.frame (not timed).
# Timed: the rpart() call, plus for mode=alpha pruning to `alpha` (in ./tree's
# units: a leaf's cost as a misclassification rate). No cross-validation
# (xval = 0). Loading and prediction are not timed; with eval=1 the training
# and test accuracy are measured after the timed fit. The peak resident memory
# (VmHWM) is read right after the fit, before evaluation.
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

# The rows are read in chunks straight into the frame's columns: one readBin of
# rows * n_features values overflows R's integers on MNIST8M (6.3e9), and a
# whole matrix plus its data.frame copy would hold the data three times.
read_part <- function(part, rows) {
  con <- file(paste0(args[[part]], ".f32"), "rb")  # STEM.f32: train= / test=
  columns <- lapply(seq_len(n_features), function(j) numeric(rows))
  chunk <- max(1L, as.integer(2^24 %/% n_features))  # rows a chunk: ~16M values
  done <- 0L
  while (done < rows) {
    k <- min(chunk, rows - done)
    values <- matrix(readBin(con, "numeric", n = k * n_features, size = 4, endian = "little"),
                     nrow = k, byrow = TRUE)
    at <- (done + 1L):(done + k)
    for (j in seq_len(n_features)) columns[[j]][at] <- values[, j]
    done <- done + k
  }
  close(con)
  con <- file(paste0(args[[part]], ".y.i32"), "rb")
  labels <- readBin(con, "integer", n = rows, size = 4, endian = "little")
  close(con)
  frame <- structure(columns, names = paste0("f", seq_len(n_features) - 1),
                     class = "data.frame", row.names = c(NA_integer_, -rows))
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

# rpart() crashes (segfault in its C code) once rows x features passes 2^31 - 1
# (MNIST8M: 8.1M x 784 = 6.3e9, 2026-11-18): stop with a clear error instead
if (as.numeric(args$n_train) * n_features > .Machine$integer.max) {
  stop(sprintf("rpart cannot fit %s rows x %d features: over 2^31 - 1 values (rpart() segfaults)",
               args$n_train, n_features))
}
start <- Sys.time()
train <- read_part("train", as.integer(args$n_train))
read_seconds <- as.numeric(difftime(Sys.time(), start, units = "secs"))
start <- Sys.time()
model <- fit(train)
seconds <- as.numeric(difftime(Sys.time(), start, units = "secs"))
status <- readLines("/proc/self/status")
peak_kib <- as.numeric(strsplit(trimws(sub("VmHWM:", "", grep("^VmHWM:", status, value = TRUE))),
                                " ")[[1]][1])

leaf <- model$frame$var == "<leaf>"
result <- sprintf(paste0('"train_seconds": %.9f, "read_seconds": %.9f, "nodes": %d, "leaves": %d, "depth": %d, ',
                         '"n_train_loaded": %d, "n_features_loaded": %d, ',
                         '"peak_rss_train_bytes": %.0f'),
                  seconds, read_seconds, nrow(model$frame), sum(leaf),
                  max(floor(log2(as.numeric(rownames(model$frame))))),
                  nrow(train), ncol(train) - 1L, peak_kib * 1024)
accuracy <- function(frame) {
  mean(as.character(predict(model, frame, type = "class")) == as.character(frame$class))
}
if (args$eval == "1") {
  result <- paste0(result, sprintf(', "train_accuracy": %.9f', accuracy(train)))
  if (as.integer(args$n_test) > 0) {  # 0: no test set
    test <- read_part("test", as.integer(args$n_test))
    result <- paste0(result, sprintf(', "test_accuracy": %.9f', accuracy(test)))
  }
}
cat("{", result, "}\n", sep = "")
