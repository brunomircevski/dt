# rpart for the pruned-accuracy benchmark: grow one CART tree (Gini, cp 0,
# minsplit 2, minbucket 1, no competitor or surrogate splits, xval 0: as
# bench/tools/adapters/rpart_fit.R), report it as grown and pruned by rpart's
# own prune() at cp = alpha * n / root risk (alpha in ./tree's units, the
# misclassification rate; the same minimal cost-complexity rule).
#
#   Rscript fit_rpart.R STEM_TRAIN STEM_TEST N_TRAIN N_TEST N_FEATURES DEPTH ALPHA OUT_JSON PRED_PREFIX
#
# Writes OUT_JSON (two records) and PRED_PREFIX-{unpruned,pruned}.i16 (int16
# test predictions).
# RPART_LIB: a library holding rpart built with rpart-64bit-index.patch, for
# more than 2^31 - 1 values (stock rpart's 32-bit offsets overflow there)
patched <- nzchar(Sys.getenv("RPART_LIB"))
if (patched) {
  suppressMessages(library(rpart, lib.loc = Sys.getenv("RPART_LIB")))
} else {
  suppressMessages(library(rpart))
}
a <- commandArgs(trailingOnly = TRUE)
train_stem <- a[1]; test_stem <- a[2]
n_train <- as.integer(a[3]); n_test <- as.integer(a[4]); n_features <- as.integer(a[5])
depth <- as.integer(a[6]); alpha <- as.numeric(a[7]); out_json <- a[8]; prefix <- a[9]
if (!patched && as.numeric(n_train) * n_features > .Machine$integer.max) {
  stop(sprintf("rpart cannot fit %d rows x %d features: over 2^31 - 1 values", n_train, n_features))
}

read_part <- function(stem, rows) {
  con <- file(paste0(stem, ".f32"), "rb")
  columns <- lapply(seq_len(n_features), function(j) numeric(rows))
  chunk <- max(1L, as.integer(2^24 %/% n_features))
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
  con <- file(paste0(stem, ".y.i32"), "rb")
  labels <- readBin(con, "integer", n = rows, size = 4, endian = "little")
  close(con)
  frame <- structure(columns, names = paste0("f", seq_len(n_features) - 1),
                     class = "data.frame", row.names = c(NA_integer_, -rows))
  frame$class <- labels
  frame
}

train <- read_part(train_stem, n_train)
train$class <- factor(train$class)
control <- rpart.control(minsplit = 2, minbucket = 1, cp = 0, maxcompete = 0, maxsurrogate = 0,
                         usesurrogate = 0, xval = 0, maxdepth = depth)
start <- Sys.time()
grown <- rpart(class ~ ., data = train, method = "class", parms = list(split = "gini"), control = control)
fit_seconds <- as.numeric(difftime(Sys.time(), start, units = "secs"))
cp <- alpha * nrow(train) / grown$frame$dev[1]
pruned <- prune(grown, cp = cp)
test <- read_part(test_stem, n_test)

record <- function(model, is_pruned) {
  # in slices of 1M rows: predict() copies its rows into a matrix of doubles
  # and an is.na() matrix (MNIST8M's training set: 51 + 25 GB at once)
  predict_int <- function(frame) {
    unlist(lapply(split(seq_len(nrow(frame)), (seq_len(nrow(frame)) - 1L) %/% 1000000L), function(rows)
      as.integer(as.character(predict(model, frame[rows, , drop = FALSE], type = "class")))),
      use.names = FALSE)
  }
  pred <- predict_int(test)
  file <- paste0(prefix, if (is_pruned) "-pruned.i16" else "-unpruned.i16")
  con <- file(file, "wb"); writeBin(pred, con, size = 2, endian = "little"); close(con)
  leaf <- model$frame$var == "<leaf>"
  sprintf(paste0('{"tool": "rpart", "algo": "cart", "version": "rpart %s", "depth_limit": %d, ',
                 '"pruned": %s, "prune": "%s", "nodes": %d, "leaves": %d, "depth": %d, ',
                 '"train_accuracy": %.9f, "test_accuracy": %.9f, "n_train": %d, "n_test": %d, ',
                 '"fit_seconds": %.3f, "pred": "%s"}'),
          paste0(packageVersion("rpart"), if (patched) " (64-bit index patch)" else ""), depth, if (is_pruned) "true" else "false",
          if (is_pruned) sprintf("prune(cp = %.6g) = alpha %g (misclassification)", cp, alpha) else "none",
          nrow(model$frame), sum(leaf), max(floor(log2(as.numeric(rownames(model$frame))))),
          mean(predict_int(train) == as.integer(as.character(train$class))), mean(pred == test$class),
          n_train, n_test, fit_seconds, file)
}
out <- paste0("[", record(grown, FALSE), ",\n", record(pruned, TRUE), "]\n")
cat(out, file = out_json)
cat(out)
