# rpart adapter: fit one protocol on a prepared dataset, print one JSON line.
#
#   Rscript rpart_fit.R data=DIR n_train=N n_test=M n_features=F warmup=subset|none \
#       eval=0|1 mode=fit|alpha|cv maxdepth=D [alpha=A] [folds=K]
#
# Timed: the rpart() call (for mode=cv with its built-in K-fold CV), plus
# pruning: to `alpha` (mode=alpha, alpha in ./tree's units: a leaf's cost as a
# misclassification rate) or to the 1-SE tree (mode=cv). Loading and
# prediction are not timed.
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
                         xval = if (mode == "cv") as.integer(args$folds) else 0,
                         maxdepth = as.integer(args$maxdepth))

fit <- function(frame) {
  set.seed(1)  # CV fold assignment
  model <- rpart(class ~ ., data = frame, method = "class", parms = list(split = "gini"),
                 control = control)
  if (mode == "alpha") {
    # cp is alpha relative to the root's risk (its misclassified rows):
    # alpha * n errors per leaf = cp * root risk.
    model <- prune(model, cp = as.numeric(args$alpha) * nrow(frame) / model$frame$dev[1])
  } else if (mode == "cv") {
    # Breiman's 1-SE rule: the smallest tree whose CV error is within one
    # standard error of the lowest. Row j is optimal for cp in [CP_j, CP_{j-1}).
    table <- model$cptable
    best <- which.min(table[, "xerror"])
    j <- min(which(table[, "xerror"] <= table[best, "xerror"] + table[best, "xstd"]))
    cp <- if (j == 1) table[1, "CP"] else sqrt(table[j, "CP"] * table[j - 1, "CP"])
    model <- prune(model, cp = cp)
  }
  model
}

train <- read_part("train", as.integer(args$n_train))
if (args$warmup == "subset") invisible(fit(train[seq_len(min(2000, nrow(train))), ]))
start <- Sys.time()
model <- fit(train)
seconds <- as.numeric(difftime(Sys.time(), start, units = "secs"))

leaf <- model$frame$var == "<leaf>"
result <- sprintf('"train_seconds": %.9f, "nodes": %d, "leaves": %d, "depth": %d',
                  seconds, nrow(model$frame), sum(leaf),
                  max(floor(log2(as.numeric(rownames(model$frame))))))
if (args$eval == "1") {
  test <- read_part("test", as.integer(args$n_test))
  predicted <- as.character(predict(model, test, type = "class"))
  result <- paste0(result, sprintf(', "test_accuracy": %.9f',
                                   mean(predicted == as.character(test$class))))
}
cat("{", result, "}\n", sep = "")
