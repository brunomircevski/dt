# rpart (CART, Gini) on the prepared chess split: one untimed warm-up fit on
# 50,000 rows, then REPS timed fits; writes the last tree and its test
# predictions for fit.py.
#
#   Rscript rpart_export.R DATA REPS DEPTH OUT_PREFIX
#
# Output: OUT_PREFIX.times (one line per fit, seconds), OUT_PREFIX.frame (one
# row per node, preorder: rpart's node number, variable, rows, class counts,
# and for splits the threshold and direction), OUT_PREFIX.pred (test classes,
# 0 = A, 1 = B). Settings as in bench/adapters/rpart_fit.R, unpruned (cp 0).
suppressMessages(library(rpart))
argv <- commandArgs(trailingOnly = TRUE)
data <- argv[1]; reps <- as.integer(argv[2]); depth <- as.integer(argv[3]); out <- argv[4]
meta_rows <- function(part) {
  text <- readLines(file.path(data, "meta.json"), warn = FALSE)
  as.integer(sub(".*: *([0-9]+).*", "\\1", grep(paste0('"n_', part, '"'), text, value = TRUE)))
}
read_part <- function(part) {
  rows <- meta_rows(part)
  con <- file(file.path(data, paste0(part, ".f32")), "rb")
  values <- readBin(con, "numeric", n = rows * 2, size = 4, endian = "little")
  close(con)
  con <- file(file.path(data, paste0(part, ".y.i32")), "rb")
  labels <- readBin(con, "integer", n = rows, size = 4, endian = "little")
  close(con)
  frame <- as.data.frame(matrix(values, nrow = rows, byrow = TRUE))
  names(frame) <- c("x", "y")
  frame$class <- factor(c("A", "B")[labels + 1], levels = c("A", "B"))
  frame
}
control <- rpart.control(minsplit = 2, minbucket = 1, cp = 0, maxcompete = 0,
                         maxsurrogate = 0, usesurrogate = 0, xval = 0, maxdepth = depth)
fit <- function(frame) rpart(class ~ ., data = frame, method = "class",
                             parms = list(split = "gini"), control = control)
train <- read_part("train")
invisible(fit(train[1:50000, ]))
times <- numeric(0)
for (r in seq_len(reps)) {
  start <- Sys.time()
  model <- fit(train)
  times <- c(times, as.numeric(difftime(Sys.time(), start, units = "secs")))
}
writeLines(sprintf("%.9f", times), paste0(out, ".times"))
f <- model$frame
split_row <- cumsum(f$var != "<leaf>")
counts <- f$yval2[, 2:3, drop = FALSE]
lines <- character(nrow(f))
for (i in seq_len(nrow(f))) {
  if (f$var[i] == "<leaf>") {
    thr <- "NA"; dir <- "0"
  } else {
    s <- model$splits[split_row[i], ]
    thr <- sprintf("%.17g", s[["index"]]); dir <- sprintf("%d", as.integer(s[["ncat"]]))
  }
  lines[i] <- paste(rownames(f)[i], as.character(f$var[i]), f$n[i],
                    counts[i, 1], counts[i, 2], c("A", "B")[f$yval[i]], thr, dir, sep = ",")
}
writeLines(c("node,var,n,A,B,class,threshold,ncat", lines), paste0(out, ".frame"))
test <- read_part("test")
pred <- as.integer(predict(model, test, type = "class")) - 1L
writeLines(as.character(pred), paste0(out, ".pred"))
writeLines(paste("rpart", as.character(packageVersion("rpart")), R.version.string),
           paste0(out, ".version"))
