#' Fiorino tabular foundation models (reticulate bridge)
#'
#' @description
#' `fiorino_classifier()` and `fiorino_regressor()` build in-context tabular
#' learners backed by the Python `fiorino` package (FiorinoNano). `fio_fit()`
#' stores the training set as the in-context prompt (no gradients);
#' `fio_predict()` / `fio_predict_proba()` answer queries in one forward pass.
#'
#' Weights download once from Hugging Face
#' (`Nanite-Labs/nanites-fiorino-tabular`) on first fit.
#'
#' @param checkpoint Local weights file (`.pt` / `.safetensors`), an HF repo
#'   id, or `NULL` (default) to resolve the bifronte head automatically.
#' @param repo_id Hugging Face repo used when `checkpoint` is `NULL`.
#' @param device `"auto"` (default), `"cpu"`, or `"cuda"`.
#' @param seed Random seed for context subsampling.
#' @param max_rows Max context rows (subsampled, default 2048).
#' @return An object of class `fiorino_classifier` / `fiorino_regressor`
#'   wrapping the Python estimator.
#' @examples
#' \dontrun{
#' clf <- fiorino_classifier()
#' clf <- fio_fit(clf, iris[, 1:4], iris$Species)
#' fio_predict(clf, iris[1:5, 1:4])
#' }
#' @export
fiorino_classifier <- function(checkpoint = NULL,
                               repo_id = "Nanite-Labs/nanites-fiorino-tabular",
                               device = "auto",
                               seed = 0,
                               max_rows = 2048) {
  py <- py_fiorino()
  obj <- py$FiorinoClassifier(
    checkpoint = reticulate::r_to_py(checkpoint),
    repo_id = repo_id, device = device,
    seed = as.integer(seed), max_rows = as.integer(max_rows)
  )
  structure(list(py = obj, kind = "classifier"), class = "fiorino_classifier")
}

#' @rdname fiorino_classifier
#' @export
fiorino_regressor <- function(checkpoint = NULL,
                              repo_id = "Nanite-Labs/nanites-fiorino-tabular",
                              device = "auto",
                              seed = 0,
                              max_rows = 2048) {
  py <- py_fiorino()
  obj <- py$FiorinoRegressor(
    checkpoint = reticulate::r_to_py(checkpoint),
    repo_id = repo_id, device = device,
    seed = as.integer(seed), max_rows = as.integer(max_rows)
  )
  structure(list(py = obj, kind = "regressor"), class = "fiorino_regressor")
}

#' Fit a Fiorino model (store the in-context prompt)
#'
#' @param object A `fiorino_classifier` or `fiorino_regressor`.
#' @param X A data frame or matrix of features.
#' @param y Class labels (classifier) or numeric targets (regressor).
#' @param ... Unused.
#' @return The fitted model (invisibly the updated object).
#' @export
fio_fit <- function(object, X, y, ...) UseMethod("fio_fit")

#' @export
fio_fit.fiorino_classifier <- function(object, X, y, ...) {
  object$py$fit(as_fiorino_X(X), as_fiorino_y(y))
  object$classes <- reticulate::py_to_r(object$py$classes_)
  invisible(object)
}

#' @export
fio_fit.fiorino_regressor <- function(object, X, y, ...) {
  object$py$fit(as_fiorino_X(X), as_fiorino_y(y))
  invisible(object)
}

#' Predict with a fitted Fiorino model
#'
#' @param object A fitted `fiorino_classifier` or `fiorino_regressor`.
#' @param newdata A data frame or matrix of query rows.
#' @param ... Unused.
#' @return Class labels (classifier) or numeric predictions (regressor).
#' @export
fio_predict <- function(object, newdata, ...) UseMethod("fio_predict")

#' @export
fio_predict.fiorino_classifier <- function(object, newdata, ...) {
  reticulate::py_to_r(object$py$predict(as_fiorino_X(newdata)))
}

#' @export
fio_predict.fiorino_regressor <- function(object, newdata, ...) {
  as.numeric(reticulate::py_to_r(object$py$predict(as_fiorino_X(newdata))))
}

#' Class probabilities from a fitted Fiorino classifier
#'
#' @param object A fitted `fiorino_classifier`.
#' @param newdata A data frame or matrix of query rows.
#' @param ... Unused.
#' @return Numeric matrix with one column per class.
#' @export
fio_predict_proba <- function(object, newdata, ...) UseMethod("fio_predict_proba")

#' @export
fio_predict_proba.fiorino_classifier <- function(object, newdata, ...) {
  p <- reticulate::py_to_r(object$py$predict_proba(as_fiorino_X(newdata)))
  p <- as.matrix(p)
  if (!is.null(object$classes)) colnames(p) <- as.character(object$classes)
  p
}

#' @export
fio_predict_proba.fiorino_regressor <- function(object, newdata, ...) {
  stop("fio_predict_proba() is only defined for fiorino classifiers.", call. = FALSE)
}

as_fiorino_X <- function(X) {
  if (is.matrix(X)) X <- as.data.frame(X)
  if (!is.data.frame(X)) stop("X must be a data frame or matrix.", call. = FALSE)
  rownames(X) <- NULL
  X
}

as_fiorino_y <- function(y) {
  if (is.factor(y)) y <- as.character(y)
  y
}
