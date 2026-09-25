py_fiorino <- function() {
  if (!reticulate::py_available(initialize = TRUE)) {
    stop("Python is not available via reticulate. ", call. = FALSE)
  }
  if (!reticulate::py_module_available("fiorino")) {
    stop(
      "The Python package 'fiorino' is not installed in the active Python ",
      "environment. Install it with install_fiorino().",
      call. = FALSE
    )
  }
  reticulate::import("fiorino", delay_load = FALSE)
}

#' Install the Python backend for fiorino
#'
#' Installs the Python `fiorino` package into the reticulate environment so
#' the R wrappers have a backend to call.
#'
#' @param ... Passed to [reticulate::py_install()].
#' @return Invisibly `TRUE` on success.
#' @export
install_fiorino <- function(...) {
  reticulate::py_install("fiorino", ...)
  invisible(TRUE)
}
