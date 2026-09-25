test_that("constructors fail gracefully without the Python backend", {
  skip_if(reticulate::py_module_available("fiorino"),
          "Python backend installed; nothing to assert here")
  expect_error(fiorino_classifier(), "not installed")
  expect_error(fiorino_regressor(), "not installed")
})

test_that("smoke: fit/predict on tiny synthetic data", {
  skip_if_not(reticulate::py_module_available("fiorino"),
              "Python package 'fiorino' not installed")
  skip_on_cran()

  set.seed(0)
  n <- 60
  X <- as.data.frame(matrix(rnorm(n * 6), ncol = 6))
  names(X) <- paste0("f", 0:5)
  y <- as.integer(X$f0 + 0.5 * X$f1 > 0)

  clf <- fiorino_classifier(device = "cpu")
  clf <- fio_fit(clf, X[1:40, ], y[1:40])
  p <- fio_predict_proba(clf, X[41:60, ])
  expect_equal(dim(p), c(20L, 2L))
  expect_true(all(is.finite(p)))
  pred <- fio_predict(clf, X[41:60, ])
  expect_length(pred, 20L)
})
