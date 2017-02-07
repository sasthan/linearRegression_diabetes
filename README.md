# linearRegression_diabetes
Linear Regression: scikit-learn + numpy + matplotlib.pyplot

Understanding Linear Regression with scikit-learn and using diabetes dataset dataset for the same.
Reference for Linear Gression example - http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

## Understanding dataset

Reference:  http://www4.stat.ncsu.edu/~boos/var.select/diabetes.html

The original dataset has been normalized with zero mean and Euclidean norm 1. N = 442 patients and p=10 predictors. One row per patient. 

## Requirements

Need python installed on computer. I used IDLE IDE to run python scripts

## Running script interactively with IDLE

```python
>>> import matplotlib.pyplot as plt
>>> import numpy as np
>>> from sklearn import datasets

# Load diabetes dataset
>>> diabetes = datasets.load_diabetes()
>>> diabetes.data
array([[ 0.03807591,  0.05068012,  0.06169621, ..., -0.00259226,
         0.01990842, -0.01764613],
       [-0.00188202, -0.04464164, -0.05147406, ..., -0.03949338,
        -0.06832974, -0.09220405],
       [ 0.08529891,  0.05068012,  0.04445121, ..., -0.00259226,
         0.00286377, -0.02593034],
       ..., 
       [ 0.04170844,  0.05068012, -0.01590626, ..., -0.01107952,
        -0.04687948,  0.01549073],
       [-0.04547248, -0.04464164,  0.03906215, ...,  0.02655962,
         0.04452837, -0.02593034],
       [-0.04547248, -0.04464164, -0.0730303 , ..., -0.03949338,
        -0.00421986,  0.00306441]])

# Split the data into training/testing sets
>>> X_train=X[:-20]
>>> X_test=X[-20:]
>>> y_train=y[:-20]
>>> y_test=y[-20:]      

# Create Linear Regression object
>>> regr=linear_model.LinearRegression()
>>> regr.fit(X_train, y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

#Print coefficients
>>> print('Coefficient: \n', regr.coef_)
# Print Mean Square error
>>> print("Mean squared error : %2f" % np.mean((regr.predict(X_test) - y_test) ** 2 ))
# Print Variance score
>>> print('Variance score : %2f' % regr.score(X_test, y_test))

```
