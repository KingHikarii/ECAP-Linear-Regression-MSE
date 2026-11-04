# Predicting Exam Scores with Linear Regression (from scratch)

A simple linear regression model that predicts a student's exam score from the number of hours they studied. Gradient descent is **implemented by hand** in Python, then checked against the exact least-squares formula and scikit-learn.

## Results at a glance

| Metric | Value | Meaning |
|---|---|---|
| Slope (m) | 9.78 | Each extra hour of study adds about 9.8 points |
| Intercept (b) | 2.46 | Predicted score at 0 hours |
| MSE | 28.88 | Average squared error |
| RMSE | 5.37 | Predictions are typically off by about 5 points |
| MAE | 4.97 | Average absolute error, in points |
| CV of RMSE | 10.4% | RMSE as a share of the average score (51.48) |
| R² | 0.953 | Hours studied explain about 95% of the variation in scores |
| Correlation | 0.976 | Very strong, positive, straight-line relationship |
| Test RMSE (80/20 split) | 4.35 | Error on 5 students the model never saw |

**Final model:** `score = 9.78 × hours + 2.46`
**Example:** a student who studies 9.25 hours is predicted to score **92.9%**.

## Project files

| File | Description |
|---|---|
| `Linear_Regression_ECAP_Improved.ipynb` | The main notebook: data exploration, gradient descent, evaluation and visualizations |
| `score.csv` | The dataset: 25 students, columns `Hours` and `Scores` |
| `Predicting_Exam_Scores_ML_Guide.pdf` | A plain-English guide to the project, with an interview Q&A and a cheat sheet |
| `README.md` | This file |

## Dataset

`score.csv` has 25 rows and 2 columns, with no missing values.

| Column | Description | Min | Max | Mean |
|---|---|---|---|---|
| `Hours` | Hours the student studied | 1.1 | 9.2 | 5.01 |
| `Scores` | Exam score (%), the target to predict | 17 | 95 | 51.48 |

## How it works

The model is a straight line, `ŷ = m·x + b`. Training means finding the `m` and `b` that minimize the mean squared error (MSE).

**Gradient descent** starts at `m = 0, b = 0` and repeats these steps:

1. Predict every score with the current line.
2. Measure the error: `y − ŷ`.
3. Compute the gradients:
   - `∂MSE/∂m = −(2/n) · Σ x(y − ŷ)`
   - `∂MSE/∂b = −(2/n) · Σ (y − ŷ)`
4. Take a small step downhill: `m ← m − L·∂MSE/∂m`, `b ← b − L·∂MSE/∂b`.

**Settings:** learning rate `L = 0.01` and 1,000 iterations. The MSE falls from 3,264 to 28.9, and most of that drop happens in the first 5 iterations. For this data, learning rates above about 0.031 make the error grow instead of shrink.

## What's in the notebook

1. **Data loading and exploration:** summary statistics, missing-value check, scatter plot, histograms, correlation.
2. **Gradient descent:** the original loop-based version, plus a vectorized NumPy version that records the training history.
3. **Visualizations:**
   - the trend line, with the gap (residual) between each real score and the prediction
   - the loss curve over the iterations
   - the line at different iterations
   - the gradient-descent path across the error surface
   - a comparison of learning rates
4. **Evaluation:** MSE, RMSE, MAE, CV of RMSE, R², plus three residual plots (predicted vs actual, residuals vs hours, and the spread of the residuals).
5. **Checking the answer:** a comparison with the exact formula (ordinary least squares, or OLS) and scikit-learn's `LinearRegression`.

   | Method | m | b | MSE |
   |---|---|---|---|
   | Our gradient descent | 9.7789 | 2.4645 | 28.8828 |
   | Exact formula (OLS) | 9.7758 | 2.4837 | 28.8827 |
   | scikit-learn | 9.7758 | 2.4837 | 28.8827 |

6. **Train/test split (80/20):** train RMSE 5.61, test RMSE 4.35. The model works about as well on new students.
7. **Predictions:** a `predict_score(hours)` helper function.

## How to run it

### Option 1: Google Colab
1. Open `Linear_Regression_ECAP_Improved.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload `score.csv` using the Files panel on the left.
3. Choose **Runtime → Run all**.

### Option 2: on your own computer
```bash
pip install pandas numpy matplotlib scikit-learn jupyter
jupyter notebook Linear_Regression_ECAP_Improved.ipynb
```
Keep `score.csv` in the same folder as the notebook. It finds the file automatically, whether it's in Colab's `/content/` folder or the current folder.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- scikit-learn (used only to check the answer and for the train/test split)

## Limitations

- **Small dataset:** with only 25 students, results can shift noticeably with a few different students.
- **One feature:** other factors (sleep, attendance, prior grades) are not included.
- **No extrapolation:** the data only covers 1.1 to 9.2 hours. At 12 hours the line predicts about 120%, which is impossible.
- **Assumes a straight line:** real study time probably has diminishing returns.
- **Correlation, not causation:** the model predicts scores but doesn't prove that studying causes higher ones.

## Possible next steps

- Use k-fold cross-validation for a more reliable estimate of the error.
- Add more features (multiple linear regression).
- Try a polynomial term to test for diminishing returns.
- Add early stopping, so training ends when the loss stops improving.
- Compare batch, stochastic and mini-batch gradient descent.
