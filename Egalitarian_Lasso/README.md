# 📊 Stepwise Penalized Regression Models for GDP Forecasting 📈

This Python script performs a multi-step regression analysis on GDP forecasting data using several penalized regression techniques (Lasso, Ridge, ElasticNet), combined with custom **egalitarian** weighting schemes and K-Fold cross-validation. It also implements the Diebold-Mariano test to compare predictive accuracy between models.

---

## 🚀 Features

- Data preprocessing and transformation of GDP forecasting data
- K-Fold cross-validation to tune regularization parameters (lambda) for Lasso, Ridge, and ElasticNet models
- Stepwise modeling:
  - Step 1: Lasso for feature selection
  - Step 2: Various penalized regression models on selected features with:
    - Egalitarian weighting adjustment (egalitarian Lasso, Ridge, ElasticNet)
    - RMSE-based weighting models
- Calculation of Root Mean Squared Error (RMSE) for model performance evaluation
- Implementation of Diebold-Mariano test to statistically compare forecasting models

---

## 🛠️ Requirements

- Python 3.7+
- numpy
- pandas
- scikit-learn
- matplotlib
- google.colab (for file upload interface in Google Colab)
- `dm_test` module (custom or provided separately)

---

## 📁 Usage

1. Upload your dataset (e.g., `H1_gdp.csv`) into the environment or specify the path in the code.

2. Run the script in a Python environment (Colab, Jupyter, or local).

3. The script will output:
   - Optimal lambda values for each model after cross-validation
   - Indices of selected features by Lasso
   - RMSE scores for each model variant
   - Results from Diebold-Mariano tests comparing forecast accuracy

---

## 🔧 Functions

- `cross_validate_lasso(X, y, lambdas, n_splits=5)`: Tunes lambda for Lasso using K-Fold CV  
- `cross_validate_ridge(X, y, lambdas, n_splits=5)`: Tunes lambda for Ridge using K-Fold CV  
- `cross_validate_en(X, y, lambdas, n_splits=5)`: Tunes lambda for ElasticNet using K-Fold CV  
- `egalitarian_transform(coefs)`: Adjusts coefficients to promote equal contribution among selected features  

---

## ⚙️ Notes

- The egalitarian transform simply adds an equal share to all coefficients to promote balanced feature contributions.
- The constant `c` controls the weighting for RMSE models and can be adjusted for experimentation.
- This script is designed to be modular and easily adapted to different datasets and model tuning parameters.

---

## 📚 References

- Friedman, Hastie, Tibshirani (2010). Regularization Paths for Generalized Linear Models via Coordinate Descent. Journal of Statistical Software.
- Diebold, Mariano (1995). Comparing Predictive Accuracy. Journal of Business & Economic Statistics.

---

## 🤝 Contributing

Feel free to fork and submit pull requests for improvements or bug fixes!

---

## 📫 Contact

For questions or collaboration, please open an issue or contact the author.

---

HAI!```
# Copy this README.md file to your Git repo for best results.
HAI!```

---

If you want me to generate a fully formatted **README.md** file ready to drop in your repo, just say!
