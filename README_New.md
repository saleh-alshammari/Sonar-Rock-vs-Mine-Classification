## Reproducing the paper's results

On Kaggle, open `SMART_PCA.ipynb` with the Sonar dataset attached and run all cells: it reproduces Tables 1, 3 and 4, the Wilson intervals, the McNemar tests and Figures 2 and 4 to 12. Locally, use the scripts below.

```
pip install -r requirements.txt
python sonar_reproduce.py                                   # Table 5
python sonar_nopca.py                                       # Table 6 (with and without PCA)
python make_figures_v3.py sonar_data.csv .                  # Figures 2 and 4 to 12
```

`sonar_reproduce.py` reads `sonar_data.csv` (208 returns, 60 features, column `target` with M or R), makes one stratified 85/15 split with `random_state=42`, fits `StandardScaler` and `PCA(n_components=8)` on the training data only, and trains the ten classifiers with the settings in Table 4. The output matches the paper: SVM 96.88% (31 of 32), Random Forest and the MLP 90.62%, and 69.6% of the variance kept by the eight components. The script suppresses warnings; without that, scikit-learn reports that the MLP did not converge within its default 200 iterations, as stated in the paper.

`sonar_nopca.py` trains the same ten classifiers with the same split, scaler and settings on the 60 standardized features without PCA, and compares each model with its PCA version by an exact McNemar test.

`make_figures_v3.py` writes the figure files used in the paper (`f_*_v3.png`).

The XGBoost result depends on the library version. With `xgboost==3.2.0` it is 84.38%, as in the paper. With `xgboost==3.4.1` the same code gives 90.62%, so keep the pinned versions in `requirements.txt`.
