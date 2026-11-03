"""Reproduces the results table of the paper.
Ten classifiers on StandardScaler + PCA(8) features of the UCI Sonar data.
One stratified 85/15 split (random_state=42). Scaler and PCA are fitted on the training data only.
Put this file next to sonar_data.csv and run:  python sonar_reproduce.py
"""
import warnings
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
print("scikit-learn", sklearn.__version__, "| numpy", np.__version__, "| xgboost", xgboost.__version__)

df = pd.read_csv("sonar_data.csv")
X = df.drop(columns="target")
le = LabelEncoder()
y = le.fit_transform(df["target"])          # M -> 0 (mine), R -> 1 (rock)
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
print("train", X_tr.shape, "test", X_te.shape, "| test class counts (mine, rock):", np.bincount(y_te))

models = {
    "SVM": SVC(probability=True, kernel="rbf", C=2, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42),
    "Neural Network (MLP)": MLPClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=120, max_depth=5, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=120, learning_rate=0.12, max_depth=2, subsample=0.8,
                             colsample_bytree=0.8, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=3, random_state=42),
    "Logistic Regression": LogisticRegression(),
    "AdaBoost": AdaBoostClassifier(n_estimators=120, learning_rate=0.1, random_state=42),
    "Naive Bayes": GaussianNB(),
}

rows = []
for name, model in models.items():
    pipe = make_pipeline(StandardScaler(), PCA(n_components=8), model)   # fitted on training data only
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="macro")
    tn_mine, fn_mine_as_rock, fa_rock_as_mine, tp_rock = confusion_matrix(y_te, pred).ravel()
    rows.append({"Model": name,
                 "Accuracy (%)": round(100 * accuracy_score(y_te, pred), 2),
                 "Precision (%)": round(100 * p, 2),
                 "Recall (%)": round(100 * r, 2),
                 "F1 (%)": round(100 * f, 2),
                 "Missed mines": int(fn_mine_as_rock),
                 "False alarms": int(fa_rock_as_mine)})
res = pd.DataFrame(rows).sort_values("Accuracy (%)", ascending=False, kind="stable").reset_index(drop=True)
print(res.to_string(index=False))

pca = make_pipeline(StandardScaler(), PCA(n_components=8)).fit(X_tr)
print("Variance kept by 8 components: %.1f%%" % (100 * pca[-1].explained_variance_ratio_.sum()))
