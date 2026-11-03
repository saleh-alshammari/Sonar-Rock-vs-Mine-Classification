"""Sonar (UCI Connectionist Bench): ten classifiers with and without PCA.

Same protocol as the paper: one stratified 85/15 split (random_state=42), StandardScaler and PCA
fitted on the 176 training returns only, the classifier settings of Table 4, and evaluation on the
same 32 test returns. The only difference between the two runs is the PCA(8) step.
Each model is also compared with its own PCA version by an exact McNemar test on the paired predictions.

Works with both file formats:
  * sonar_data.csv  (header feature_0..feature_59, target)
  * the original header-less file, e.g. Kaggle "/kaggle/input/sonar-mine-dataset/sonar data.csv"
Requirements: scikit-learn==1.9.1, xgboost==3.2.0, numpy, pandas, scipy
"""
import os, sys, platform, warnings
import numpy as np, pandas as pd, scipy, sklearn, xgboost
from scipy.stats import binomtest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------- data
DATA_PATH = None   # put your file path here, or leave None to search the usual places
CANDIDATES = ["sonar_data.csv", "sonar data.csv", "sonar.all-data",
              "/kaggle/input/sonar-mine-dataset/sonar data.csv"]
path = DATA_PATH or next((p for p in CANDIDATES if os.path.exists(p)), None)
if path is None:
    sys.exit("Data file not found: set DATA_PATH at the top of the script.")

df = pd.read_csv(path)
if "target" in df.columns:                      # sonar_data.csv
    X, labels = df.drop(columns="target"), df["target"]
else:                                           # original header-less file
    df = pd.read_csv(path, header=None)
    X, labels = df.iloc[:, :-1], df.iloc[:, -1]
assert X.shape == (208, 60), X.shape
assert set(labels) == {"M", "R"}, set(labels)

le = LabelEncoder()
y = le.fit_transform(labels)                    # M -> 0 (mine), R -> 1 (rock)

print("file:", path)
print("python", sys.version.split()[0], "|", platform.system(), platform.machine(),
      "| numpy", np.__version__, "| scipy", scipy.__version__,
      "| scikit-learn", sklearn.__version__, "| xgboost", xgboost.__version__)
print("label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
print("train", X_tr.shape[0], "test", X_te.shape[0],
      "| test mines", int((y_te == 0).sum()), "rocks", int((y_te == 1).sum()))

# ---------------------------------------------------------------- models (Table 4)
def make_models():
    return {
        "SVM": SVC(probability=True, kernel="rbf", C=2, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42),
        "MLP": MLPClassifier(random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=120, max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=120, learning_rate=0.12, max_depth=2,
                                 subsample=0.8, colsample_bytree=0.8, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.1,
                                                        max_depth=3, random_state=42),
        "Logistic Regression": LogisticRegression(),
        "AdaBoost": AdaBoostClassifier(n_estimators=120, learning_rate=0.1, random_state=42),
        "Naive Bayes": GaussianNB(),
    }

def run(model, use_pca):
    steps = [StandardScaler()] + ([PCA(n_components=8)] if use_pca else []) + [model]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        pred = make_pipeline(*steps).fit(X_tr, y_tr).predict(X_te)   # fitted on training data only
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="macro")
    cm = confusion_matrix(y_te, pred, labels=[0, 1])
    return dict(pred=pred, acc=100 * accuracy_score(y_te, pred), prec=100 * p, rec=100 * r,
                f1=100 * f, missed=int(cm[0, 1]), false_alarms=int(cm[1, 0]), converged=converged)

# ---------------------------------------------------------------- experiment
rows = []
for name in make_models():
    a = run(make_models()[name], use_pca=True)
    b = run(make_models()[name], use_pca=False)
    ok_a, ok_b = a["pred"] == y_te, b["pred"] == y_te
    only_pca, only_nopca = int((ok_a & ~ok_b).sum()), int((~ok_a & ok_b).sum())
    n = only_pca + only_nopca
    p_mcnemar = binomtest(min(only_pca, only_nopca), n, 0.5).pvalue if n else 1.0
    rows.append({"Model": name,
                 "Acc noPCA": b["acc"], "Prec": b["prec"], "Rec": b["rec"], "F1": b["f1"],
                 "Missed": b["missed"], "FalseAl": b["false_alarms"],
                 "Acc PCA8": a["acc"], "Missed PCA8": a["missed"], "FalseAl PCA8": a["false_alarms"],
                 "PCA-only right": only_pca, "noPCA-only right": only_nopca,
                 "McNemar p": p_mcnemar,
                 "MLP/other converged (noPCA)": b["converged"]})

res = pd.DataFrame(rows).sort_values("Acc noPCA", ascending=False, kind="stable")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res.round(2).to_string(index=False))
print("\nMean accuracy over the ten models: noPCA %.2f%% | PCA(8) %.2f%%"
      % (res["Acc noPCA"].mean(), res["Acc PCA8"].mean()))
