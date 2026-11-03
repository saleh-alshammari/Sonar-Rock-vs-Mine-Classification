"""Regenerates five figures of the paper from sonar_data.csv (v3 file names).
f_signals_v3.png, f_feature3_v3.png, f_corr_train_v3.png, f_confusion_v3.png, f_accuracy_v3.png"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import xgboost; from xgboost import XGBClassifier
out = sys.argv[2] if len(sys.argv) > 2 else "."
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25})
BLUE, ORANGE = "#1f77b4", "#e07b00"
df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "sonar_data.csv"); X = df.drop(columns="target"); y = LabelEncoder().fit_transform(df["target"])
feat = X.values
def save(fig, name, dpi=250): fig.tight_layout(); fig.savefig(f"{out}/{name}", dpi=dpi); plt.close(fig)
# one rock and one mine return (first of each class in the file)
r_i, m_i = int(np.where(y == 1)[0][0]), int(np.where(y == 0)[0][0]); print("rock row", r_i, "mine row", m_i)
fig, ax = plt.subplots(figsize=(7.5, 3.2)); ax.plot(range(60), feat[r_i], color=BLUE, label=f"Rock return"); ax.plot(range(60), feat[m_i], color=ORANGE, label="Mine return")
ax.set_xticks(range(0, 60, 5)); ax.set_xlim(-1, 60); ax.set_xlabel("Feature index (frequency band)"); ax.set_ylabel("Energy"); ax.legend(frameon=False); save(fig, "f_signals_v3.png")
# feature 3 raw vs standardized (all 208 returns, illustration)
v = feat[:, 3]; z = (v - v.mean()) / v.std()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.0))
a1.hist(v, bins=20, color=BLUE, edgecolor="white"); a1.set_title("Feature 3, raw"); a1.set_xlabel("Raw value"); a1.set_ylabel("Count")
a2.hist(z, bins=20, color="#2a9d5c", edgecolor="white"); a2.set_title("Feature 3, standardized"); a2.set_xlabel("Standardized value")
save(fig, "f_feature3_v3.png")
# split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
C = np.corrcoef(Xtr.values.T); print("train min corr %.3f" % C.min())
fig, ax = plt.subplots(figsize=(6.0, 5.0)); im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1); ax.grid(False)
ax.set_xticks(range(0, 60, 10)); ax.set_yticks(range(0, 60, 10)); ax.set_xlabel("Feature index"); ax.set_ylabel("Feature index")
fig.colorbar(im, ax=ax, label="Pearson correlation"); save(fig, "f_corr_train_v3.png")
models = {
    "SVM": SVC(probability=True, kernel="rbf", C=2, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=120, max_depth=5, random_state=42),
    "MLP": MLPClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(n_estimators=120, max_depth=5, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=120, learning_rate=0.12, max_depth=2, subsample=0.8, colsample_bytree=0.8, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.1, max_depth=3, random_state=42),
    "Logistic Regression": LogisticRegression(),
    "AdaBoost": AdaBoostClassifier(n_estimators=120, learning_rate=0.1, random_state=42),
    "Naïve Bayes": GaussianNB(),
}
cms, acc = {}, {}
for n, m in models.items():
    p = make_pipeline(StandardScaler(), PCA(n_components=8), m).fit(Xtr, ytr).predict(Xte)
    cms[n] = confusion_matrix(yte, p, labels=[0, 1]); acc[n] = 100 * (p == yte).mean()
    print(f"{n:20s} {acc[n]:6.2f}  missed {cms[n][0,1]}  false alarms {cms[n][1,0]}")
fig, axes = plt.subplots(2, 5, figsize=(16, 7))
for ax, n in zip(axes.ravel(), models):
    cm = cms[n]; ax.imshow(cm, cmap="Blues", vmin=0, vmax=17); ax.grid(False)
    for i in range(2):
        for j in range(2): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > 8 else "black", fontsize=16)
    ax.set_xticks([0, 1], ["Mine", "Rock"], fontsize=12); ax.set_yticks([0, 1], ["Mine", "Rock"], fontsize=12)
    ax.set_title(n, fontsize=14); ax.set_xlabel("Predicted", fontsize=12); ax.set_ylabel("True", fontsize=12)
save(fig, "f_confusion_v3.png", 200)
names = list(models); vals = [acc[n] for n in names]
fig, ax = plt.subplots(figsize=(8, 3.8)); b = ax.bar(names, vals, color=[ORANGE] + [BLUE] * 9, width=.65)
for r_, a in zip(b, vals): ax.text(r_.get_x() + r_.get_width() / 2, a + 1, f"{a:.2f}", ha="center", fontsize=9)
ax.set_ylim(0, 105); ax.set_ylabel("Test accuracy (%)"); ax.grid(axis="x", visible=False)
plt.xticks(rotation=40, ha="right"); save(fig, "f_accuracy_v3.png")
print("xgboost", xgboost.__version__)
# ---- remaining data figures, without baked-in titles (v3)
fig, ax = plt.subplots(figsize=(8, 3.6)); xs = np.arange(60)
for lab, col, nm in [(1, BLUE, "Rock"), (0, ORANGE, "Mine")]:
    mu, sd = feat[y == lab].mean(0), feat[y == lab].std(0)
    ax.plot(xs, mu, color=col, lw=2, label=f"{nm} (mean, n={int((y==lab).sum())})"); ax.fill_between(xs, mu-sd, mu+sd, color=col, alpha=.18, lw=0)
ax.set_xlim(0, 59); ax.set_xlabel("Feature index (frequency band)"); ax.set_ylabel("Energy"); ax.legend(frameon=False, title="Class mean ± 1 SD"); save(fig, "f_classmeans_v3.png")
fig, axes = plt.subplots(6, 10, figsize=(15, 9.0), sharex=True, sharey=True)
for i, ax in enumerate(axes.ravel()):
    ax.hist(feat[:, i], bins=np.linspace(0, 1, 21), color=BLUE, edgecolor="white", lw=.3); ax.set_title(f"Feature {i}", fontsize=9, pad=2); ax.grid(False); ax.tick_params(labelsize=9)
for ax in axes[-1]: ax.set_xlabel("Value", fontsize=9)
for ax in axes[:, 0]: ax.set_ylabel("Count", fontsize=9)
save(fig, "f_hist_v3.png", 200)
S = StandardScaler().fit_transform(feat)
fig, ax = plt.subplots(figsize=(10, 3.6)); ax.boxplot(S, widths=.6, flierprops=dict(marker=".", ms=2, alpha=.5), medianprops=dict(color=ORANGE), patch_artist=True, boxprops=dict(facecolor="#cfe3f3", lw=.6))
ax.set_xticks(range(1, 61, 5), range(0, 60, 5)); ax.set_xlabel("Feature index"); ax.set_ylabel("Standardized value"); save(fig, "f_box_v3.png")
sc = StandardScaler().fit(Xtr); Str = sc.transform(Xtr)
cum = np.cumsum(PCA().fit(Str).explained_variance_ratio_)
fig, ax = plt.subplots(figsize=(6.4, 3.4)); ax.plot(range(1, 61), cum, marker="o", ms=3, color=BLUE)
ax.axvline(8, ls="--", color="gray", lw=1); ax.scatter([8], [cum[7]], color=ORANGE, zorder=5)
ax.annotate(f"8 components: {cum[7]*100:.1f}%", (8, cum[7]), xytext=(14, cum[7]-0.15), arrowprops=dict(arrowstyle="-", color="gray"))
ax.set_xlabel("Number of principal components"); ax.set_ylabel("Cumulative explained variance"); ax.set_ylim(0, 1.02); save(fig, "f_pcavar_v3.png")
Ztr = PCA(n_components=8).fit(Str).transform(Str); C = np.corrcoef(Ztr.T)
fig, ax = plt.subplots(figsize=(5.2, 4.2)); im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1); ax.grid(False)
for i in range(8):
    for j in range(8): ax.text(j, i, f"{C[i,j]:.2f}".replace("-0.00", "0.00"), ha="center", va="center", fontsize=7, color="white" if abs(C[i,j]) > .6 else "black")
ax.set_xticks(range(8), [f"PC{i+1}" for i in range(8)]); ax.set_yticks(range(8), [f"PC{i+1}" for i in range(8)])
fig.colorbar(im, ax=ax, label="Pearson correlation"); save(fig, "f_pccorr_v3.png")
print("8 comps %.4f, max offdiag %.2e" % (cum[7], np.abs(C - np.eye(8)).max()))
