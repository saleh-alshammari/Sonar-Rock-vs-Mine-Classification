import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
fig, ax = plt.subplots(figsize=(7.6, 6.4)); ax.set_xlim(0, 10); ax.set_ylim(0.6, 10.5); ax.axis("off")
BLUE, ORANGE, GRAY = "#dbe9f6", "#fde3c8", "#eeeeee"
def box(x, y, w, h, text, fc, bold=False):
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec="#333333", lw=1))
    ax.text(x, y, text, ha="center", va="center", fontsize=9, fontweight="bold" if bold else "normal")
def arrow(x1, y1, x2, y2, ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1, linestyle=ls))
box(5, 10.0, 6.4, 0.7, "UCI Sonar dataset: 208 returns, 60 band energies (111 M, 97 R)", GRAY, True)
box(5, 8.95, 6.4, 0.7, "Label encoding (M = 0, R = 1)", GRAY)
box(5, 7.9, 6.4, 0.7, "Stratified 85/15 split (random_state = 42)", GRAY, True)
arrow(5, 9.65, 5, 9.3); arrow(5, 8.6, 5, 8.25)
box(2.6, 6.75, 4.2, 0.7, "Training set: 176 returns", BLUE, True)
box(7.4, 6.75, 4.2, 0.7, "Test set: 32 returns", ORANGE, True)
arrow(4.2, 7.55, 2.9, 7.1); arrow(5.8, 7.55, 7.1, 7.1)
box(2.6, 5.7, 4.2, 0.7, "Fit StandardScaler", BLUE)
box(2.6, 4.65, 4.2, 0.7, "Fit PCA (8 components)\nor no PCA (60 features)", BLUE)
box(2.6, 3.5, 4.2, 0.8, "Fit ten classifiers\n(fixed settings, Table 2)", BLUE)
arrow(2.6, 6.4, 2.6, 6.05); arrow(2.6, 5.35, 2.6, 5.0); arrow(2.6, 4.3, 2.6, 3.9)
box(7.4, 5.7, 4.2, 0.7, "Apply fitted scaler", ORANGE)
box(7.4, 4.65, 4.2, 0.7, "Apply fitted PCA\n(PCA runs only)", ORANGE)
box(7.4, 3.5, 4.2, 0.8, "Predict with the\nfitted classifiers", ORANGE)
arrow(7.4, 6.4, 7.4, 6.05); arrow(7.4, 5.35, 7.4, 5.0); arrow(7.4, 4.3, 7.4, 3.9)
arrow(4.7, 5.7, 5.3, 5.7, ls="--"); arrow(4.7, 4.65, 5.3, 4.65, ls="--"); arrow(4.7, 3.5, 5.3, 3.5, ls="--")
box(5, 1.95, 8.8, 1.1, "Evaluation on the 32 test returns: accuracy, precision, recall, F1, confusion matrices\n(missed mines, false alarms), Wilson intervals, exact McNemar tests", GRAY)
arrow(7.4, 3.1, 6.2, 2.5)
ax.text(5, 0.95, "Dashed arrows: parameters learned on the training set are reused unchanged on the test set.", ha="center", fontsize=8, style="italic")
fig.tight_layout(); fig.savefig("/home/claude/ov/f_workflow_v3.png", dpi=250)
