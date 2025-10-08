"""
SDOH-Driven 30-Day Readmission Risk Prediction Model
Identifies patients who need patient navigation most using clinical + social features.
Author: Sai Manasa Adduru, MPH, PharmD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, average_precision_score,
                              precision_recall_curve)
import statsmodels.formula.api as smf
from scipy import stats
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("output/figures", exist_ok=True)

BLUE  = "#1B4F8A"
TEAL  = "#2AAFA4"
GREEN = "#27AE60"
RED   = "#C0392B"
GOLD  = "#F39C12"
GRAY  = "#7F8C8D"
LIGHT = "#F4F6F7"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 12, "axes.titleweight": "bold",
})

df = pd.read_csv("data/sdoh_patient_cohort.csv")
print(f"\n{'='*60}")
print("  SDOH READMISSION RISK PREDICTION MODEL")
print(f"  N = {len(df):,} patients")
print(f"  30-day readmission rate: {df.readmit_30d.mean():.1%}")
print(f"{'='*60}\n")

# ── 1. Feature Sets ───────────────────────────────────────────
clinical_features = ["age","female","cci","index_los","prior_admits_12m",
                     "chf","copd","diabetes","ckd","discharge_to_snf"]
sdoh_features     = ["housing_instability","food_insecurity","transportation_barrier",
                     "social_isolation","low_health_literacy","uninsured_gap",
                     "area_deprivation_index","dual_eligible","sdoh_burden_score"]
all_features = clinical_features + sdoh_features

X_all      = df[all_features]
X_clinical = df[clinical_features]
y          = df["readmit_30d"]

X_train, X_test, y_train, y_test = train_test_split(X_all, y, test_size=0.25, random_state=42, stratify=y)
Xc_train, Xc_test = train_test_split(X_clinical, test_size=0.25, random_state=42, stratify=y)

# ── 2. Model Training ─────────────────────────────────────────
print("MODEL TRAINING")

# Clinical-only baseline
lr_clinical = LogisticRegression(max_iter=1000, C=1.0)
lr_clinical.fit(Xc_train, y_train)
auc_clinical = roc_auc_score(y_test, lr_clinical.predict_proba(Xc_test)[:, 1])

# Clinical + SDOH logistic regression
lr_full = LogisticRegression(max_iter=1000, C=1.0)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
lr_full.fit(X_train_s, y_train)
auc_lr_full = roc_auc_score(y_test, lr_full.predict_proba(X_test_s)[:, 1])

# Gradient Boosting (main model)
gb = GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                 subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
y_prob = gb.predict_proba(X_test)[:, 1]
auc_gb = roc_auc_score(y_test, y_prob)
ap_gb  = average_precision_score(y_test, y_prob)

# CV AUC
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_aucs = cross_val_score(gb, X_all, y, cv=cv, scoring="roc_auc")

print(f"  Clinical-only model AUC:          {auc_clinical:.3f}")
print(f"  Clinical + SDOH (LR) AUC:         {auc_lr_full:.3f}")
print(f"  Clinical + SDOH (GBM) AUC:        {auc_gb:.3f}")
print(f"  5-fold CV AUC:                    {cv_aucs.mean():.3f} ± {cv_aucs.std():.3f}")
print(f"  Average Precision Score:          {ap_gb:.3f}")
print(f"  AUC improvement from SDOH:        +{auc_gb - auc_clinical:.3f}")

# ── 3. Feature Importance ─────────────────────────────────────
feat_imp = pd.Series(gb.feature_importances_, index=all_features).sort_values(ascending=False)
print("\nTOP 10 FEATURES BY IMPORTANCE")
for feat, imp in feat_imp.head(10).items():
    tag = "[SDOH]" if feat in sdoh_features else "[Clin]"
    print(f"  {tag} {feat:35s}  {imp:.4f}")

# ── 4. Navigation Need Score ──────────────────────────────────
df["readmit_risk_score"] = gb.predict_proba(X_all)[:, 1]
df["nav_priority"] = pd.cut(df["readmit_risk_score"],
                             bins=[0, 0.15, 0.30, 0.50, 1.0],
                             labels=["Low","Moderate","High","Critical"])

print("\nNAVIGATION PRIORITY DISTRIBUTION")
for tier in ["Critical","High","Moderate","Low"]:
    sub = df[df.nav_priority == tier]
    print(f"  {tier:10s}: {len(sub):,} patients ({len(sub)/len(df):.0%})  "
          f"actual readmit: {sub.readmit_30d.mean():.1%}")

# ── 5. SDOH Burden vs Readmission ─────────────────────────────
print("\nSDOH BURDEN vs 30-DAY READMISSION")
for score in range(0, 7):
    sub = df[df.sdoh_burden_score == score]
    if len(sub) > 20:
        print(f"  SDOH burden {score}: {len(sub):4d} patients  readmit rate {sub.readmit_30d.mean():.1%}")

# ── 6. FIGURES ────────────────────────────────────────────────
print("\nGENERATING FIGURES…")

# Figure 1 — ROC Curves (all 3 models)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for label, model_, X_, color in [
    ("Clinical Only (LR)", lr_clinical, Xc_test, GRAY),
    ("Clinical + SDOH (LR)", lr_full, X_test_s, GOLD),
    ("Clinical + SDOH (GBM)", gb, X_test, BLUE),
]:
    probs = model_.predict_proba(X_)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{label} (AUC={auc:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.4, label="Random (AUC=0.500)")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("Figure 1A — ROC Curves\nClinical vs. Clinical + SDOH Models")
ax.legend(fontsize=8); ax.set_facecolor(LIGHT)

ax = axes[1]
prec, rec, _ = precision_recall_curve(y_test, y_prob)
ax.plot(rec, prec, color=BLUE, lw=2.5)
ax.axhline(y_test.mean(), color=GRAY, ls="--", lw=1.5, label=f"Baseline prevalence ({y_test.mean():.1%})")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
ax.set_title(f"Figure 1B — Precision-Recall Curve\n(AP Score = {ap_gb:.3f})")
ax.legend(fontsize=9); ax.set_facecolor(LIGHT)
plt.suptitle("Model Performance: 30-Day Readmission Risk", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("output/figures/fig1_roc_curves.png", dpi=180, bbox_inches="tight")
plt.close(); print("  ✓ Figure 1 saved")

# Figure 2 — Feature Importance (SDOH vs Clinical split)
fig, ax = plt.subplots(figsize=(10, 6.5))
top15 = feat_imp.head(15)
colors_fi = [RED if f in sdoh_features else BLUE for f in top15.index]
labels_fi = [f"[SDOH] {f}" if f in sdoh_features else f"[Clin] {f}" for f in top15.index]
bars = ax.barh(labels_fi[::-1], top15.values[::-1], color=colors_fi[::-1], edgecolor="white", height=0.7)
for bar, val in zip(bars, top15.values[::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
            f"{val:.4f}", va="center", fontsize=9)
ax.set_xlabel("Feature Importance (Gain)")
ax.set_title("Figure 2 — Feature Importance: Clinical vs. SDOH Features\n(Red = SDOH factor, Blue = Clinical factor)")
ax.set_facecolor(LIGHT)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=RED, label="SDOH Feature"), Patch(color=BLUE, label="Clinical Feature")],
          loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("output/figures/fig2_feature_importance.png", dpi=180, bbox_inches="tight")
plt.close(); print("  ✓ Figure 2 saved")

# Figure 3 — SDOH Burden vs Readmission + Navigation Priority
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
burden_groups = df.groupby("sdoh_burden_score")["readmit_30d"].agg(["mean","count"]).reset_index()
burden_groups = burden_groups[burden_groups["count"] > 30]
bar_colors = [GREEN if m < 0.18 else GOLD if m < 0.28 else RED
              for m in burden_groups["mean"]]
bars = ax.bar(burden_groups["sdoh_burden_score"], burden_groups["mean"] * 100,
              color=bar_colors, edgecolor="white", width=0.7)
for bar, val in zip(bars, burden_groups["mean"]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.0%}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xlabel("SDOH Burden Score (0 = none, 8 = maximum)")
ax.set_ylabel("30-Day Readmission Rate (%)")
ax.set_title("Figure 3A — Readmission Rate by SDOH Burden\n(Higher burden = greater readmission risk)")
ax.set_facecolor(LIGHT)

ax = axes[1]
priority_counts = df.groupby("nav_priority")["readmit_30d"].agg(["mean","count"]).reindex(["Critical","High","Moderate","Low"])
tier_colors = [RED, GOLD, GREEN, TEAL]
bars = ax.bar(priority_counts.index, priority_counts["mean"] * 100,
              color=tier_colors, edgecolor="white", width=0.6)
for bar, (_, row) in zip(bars, priority_counts.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{row['mean']:.0%}\nn={row['count']:.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xlabel("Navigation Priority Tier")
ax.set_ylabel("Actual 30-Day Readmission Rate (%)")
ax.set_title("Figure 3B — Navigation Priority Tiers\n(Model-assigned risk stratification)")
ax.set_facecolor(LIGHT)
plt.suptitle("SDOH Burden and Navigation Need: Risk Stratification Analysis", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("output/figures/fig3_sdoh_risk.png", dpi=180, bbox_inches="tight")
plt.close(); print("  ✓ Figure 3 saved")

# Figure 4 — Risk Score Distribution by Navigation Status
fig, ax = plt.subplots(figsize=(9, 5))
nav_scores  = df[df.navigated == 1]["readmit_risk_score"]
ctrl_scores = df[df.navigated == 0]["readmit_risk_score"]
ax.hist(ctrl_scores, bins=40, alpha=0.55, color=GRAY,  label=f"Not Navigated (n={len(ctrl_scores):,})", density=True)
ax.hist(nav_scores,  bins=40, alpha=0.55, color=BLUE,  label=f"Navigated (n={len(nav_scores):,})",     density=True)
ax.axvline(nav_scores.mean(),  color=BLUE, ls="--", lw=2, label=f"Mean navigated: {nav_scores.mean():.2f}")
ax.axvline(ctrl_scores.mean(), color=GRAY, ls="--", lw=2, label=f"Mean control: {ctrl_scores.mean():.2f}")
ax.set_xlabel("Predicted 30-Day Readmission Risk Score")
ax.set_ylabel("Density")
ax.set_title("Figure 4 — Risk Score Distribution: Navigated vs. Non-Navigated\n(Navigated patients are higher risk — confirming appropriate targeting)")
ax.legend(fontsize=9); ax.set_facecolor(LIGHT)
plt.tight_layout()
plt.savefig("output/figures/fig4_risk_distribution.png", dpi=180, bbox_inches="tight")
plt.close(); print("  ✓ Figure 4 saved")

print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"  GBM model AUC:                {auc_gb:.3f}")
print(f"  SDOH adds over clinical only: +{auc_gb - auc_clinical:.3f} AUC")
print(f"  Critical priority patients:   {(df.nav_priority=='Critical').sum():,}  ({(df.nav_priority=='Critical').mean():.0%})")
print(f"  Their actual readmit rate:    {df[df.nav_priority=='Critical'].readmit_30d.mean():.1%}")
print(f"  Figures saved to output/figures/")
print(f"{'='*60}")
