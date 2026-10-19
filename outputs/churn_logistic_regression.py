"""
Binary Classification using Logistic Regression - Telco Customer Churn
Full pipeline: EDA -> Preprocessing -> Modeling -> Evaluation -> Interpretation
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                              recall_score, f1_score, classification_report,
                              roc_curve, roc_auc_score, RocCurveDisplay)
import json, os

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120
CHARTS = "charts"
OUT = "outputs"
os.makedirs(CHARTS, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ============================================================
# PHASE 1: PROBLEM DEFINITION & EDA
# ============================================================
df = pd.read_csv("data/Dataset.csv")
df.drop(columns=["customerID"], inplace=True)

# TotalCharges has blank strings -> convert to numeric, coerce errors to NaN
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

numerical_features = ["tenure", "MonthlyCharges", "TotalCharges"]
categorical_features = [c for c in df.columns if c not in numerical_features + ["Churn"]]

eda_summary = {
    "shape": df.shape,
    "numerical_features": numerical_features,
    "categorical_features": categorical_features,
    "missing_values": df.isnull().sum().to_dict(),
    "churn_distribution": df["Churn"].value_counts().to_dict(),
    "churn_distribution_pct": (df["Churn"].value_counts(normalize=True) * 100).round(2).to_dict(),
}

# Chart 1: Target class distribution
plt.figure(figsize=(5, 4))
ax = sns.countplot(x="Churn", data=df, palette=["#4C72B0", "#DD8452"])
plt.title("Churn Class Distribution")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.savefig(f"{CHARTS}/01_churn_distribution.png")
plt.close()

# Chart 2: Histograms of numerical features
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, numerical_features):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#4C72B0")
    ax.set_title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(f"{CHARTS}/02_numerical_histograms.png")
plt.close()

# Chart 3: Boxplots numerical features vs Churn
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, numerical_features):
    sns.boxplot(x="Churn", y=col, data=df, ax=ax, palette=["#4C72B0", "#DD8452"])
    ax.set_title(f"{col} by Churn")
plt.tight_layout()
plt.savefig(f"{CHARTS}/03_boxplots_by_churn.png")
plt.close()

# Chart 4: Correlation heatmap (numeric + encoded target)
corr_df = df[numerical_features].copy()
corr_df["Churn"] = (df["Churn"] == "Yes").astype(int)
plt.figure(figsize=(5, 4))
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{CHARTS}/04_correlation_heatmap.png")
plt.close()

# Chart 5: Key categorical features vs churn rate
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.flatten(), ["Contract", "InternetService", "PaymentMethod", "SeniorCitizen"]):
    rate = df.groupby(col)["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    rate.sort_values().plot(kind="barh", ax=ax, color="#55A868")
    ax.set_title(f"Churn Rate (%) by {col}")
    ax.set_xlabel("Churn Rate (%)")
plt.tight_layout()
plt.savefig(f"{CHARTS}/05_churn_rate_by_category.png")
plt.close()

# ============================================================
# PHASE 2: DATA PREPROCESSING
# ============================================================
# Handle missing values: 11 rows with blank TotalCharges are new customers (tenure=0)
# -> impute with 0 (matches MonthlyCharges * tenure logic)
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Encode target
df["Churn"] = (df["Churn"] == "Yes").astype(int)

# One-hot encode categorical features (drop_first to avoid dummy trap)
df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numerical features only (fit on train, apply to both)
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

preprocessing_summary = {
    "rows_after_cleaning": df.shape[0],
    "features_after_encoding": X.shape[1],
    "train_size": X_train.shape[0],
    "test_size": X_test.shape[0],
    "train_churn_rate": round(y_train.mean() * 100, 2),
    "test_churn_rate": round(y_test.mean() * 100, 2),
}

# ============================================================
# PHASE 3: MODEL IMPLEMENTATION
# ============================================================
# class_weight='balanced' chosen to counter the ~73/27 class imbalance
# penalty='l2' (default, works with all solvers), C=1.0 (moderate regularization)
# solver='lbfgs' (efficient default for this dataset size), max_iter increased for convergence
model = LogisticRegression(
    penalty="l2", C=1.0, solver="lbfgs", class_weight="balanced",
    max_iter=1000, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# ============================================================
# PHASE 4: MODEL EVALUATION
# ============================================================
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["No Churn", "Churn"])
auc = roc_auc_score(y_test, y_proba)

# Chart 6: Confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{CHARTS}/06_confusion_matrix.png")
plt.close()

# Chart 7: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})", color="#4C72B0")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{CHARTS}/07_roc_curve.png")
plt.close()

evaluation_summary = {
    "confusion_matrix": cm.tolist(),
    "accuracy": round(acc, 4),
    "precision": round(prec, 4),
    "recall": round(rec, 4),
    "f1_score": round(f1, 4),
    "roc_auc": round(auc, 4),
    "classification_report": report,
}

# ============================================================
# PHASE 5: INTERPRETATION & INSIGHTS
# ============================================================
coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
top_positive = coefs.tail(10)[::-1]   # increase churn odds
top_negative = coefs.head(10)         # decrease churn odds

# Chart 8: Top coefficients
fig, ax = plt.subplots(figsize=(8, 7))
combined = pd.concat([top_negative, top_positive]).sort_values()
colors = ["#55A868" if v < 0 else "#C44E52" for v in combined]
combined.plot(kind="barh", color=colors, ax=ax)
ax.set_title("Top Feature Coefficients (Logistic Regression)\nRed = increases churn risk, Green = decreases churn risk")
ax.set_xlabel("Coefficient (log-odds impact)")
plt.tight_layout()
plt.savefig(f"{CHARTS}/08_top_coefficients.png")
plt.close()

interpretation_summary = {
    "top_positive_drivers": top_positive.round(3).to_dict(),
    "top_negative_drivers": top_negative.round(3).to_dict(),
}

# ============================================================
# SAVE ALL SUMMARIES
# ============================================================
def convert(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, tuple):
        return list(o)
    return str(o)

all_summary = {
    "eda": eda_summary,
    "preprocessing": preprocessing_summary,
    "evaluation": evaluation_summary,
    "interpretation": interpretation_summary,
}
with open(f"{OUT}/summary.json", "w") as f:
    json.dump(all_summary, f, default=convert, indent=2)

print("=== EDA ===")
print(eda_summary)
print("\n=== PREPROCESSING ===")
print(preprocessing_summary)
print("\n=== EVALUATION ===")
print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}  AUC: {auc:.4f}")
print(cm)
print(report)
print("\n=== TOP POSITIVE DRIVERS (increase churn) ===")
print(top_positive)
print("\n=== TOP NEGATIVE DRIVERS (decrease churn) ===")
print(top_negative)

# Save cleaned dataset used for modeling
df.to_csv(f"{OUT}/Dataset_clean.csv", index=False)

# Save the trained model
import pickle
with open(f"{OUT}/logistic_regression_model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler, "feature_columns": list(X.columns),
                 "numerical_features": numerical_features}, f)

print("\nDone. Charts saved to", CHARTS)
