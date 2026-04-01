"""
Student Performance Analysis & Prediction ML Pipeline
======================================================
Models: Random Forest + XGBoost
Explainability: SHAP values, Feature Importance, LIME
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import os

# ─────────────────────────────────────────────
# 1. DATA GENERATION (synthetic student dataset)
# ─────────────────────────────────────────────

def generate_student_data(n_samples=1000, random_state=42):
    """Generate realistic synthetic student performance dataset."""
    np.random.seed(random_state)

    data = {
        "study_hours_per_day":    np.clip(np.random.normal(4, 1.5, n_samples), 0, 12),
        "attendance_rate":        np.clip(np.random.normal(75, 15, n_samples), 30, 100),
        "previous_grades":        np.clip(np.random.normal(65, 15, n_samples), 20, 100),
        "sleep_hours":            np.clip(np.random.normal(7, 1.5, n_samples), 3, 12),
        "extracurricular_hours":  np.clip(np.random.normal(2, 1.2, n_samples), 0, 8),
        "parental_education":     np.random.choice(["None","High School","Bachelor","Master","PhD"], n_samples,
                                                    p=[0.05, 0.35, 0.35, 0.20, 0.05]),
        "internet_access":        np.random.choice(["Yes","No"], n_samples, p=[0.80, 0.20]),
        "tutoring":               np.random.choice(["Yes","No"], n_samples, p=[0.40, 0.60]),
        "family_income":          np.random.choice(["Low","Medium","High"], n_samples, p=[0.30, 0.45, 0.25]),
        "motivation_level":       np.random.choice(["Low","Medium","High"], n_samples, p=[0.25, 0.50, 0.25]),
        "gender":                 np.random.choice(["Male","Female"], n_samples, p=[0.50, 0.50]),
        "school_type":            np.random.choice(["Public","Private"], n_samples, p=[0.70, 0.30]),
        "distance_to_school_km":  np.clip(np.random.exponential(5, n_samples), 0.5, 50),
        "teacher_quality":        np.random.choice(["Low","Medium","High"], n_samples, p=[0.20, 0.55, 0.25]),
        "stress_level":           np.clip(np.random.normal(5, 2, n_samples), 1, 10),
    }

    df = pd.DataFrame(data)

    # Deterministic target based on features
    score = (
        df["study_hours_per_day"] * 4.0
        + df["attendance_rate"] * 0.25
        + df["previous_grades"] * 0.30
        + (df["sleep_hours"] - 7).abs() * -2.0
        + df["extracurricular_hours"] * 0.5
        + (df["parental_education"].map({"None":0,"High School":1,"Bachelor":2,"Master":3,"PhD":4}) * 2.0)
        + (df["internet_access"] == "Yes").astype(int) * 3.0
        + (df["tutoring"] == "Yes").astype(int) * 4.0
        + df["family_income"].map({"Low":0,"Medium":2,"High":4})
        + df["motivation_level"].map({"Low":-3,"Medium":0,"High":5})
        + df["teacher_quality"].map({"Low":-3,"Medium":0,"High":4})
        - df["stress_level"] * 0.8
        - df["distance_to_school_km"] * 0.1
        + np.random.normal(0, 5, n_samples)   # noise
    )

    # Classify: 0=At Risk, 1=Average, 2=High Performer
    percentiles = np.percentile(score, [33, 66])
    df["performance"] = pd.cut(score, bins=[-np.inf, percentiles[0], percentiles[1], np.inf],
                               labels=["At Risk", "Average", "High Performer"])

    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    df = df.copy()

    # Ordinal mappings
    ordinal_maps = {
        "parental_education": {"None":0,"High School":1,"Bachelor":2,"Master":3,"PhD":4},
        "family_income":      {"Low":0,"Medium":1,"High":2},
        "motivation_level":   {"Low":0,"Medium":1,"High":2},
        "teacher_quality":    {"Low":0,"Medium":1,"High":2},
    }
    for col, mapping in ordinal_maps.items():
        df[col] = df[col].map(mapping)

    # Binary
    df["internet_access"] = (df["internet_access"] == "Yes").astype(int)
    df["tutoring"]         = (df["tutoring"] == "Yes").astype(int)
    df["gender"]           = (df["gender"] == "Female").astype(int)
    df["school_type"]      = (df["school_type"] == "Private").astype(int)

    # Encode target
    le = LabelEncoder()
    df["performance_encoded"] = le.fit_transform(df["performance"])

    feature_cols = [c for c in df.columns if c not in ["performance", "performance_encoded"]]
    X = df[feature_cols]
    y = df["performance_encoded"]

    return X, y, le, feature_cols


# ─────────────────────────────────────────────
# 3. TRAIN MODELS
# ─────────────────────────────────────────────

def train_models(X_train, y_train):
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                                                random_state=42, n_jobs=-1),
        "XGBoost":       XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                       use_label_encoder=False, eval_metric="mlogloss",
                                       random_state=42, n_jobs=-1),
        "Logistic Reg":  LogisticRegression(max_iter=1000, random_state=42),
    }
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"  ✓ Trained: {name}")
    return trained


# ─────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────

def evaluate_models(models, X_test, y_test, label_encoder):
    results = {}
    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)
        acc     = accuracy_score(y_test, y_pred)
        auc     = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
        cv_scores = cross_val_score(model, X_test, y_test, cv=5, scoring="accuracy")

        results[name] = {
            "accuracy": acc,
            "auc":      auc,
            "cv_mean":  cv_scores.mean(),
            "cv_std":   cv_scores.std(),
            "report":   classification_report(y_test, y_pred,
                            target_names=label_encoder.classes_, output_dict=True),
            "confusion": confusion_matrix(y_test, y_pred),
        }
        print(f"  {name:15s} | Acc: {acc:.3f} | AUC: {auc:.3f} | CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
    return results


# ─────────────────────────────────────────────
# 5. EXPLAINABILITY – SHAP
# ─────────────────────────────────────────────

def compute_shap(model, X_train, X_test, model_name, out_dir):
    print(f"  Computing SHAP for {model_name}...")
    if "XGBoost" in model_name or "Random Forest" in model_name:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.LinearExplainer(model, X_train)

    shap_values = explainer.shap_values(X_test)

    # For multiclass RF, shap_values is a list; pick the "High Performer" class (index 2)
    sv = shap_values[2] if isinstance(shap_values, list) else shap_values

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Summary bar
    plt.sca(axes[0])
    shap.summary_plot(sv, X_test, plot_type="bar", show=False, max_display=10)
    axes[0].set_title(f"SHAP Feature Importance\n({model_name})", fontsize=13, fontweight="bold")

    # Beeswarm
    plt.sca(axes[1])
    shap.summary_plot(sv, X_test, show=False, max_display=10)
    axes[1].set_title(f"SHAP Beeswarm Plot\n({model_name})", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, f"shap_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")
    return shap_values


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE CHART
# ─────────────────────────────────────────────

def plot_feature_importance(model, feature_cols, model_name, out_dir):
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
    else:
        return

    idx  = np.argsort(fi)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(idx)))[::-1]
    bars = ax.barh([feature_cols[i] for i in idx[::-1]], fi[idx[::-1]], color=colors)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Feature Importance – {model_name}", fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, f"feature_importance_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")


# ─────────────────────────────────────────────
# 7. CONFUSION MATRIX PLOT
# ─────────────────────────────────────────────

def plot_confusion(cm, classes, model_name, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax, linewidths=0.5)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"confusion_{model_name.replace(' ','_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────
# 8. EDA PLOTS
# ─────────────────────────────────────────────

def plot_eda(df, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Exploratory Data Analysis – Student Performance", fontsize=16, fontweight="bold", y=1.01)

    palette = {"At Risk": "#e74c3c", "Average": "#f39c12", "High Performer": "#27ae60"}

    # 1. Distribution
    ax = axes[0, 0]
    counts = df["performance"].value_counts()
    bars = ax.bar(counts.index, counts.values, color=[palette[c] for c in counts.index], edgecolor="white", linewidth=1.5)
    ax.bar_label(bars, padding=3, fontsize=10)
    ax.set_title("Performance Distribution", fontsize=12, fontweight="bold")
    ax.set_ylabel("Count")

    # 2. Study hours vs performance
    ax = axes[0, 1]
    for perf, grp in df.groupby("performance"):
        ax.hist(grp["study_hours_per_day"], bins=20, alpha=0.7, label=perf, color=palette[perf], edgecolor="white")
    ax.set_title("Study Hours by Performance", fontsize=12, fontweight="bold")
    ax.set_xlabel("Study Hours / Day")
    ax.legend()

    # 3. Attendance vs previous grades scatter
    ax = axes[0, 2]
    for perf, grp in df.groupby("performance"):
        ax.scatter(grp["attendance_rate"], grp["previous_grades"],
                   alpha=0.4, s=20, label=perf, color=palette[perf])
    ax.set_title("Attendance vs Previous Grades", fontsize=12, fontweight="bold")
    ax.set_xlabel("Attendance Rate (%)")
    ax.set_ylabel("Previous Grades")
    ax.legend()

    # 4. Correlation heatmap (numeric only)
    ax = axes[1, 0]
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0, annot=False, linewidths=0.3, fmt=".1f")
    ax.set_title("Feature Correlation Heatmap", fontsize=12, fontweight="bold")

    # 5. Motivation vs performance
    ax = axes[1, 1]
    ct = pd.crosstab(df["motivation_level"], df["performance"], normalize="index") * 100
    ct.plot(kind="bar", ax=ax, color=[palette[c] for c in ct.columns], edgecolor="white", linewidth=0.8)
    ax.set_title("Motivation Level vs Performance (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Motivation Level")
    ax.set_ylabel("Percentage")
    ax.legend(title="Performance", bbox_to_anchor=(1, 1))
    ax.tick_params(axis="x", rotation=0)

    # 6. Sleep hours boxplot
    ax = axes[1, 2]
    order = ["At Risk", "Average", "High Performer"]
    df.boxplot(column="sleep_hours", by="performance", ax=ax,
               boxprops=dict(linewidth=2),
               medianprops=dict(color="black", linewidth=2))
    ax.set_title("Sleep Hours by Performance", fontsize=12, fontweight="bold")
    ax.set_xlabel("Performance Category")
    ax.set_ylabel("Sleep Hours")
    plt.sca(ax)
    plt.title("Sleep Hours by Performance", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out_dir, "eda_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")


# ─────────────────────────────────────────────
# 9. MODEL COMPARISON CHART
# ─────────────────────────────────────────────

def plot_model_comparison(results, out_dir):
    names  = list(results.keys())
    accs   = [results[n]["accuracy"] for n in names]
    aucs   = [results[n]["auc"] for n in names]
    cv_m   = [results[n]["cv_mean"] for n in names]
    cv_s   = [results[n]["cv_std"] for n in names]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - width, accs, width, label="Test Accuracy", color="#3498db", edgecolor="white")
    b2 = ax.bar(x,         aucs, width, label="AUC-ROC",       color="#2ecc71", edgecolor="white")
    b3 = ax.bar(x + width, cv_m, width, label="CV Accuracy",   color="#e67e22", edgecolor="white",
                yerr=cv_s, capsize=5)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b3, fmt="%.3f", padding=2, fontsize=8)
    ax.axhline(0.8, color="red", linestyle="--", alpha=0.4, label="0.8 baseline")
    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")


# ─────────────────────────────────────────────
# 10. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    OUT_DIR = "outputs"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "="*60)
    print("  STUDENT PERFORMANCE ANALYSIS & PREDICTION")
    print("="*60)

    # Generate data
    print("\n[1/6] Generating synthetic student dataset...")
    df = generate_student_data(n_samples=1500)
    df.to_csv(os.path.join(OUT_DIR, "student_data.csv"), index=False)
    print(f"      {len(df)} records generated. Shape: {df.shape}")
    print(f"      Class distribution:\n{df['performance'].value_counts().to_string()}")

    # EDA
    print("\n[2/6] Generating EDA plots...")
    plot_eda(df, OUT_DIR)

    # Preprocess
    print("\n[3/6] Preprocessing...")
    X, y, le, feature_cols = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                         random_state=42, stratify=y)
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")

    # Save artifacts for Streamlit
    joblib.dump(le, os.path.join(OUT_DIR, "label_encoder.pkl"))
    joblib.dump(feature_cols, os.path.join(OUT_DIR, "feature_cols.pkl"))

    # Train
    print("\n[4/6] Training models...")
    models = train_models(X_train, y_train)

    # Evaluate
    print("\n[5/6] Evaluating models...")
    results = evaluate_models(models, X_test, y_test, le)
    plot_model_comparison(results, OUT_DIR)

    for name, res in results.items():
        plot_confusion(res["confusion"], le.classes_, name, OUT_DIR)

    # Save best model (XGBoost)
    best_model = models["XGBoost"]
    joblib.dump(best_model, os.path.join(OUT_DIR, "best_model_xgb.pkl"))
    joblib.dump(models["Random Forest"], os.path.join(OUT_DIR, "model_rf.pkl"))
    print("      Models saved to outputs/")

    # Explainability
    print("\n[6/6] Computing SHAP explainability...")
    for name in ["XGBoost", "Random Forest"]:
        sv = compute_shap(models[name], X_train, X_test, name, OUT_DIR)
        plot_feature_importance(models[name], feature_cols, name, OUT_DIR)

    print("\n" + "="*60)
    print("  ✅ Pipeline complete! Check the 'outputs/' folder.")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()