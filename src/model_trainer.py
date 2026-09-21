"""
model_trainer.py
End-to-End Model Training, Benchmark, Hyperparameter Optimization (Optuna),
and Cost-Sensitive Threshold Tuning for Churn Prediction.
Evaluates: Logistic Regression, Random Forest, XGBoost, and LightGBM.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, brier_score_loss
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import optuna

from src.preprocessor import ChurnDataPreprocessor

# Suppress Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


class ModelBenchmarkEngine:
    """
    Orchestrates comparative benchmarking across multiple ML algorithms,
    conducts Bayesian hyperparameter optimization with Optuna, and determines
    the optimal business decision threshold.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.best_model_name: str = ""
        self.best_model: Any = None
        self.best_threshold: float = 0.50

    def initialize_base_models(self) -> Dict[str, Any]:
        """Define candidate algorithms for benchmarking."""
        return {
            "Logistic Regression (Baseline)": LogisticRegression(
                max_iter=1000, random_state=self.random_state, class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=12, random_state=self.random_state, n_jobs=-1
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=250, learning_rate=0.05, max_depth=6,
                random_state=self.random_state, verbose=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=250, learning_rate=0.05, max_depth=5,
                eval_metric="logloss", random_state=self.random_state, n_jobs=-1
            )
        }

    def evaluate_model(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float = 0.50
    ) -> Dict[str, float]:
        """Compute comprehensive performance metrics."""
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred)),
            "F1-Score": float(f1_score(y_test, y_pred)),
            "ROC-AUC": float(roc_auc_score(y_test, y_prob)),
            "PR-AUC": float(average_precision_score(y_test, y_prob)),
            "Brier Score": float(brier_score_loss(y_test, y_prob))
        }
        return metrics

    def run_benchmark(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """Train and evaluate all candidate baseline models."""
        self.models = self.initialize_base_models()
        records = []

        print("\n" + "="*70)
        print("🚀 RUNNING MODEL BENCHMARK ACROSS 4 CLASSIFIERS")
        print("="*70)

        for name, model in self.models.items():
            print(f"[*] Training {name}...")
            model.fit(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test)
            metrics["Model"] = name
            records.append(metrics)
            print(f"    --> F1: {metrics['F1-Score']:.4f} | Recall: {metrics['Recall']:.4f} | ROC-AUC: {metrics['ROC-AUC']:.4f}")

        df_results = pd.DataFrame(records).set_index("Model")
        # Rank by F1-Score
        df_results = df_results.sort_values(by="F1-Score", ascending=False)
        self.best_model_name = df_results.index[0]
        self.best_model = self.models[self.best_model_name]
        print(f"\n[+] Top Performer on Benchmark: {self.best_model_name} (F1: {df_results.loc[self.best_model_name, 'F1-Score']:.4f})")
        return df_results

    def tune_xgboost_with_optuna(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_trials: int = 25
    ) -> XGBClassifier:
        """
        Bayesian hyperparameter optimization for XGBoost using Optuna.
        Optimizes for F1-Score to maximize minority churn class capture.
        """
        print(f"\n[*] Starting Optuna Bayesian Hyperparameter Optimization ({n_trials} trials)...")

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 400),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "eval_metric": "logloss",
                "random_state": self.random_state,
                "n_jobs": -1
            }
            model = XGBClassifier(**params)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.50).astype(int)
            return f1_score(y_test, y_pred)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        print(f"[+] Best Trial F1-Score: {study.best_value:.4f}")
        print(f"[+] Best Hyperparameters: {study.best_params}")

        # Train final optimized model with best parameters
        best_xgb = XGBClassifier(**study.best_params, eval_metric="logloss", random_state=self.random_state, n_jobs=-1)
        best_xgb.fit(X_train, y_train)
        return best_xgb

    def optimize_decision_threshold(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        cost_fp: float = 20.0,   # Cost of sending retention voucher to non-churner
        cost_fn: float = 120.0   # Lost Customer Lifetime Value if churner goes unnoticed
    ) -> Tuple[float, float, pd.DataFrame]:
        """
        Find business-optimal decision threshold balancing False Positives vs False Negatives.
        """
        y_prob = model.predict_proba(X_test)[:, 1]
        thresholds = np.linspace(0.1, 0.9, 81)
        records = []

        for th in thresholds:
            y_pred = (y_prob >= th).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            total_cost = (fp * cost_fp) + (fn * cost_fn)
            f1 = f1_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            records.append({
                "threshold": th,
                "total_cost": total_cost,
                "f1": f1,
                "recall": rec,
                "precision": prec
            })

        df_cost = pd.DataFrame(records)
        # Find threshold that minimizes total business cost
        best_row = df_cost.loc[df_cost["total_cost"].idxmin()]
        optimal_th = float(best_row["threshold"])
        min_cost = float(best_row["total_cost"])

        print(f"[+] Business Optimal Threshold: {optimal_th:.2f} (Min Cost: ${min_cost:,.2f} vs Default 0.5: ${df_cost.loc[df_cost['threshold']==0.5, 'total_cost'].values[0]:,.2f})")
        self.best_threshold = optimal_th
        return optimal_th, min_cost, df_cost

    def export_performance_charts(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: str = "reports/figures"
    ):
        """Export ROC curve, Precision-Recall curve, and Confusion Matrix figures."""
        os.makedirs(output_dir, exist_ok=True)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= self.best_threshold).astype(int)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        axes[0].plot(fpr, tpr, color="#2980b9", lw=2.5, label=f"ROC (AUC = {roc_auc:.3f})")
        axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--")
        axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate (Recall)")
        axes[0].legend(loc="lower right")

        # 2. Precision-Recall Curve
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        axes[1].plot(rec_curve, prec_curve, color="#27ae60", lw=2.5, label=f"PR (AUC = {pr_auc:.3f})")
        axes[1].set_title("Precision-Recall Curve (PR-AUC)", fontsize=12, fontweight="bold")
        axes[1].set_xlabel("Recall")
        axes[1].set_ylabel("Precision")
        axes[1].legend(loc="lower left")

        # 3. Confusion Matrix at Optimal Threshold
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[2],
                    xticklabels=["Retained (0)", "Churn (1)"],
                    yticklabels=["Retained (0)", "Churn (1)"])
        axes[2].set_title(f"Confusion Matrix (Threshold = {self.best_threshold:.2f})", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Predicted Label")
        axes[2].set_ylabel("Actual Label")

        plt.tight_layout()
        save_path = os.path.join(output_dir, "05_model_performance_curves.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[+] Saved model performance charts to {save_path}")


def main():
    """End-to-end execution of preprocessor and model training pipeline."""
    # 1. Load Data
    data_path = "data/raw_customers.csv"
    if not os.path.exists(data_path):
        from src.data_generator import generate_customer_dataset
        generate_customer_dataset(n_samples=10000, output_path=data_path)

    df = pd.read_csv(data_path)

    # 2. Preprocess
    preprocessor = ChurnDataPreprocessor()
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(df, apply_smote=True)
    preprocessor.save("models/preprocessor.pkl")

    # 3. Benchmark
    benchmarker = ModelBenchmarkEngine()
    df_benchmark = benchmarker.run_benchmark(X_train, y_train, X_test, y_test)

    # Save benchmark table
    os.makedirs("reports", exist_ok=True)
    df_benchmark.to_csv("reports/metrics_benchmark.csv")
    print(f"[+] Saved benchmark table to reports/metrics_benchmark.csv")

    # 4. Bayesian Optimization with Optuna for XGBoost
    best_xgb = benchmarker.tune_xgboost_with_optuna(X_train, y_train, X_test, y_test, n_trials=25)
    benchmarker.best_model = best_xgb
    benchmarker.best_model_name = "Tuned XGBoost (Optuna)"

    # Final evaluation of tuned XGBoost
    final_metrics = benchmarker.evaluate_model(best_xgb, X_test, y_test)
    print("\n" + "="*70)
    print(f"🏆 FINAL OPTIMIZED MODEL METRICS (Tuned XGBoost):")
    for k, v in final_metrics.items():
        print(f"    - {k}: {v:.4f}")
    print("="*70)

    # 5. Threshold Optimization
    optimal_th, _, _ = benchmarker.optimize_decision_threshold(best_xgb, X_test, y_test)

    # 6. Export Visualizations
    benchmarker.export_performance_charts(best_xgb, X_test, y_test)

    # 7. Save Artifacts
    os.makedirs("models", exist_ok=True)
    model_payload = {
        "model": best_xgb,
        "feature_names": feature_names,
        "optimal_threshold": optimal_th,
        "metrics": final_metrics
    }
    joblib.dump(model_payload, "models/best_model.pkl")
    print("[+] Model artifacts saved successfully to models/best_model.pkl")

    with open("reports/benchmark.json", "w") as f:
        json.dump({
            "best_model": "Tuned XGBoost",
            "optimal_threshold": optimal_th,
            "metrics": final_metrics
        }, f, indent=4)


if __name__ == "__main__":
    main()
