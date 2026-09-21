"""
explainability.py
Model Interpretability & Explainability module using SHAP (SHapley Additive exPlanations).
Generates global feature impact plots (Beeswarm, Bar) and local waterfall explanations
to translate complex model outputs into actionable business intelligence.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from typing import List, Dict, Any


class ChurnModelExplainer:
    """
    Interprets tree-based churn models using SHAP TreeExplainer.
    Provides global insights and individual-level reason codes.
    """

    def __init__(self, model_path: str = "models/best_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        payload = joblib.load(model_path)
        self.model = payload["model"]
        self.feature_names = payload["feature_names"]
        self.explainer = shap.TreeExplainer(self.model)

    def generate_global_explanations(
        self,
        X_sample: np.ndarray,
        output_dir: str = "reports/figures",
        max_display: int = 15
    ):
        """
        Compute SHAP values for a sample and export beeswarm & bar importance plots.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"[*] Calculating SHAP values for {len(X_sample)} test samples...")
        shap_values = self.explainer(X_sample)

        # 1. Beeswarm Plot (Global impact direction & magnitude)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values.values,
            X_sample,
            feature_names=self.feature_names,
            max_display=max_display,
            show=False
        )
        plt.title("SHAP Global Summary: Feature Impact on Churn Probability", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        beeswarm_path = os.path.join(output_dir, "06_shap_beeswarm.png")
        plt.savefig(beeswarm_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved SHAP beeswarm plot to {beeswarm_path}")

        # 2. Local Waterfall Plot for a representative high-risk churner
        # Find index with highest predicted churn probability
        probs = self.model.predict_proba(X_sample)[:, 1]
        high_risk_idx = int(np.argmax(probs))

        plt.figure(figsize=(10, 6))
        sample_explanation = shap.Explanation(
            values=shap_values.values[high_risk_idx],
            base_values=shap_values.base_values[high_risk_idx] if hasattr(shap_values.base_values, "__len__") else shap_values.base_values,
            data=X_sample[high_risk_idx],
            feature_names=self.feature_names
        )
        shap.plots.waterfall(sample_explanation, max_display=10, show=False)
        plt.title(f"SHAP Waterfall: Churn Drivers for High-Risk Customer (Prob: {probs[high_risk_idx]:.1%})", fontsize=12, fontweight="bold", pad=12)
        plt.tight_layout()
        waterfall_path = os.path.join(output_dir, "07_shap_waterfall_high_risk.png")
        plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[+] Saved SHAP waterfall plot to {waterfall_path}")

    def explain_individual(self, X_single: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """
        Explain a single customer prediction for the REST API / Dashboard.
        Returns top risk factors (pushing towards churn) and retention drivers.
        """
        shap_vals = self.explainer.shap_values(X_single)[0]
        feature_impacts = list(zip(self.feature_names, shap_vals))

        # Top positive contributors (pushing towards Churn = 1)
        top_churn_drivers = sorted(
            [item for item in feature_impacts if item[1] > 0],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Top negative contributors (pushing towards Retention = 0)
        top_retention_drivers = sorted(
            [item for item in feature_impacts if item[1] < 0],
            key=lambda x: x[1]
        )[:top_k]

        return {
            "top_churn_drivers": [{"feature": f, "impact_score": float(v)} for f, v in top_churn_drivers],
            "top_retention_drivers": [{"feature": f, "impact_score": float(v)} for f, v in top_retention_drivers]
        }


if __name__ == "__main__":
    from src.preprocessor import ChurnDataPreprocessor
    import pandas as pd

    df = pd.read_csv("data/raw_customers.csv")
    preprocessor = ChurnDataPreprocessor.load("models/preprocessor.pkl")
    _, X_test, _, _, _ = preprocessor.prepare_data(df, apply_smote=False)

    explainer = ChurnModelExplainer("models/best_model.pkl")
    explainer.generate_global_explanations(X_test[:500])
