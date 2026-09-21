"""
segmentation.py
Customer Value & Risk Segmentation using K-Means Clustering.
Synthesizes ML Churn Probabilities with Customer Lifetime Value (CLV / Spend)
to generate actionable business intervention matrices.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class CustomerValueSegmentation:
    """
    Groups customers into strategic business quadrants based on
    Predictive Churn Risk vs Lifetime Revenue Potential.
    """

    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_labels_map = {}

    def fit_predict(self, df_with_probs: pd.DataFrame) -> pd.DataFrame:
        """
        Segment customers into 4 strategic categories based on:
        - churn_prob (from ML model)
        - monthly_charges (spend proxy)
        - tenure_months (loyalty proxy)
        """
        df = df_with_probs.copy()
        features = ["churn_prob", "monthly_charges", "tenure_months"]
        X_scaled = self.scaler.fit_transform(df[features])

        cluster_ids = self.kmeans.fit_predict(X_scaled)
        df["cluster_id"] = cluster_ids

        # Characterize clusters to map human-readable business personas
        summary = df.groupby("cluster_id")[features].mean()

        labels = {}
        for cid, row in summary.iterrows():
            prob = row["churn_prob"]
            spend = row["monthly_charges"]
            
            if prob >= 0.50 and spend >= 65:
                labels[cid] = "High-Value At Risk (Priority 1)"
            elif prob >= 0.50 and spend < 65:
                labels[cid] = "Low-Value Churner (Automated Outreach)"
            elif prob < 0.50 and spend >= 65:
                labels[cid] = "VIP Champions (Cross-Sell / Loyalty)"
            else:
                labels[cid] = "Stable Regulars (Low Touch)"

        self.cluster_labels_map = labels
        df["segment_name"] = df["cluster_id"].map(labels)
        return df

    def calculate_business_roi(
        self,
        df_segmented: pd.DataFrame,
        save_rate: float = 0.25,
        intervention_cost_per_vip: float = 35.0
    ) -> dict:
        """
        Simulate annual revenue preserved by deploying the predictive model.
        Assumes proactive concierge team retains 25% of flagged High-Value at-risk customers.
        """
        vip_risk = df_segmented[df_segmented["segment_name"] == "High-Value At Risk (Priority 1)"]
        n_targeted = len(vip_risk)
        monthly_revenue_at_risk = vip_risk["monthly_charges"].sum()
        annual_revenue_at_risk = monthly_revenue_at_risk * 12.0

        n_saved = int(n_targeted * save_rate)
        annual_revenue_preserved = annual_revenue_at_risk * save_rate
        total_campaign_cost = n_targeted * intervention_cost_per_vip
        net_roi_dollar = annual_revenue_preserved - total_campaign_cost
        roi_percentage = (net_roi_dollar / total_campaign_cost) * 100 if total_campaign_cost > 0 else 0

        impact = {
            "targeted_vip_customers": n_targeted,
            "estimated_customers_saved": n_saved,
            "annual_revenue_at_risk": float(annual_revenue_at_risk),
            "annual_revenue_saved": float(annual_revenue_preserved),
            "campaign_cost": float(total_campaign_cost),
            "net_roi_profit": float(net_roi_dollar),
            "roi_percentage": float(roi_percentage)
        }
        return impact

    def export_segmentation_visual(self, df_segmented: pd.DataFrame, output_path: str = "reports/figures/08_segmentation_matrix.png"):
        """Plot scatter matrix of Churn Risk vs Monthly Spend colored by Strategic Segments."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.figure(figsize=(10, 6))

        palette = {
            "High-Value At Risk (Priority 1)": "#e74c3c",
            "Low-Value Churner (Automated Outreach)": "#e67e22",
            "VIP Champions (Cross-Sell / Loyalty)": "#2ecc71",
            "Stable Regulars (Low Touch)": "#3498db"
        }

        sns.scatterplot(
            data=df_segmented,
            x="churn_prob",
            y="monthly_charges",
            hue="segment_name",
            palette=palette,
            alpha=0.6,
            s=40
        )
        plt.axvline(x=0.5, color="grey", linestyle="--", alpha=0.7)
        plt.axhline(y=65.0, color="grey", linestyle="--", alpha=0.7)
        plt.title("Actionable Customer Segmentation: Churn Risk vs Monthly Value", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted Churn Probability (ML Model)")
        plt.ylabel("Monthly Spend ($)")
        plt.legend(title="Strategic Segment", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[+] Saved segmentation visual to {output_path}")

    def save(self, filepath: str = "models/segmentation_kmeans.pkl"):
        """Save fitted segmenter to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[+] Segmentation model saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/segmentation_kmeans.pkl") -> "CustomerValueSegmentation":
        """Load fitted segmenter from disk."""
        return joblib.load(filepath)
