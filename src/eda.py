"""
eda.py
Exploratory Data Analysis (EDA) module.
Analyzes feature distributions, missingness, churn correlation,
and exports high-resolution visual reports for business stakeholders.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CLI environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def run_eda(data_path: str = "data/raw_customers.csv", output_dir: str = "reports/figures"):
    """
    Run comprehensive statistical analysis and generate publish-ready figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 10})

    print(f"[*] Loading data for EDA from {data_path}...")
    df = pd.read_csv(data_path)
    n_total = len(df)
    churn_cnt = df["churn"].sum()
    churn_pct = (churn_cnt / n_total) * 100

    print(f"[+] Total Records: {n_total:,} | Churn: {churn_cnt:,} ({churn_pct:.2f}%)")

    # -------------------------------------------------------------
    # Figure 1: Target Class Imbalance
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    palette = ["#2ecc71", "#e74c3c"]
    sns.countplot(data=df, x="churn", palette=palette, ax=ax)
    ax.set_title("Customer Churn Class Distribution (Imbalance: ~18% Churn)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Churn Status (0: Retained, 1: Churned)")
    ax.set_ylabel("Customer Count")
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height:,}\n({height/n_total:.1%})",
                    (p.get_x() + p.get_width() / 2., height / 2),
                    ha='center', va='center', color='white', fontweight='bold', fontsize=11)
    plt.tight_layout()
    fig_path1 = os.path.join(output_dir, "01_class_imbalance.png")
    plt.savefig(fig_path1, dpi=300)
    plt.close()
    print(f"[+] Saved figure: {fig_path1}")

    # -------------------------------------------------------------
    # Figure 2: Numerical Correlation Matrix with Churn
    # -------------------------------------------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmin=-0.6, vmax=0.6,
                square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
    ax.set_title("Feature Correlation Matrix (Focus on Correlation with Churn)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig_path2 = os.path.join(output_dir, "02_correlation_matrix.png")
    plt.savefig(fig_path2, dpi=300)
    plt.close()
    print(f"[+] Saved figure: {fig_path2}")

    # -------------------------------------------------------------
    # Figure 3: Contract Type vs Churn Rate
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    contract_churn = df.groupby("contract_type")["churn"].agg(["count", "mean"]).reset_index()
    contract_churn["mean"] *= 100

    bars = sns.barplot(data=contract_churn, x="contract_type", y="mean", palette="Blues_r", ax=ax)
    ax.set_title("Churn Rate by Contract Type (Month-to-Month High Risk)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Contract Agreement")
    ax.set_ylabel("Churn Probability (%)")
    ax.set_ylim(0, 50)
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height() + 1.2),
                    ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    fig_path3 = os.path.join(output_dir, "03_contract_vs_churn.png")
    plt.savefig(fig_path3, dpi=300)
    plt.close()
    print(f"[+] Saved figure: {fig_path3}")

    # -------------------------------------------------------------
    # Figure 4: Support Ticket Frequency & Inactivity vs Churn
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Support Tickets vs Churn
    sns.barplot(data=df, x="support_tickets", y="churn", palette="Reds", ax=ax1, ci=None)
    ax1.set_title("Churn Probability vs Number of Support Tickets", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Customer Support Tickets in Last 6 Months")
    ax1.set_ylabel("Churn Rate")

    # Days Since Last Login (Recency) Distribution
    sns.kdeplot(data=df[df["churn"] == 0]["days_since_last_login"], label="Retained (0)", color="#2ecc71", shade=True, ax=ax2)
    sns.kdeplot(data=df[df["churn"] == 1]["days_since_last_login"], label="Churned (1)", color="#e74c3c", shade=True, ax=ax2)
    ax2.set_title("Recency: Days Since Last Login Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Days Since Last Login")
    ax2.legend()

    plt.tight_layout()
    fig_path4 = os.path.join(output_dir, "04_behavioral_signals.png")
    plt.savefig(fig_path4, dpi=300)
    plt.close()
    print(f"[+] Saved figure: {fig_path4}")

    print("[+] Automated EDA completed successfully! All charts exported to reports/figures/")


if __name__ == "__main__":
    run_eda()
