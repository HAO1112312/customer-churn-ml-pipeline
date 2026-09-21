"""
data_generator.py
Synthetic data generator for Customer Churn & Business Intelligence pipeline.
Generates realistic customer demographics, usage metrics, engagement signals, and ground-truth churn labels.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path


def generate_customer_dataset(
    n_samples: int = 10000,
    random_state: int = 42,
    output_path: str = "data/raw_customers.csv"
) -> pd.DataFrame:
    """
    Generate synthetic customer churn dataset with realistic non-linear relationships,
    imbalanced churn distribution (~18%), missing values, and outliers.
    """
    np.random.seed(random_state)
    print(f"[*] Generating {n_samples:,} synthetic customer records...")

    # 1. Identifiers & Demographics
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_samples + 1)]
    age = np.random.randint(18, 75, size=n_samples)
    has_partner = np.random.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = np.where(has_partner == "Yes", np.random.choice(["Yes", "No"], size=n_samples, p=[0.45, 0.55]), "No")
    
    # 2. Account & Contract Information
    tenure_months = np.random.exponential(scale=24, size=n_samples).astype(int)
    tenure_months = np.clip(tenure_months, 1, 72)

    contract_choices = ["Month-to-month", "One year", "Two year"]
    # Long tenure customers are more likely to have multi-year contracts
    contract_p_base = np.array([0.55, 0.25, 0.20])
    contract_type = []
    for t in tenure_months:
        if t < 12:
            p = [0.80, 0.15, 0.05]
        elif t < 36:
            p = [0.50, 0.30, 0.20]
        else:
            p = [0.25, 0.35, 0.40]
        contract_type.append(np.random.choice(contract_choices, p=p))
    contract_type = np.array(contract_type)

    payment_method = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n_samples,
        p=[0.34, 0.18, 0.24, 0.24]
    )
    paperless_billing = np.random.choice(["Yes", "No"], size=n_samples, p=[0.60, 0.40])

    # 3. Product & Services
    internet_service = np.random.choice(["Fiber optic", "DSL", "No"], size=n_samples, p=[0.45, 0.35, 0.20])
    online_security = np.where(internet_service != "No", np.random.choice(["Yes", "No"], size=n_samples, p=[0.40, 0.60]), "No internet service")
    tech_support = np.where(internet_service != "No", np.random.choice(["Yes", "No"], size=n_samples, p=[0.38, 0.62]), "No internet service")
    streaming_tv = np.where(internet_service != "No", np.random.choice(["Yes", "No"], size=n_samples, p=[0.50, 0.50]), "No internet service")

    # 4. Behavioral & Engagement Metrics (Crucial for Data Science & Churn)
    monthly_charges = np.random.normal(loc=65, scale=25, size=n_samples)
    monthly_charges = np.clip(monthly_charges, 18.5, 125.0).round(2)
    # Total charges with slight realistic variation
    total_charges = (tenure_months * monthly_charges * np.random.uniform(0.95, 1.05, size=n_samples)).round(2)

    # Days since last login (Recency)
    days_since_last_login = np.random.geometric(p=0.08, size=n_samples)
    days_since_last_login = np.clip(days_since_last_login, 1, 90)

    # Login frequency per month (Frequency)
    login_frequency = np.random.poisson(lam=16, size=n_samples)
    login_frequency = np.clip(login_frequency, 0, 50)

    # Usage drop percentage in the last 30 days vs 90 days prior (Spend/Engagement Velocity)
    usage_drop_rate = np.random.normal(loc=0.0, scale=25.0, size=n_samples).round(1)

    # Customer support tickets submitted in the last 6 months
    support_tickets = np.random.negative_binomial(n=1, p=0.35, size=n_samples)
    support_tickets = np.clip(support_tickets, 0, 10)

    # CSAT Customer Satisfaction score (1 to 5)
    satisfaction_score = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.12, 0.18, 0.30, 0.25, 0.15])

    # 5. Non-linear Ground Truth Churn Probability Calculation (Realistic Simulation)
    logit = (
        -1.8
        + 1.5 * (contract_type == "Month-to-month")
        - 0.8 * (contract_type == "Two year")
        - 0.04 * tenure_months
        + 0.35 * support_tickets
        + 0.02 * days_since_last_login
        - 0.04 * login_frequency
        + 0.025 * np.maximum(usage_drop_rate, 0)
        + 0.6 * (satisfaction_score <= 2)
        - 0.5 * (satisfaction_score >= 4)
        + 0.4 * (internet_service == "Fiber optic")
        - 0.3 * (online_security == "Yes")
        + 0.01 * (monthly_charges - 65)
        + np.random.normal(0, 0.45, size=n_samples)  # Environmental noise
    )
    churn_prob = 1.0 / (1.0 + np.exp(-logit))
    churn = (churn_prob > 0.50).astype(int)

    # 6. Inject Realistic Data Flaws (Missing Values & Outliers)
    # A few total_charges are missing (common in new customers)
    missing_mask_total = np.random.rand(n_samples) < 0.015
    total_charges[missing_mask_total] = np.nan

    # A few missing satisfaction scores (unanswered surveys)
    satisfaction_with_na = satisfaction_score.astype(float)
    missing_mask_sat = np.random.rand(n_samples) < 0.035
    satisfaction_with_na[missing_mask_sat] = np.nan

    # A few anomalous monthly charge values (outliers)
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.005), replace=False)
    monthly_charges[outlier_indices] = monthly_charges[outlier_indices] * 2.5

    # 7. Construct DataFrame
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "has_partner": has_partner,
        "dependents": dependents,
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "payment_method": payment_method,
        "paperless_billing": paperless_billing,
        "internet_service": internet_service,
        "online_security": online_security,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "days_since_last_login": days_since_last_login,
        "login_frequency": login_frequency,
        "usage_drop_rate": usage_drop_rate,
        "support_tickets": support_tickets,
        "satisfaction_score": satisfaction_with_na,
        "churn": churn
    })

    # Save to disk
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    churn_rate = df["churn"].mean() * 100
    print(f"[+] Dataset saved to: {output_path}")
    print(f"[+] Total Records: {len(df):,} | Churn Rate: {churn_rate:.2f}% (Class Imbalance ~{100-churn_rate:.1f}% vs {churn_rate:.1f}%)")
    print(f"[+] Features: {df.shape[1] - 2} predictors, 1 ID, 1 Target ('churn')")
    return df


if __name__ == "__main__":
    generate_customer_dataset(n_samples=10000, output_path="data/raw_customers.csv")
