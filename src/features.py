"""
features.py
Feature Engineering module for Customer Churn Modeling.
Constructs domain-specific business features including RFM indicators,
engagement velocity, support intensity, and tenure-based cohorts.
"""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate rich domain features from raw customer attributes.
    Returns an enriched DataFrame.
    """
    data = df.copy()

    # 1. Tenure Cohorts & Lifecycle Stage
    # Early tenure customers (<= 6 months) often have highest churn risk
    data["tenure_cohort"] = pd.cut(
        data["tenure_months"],
        bins=[-1, 6, 24, 48, 120],
        labels=["Newbie_0_6m", "Developing_7_24m", "Mature_25_48m", "Loyal_48m_plus"]
    )

    # 2. Spend & Monetary Dynamics
    # Ratio between calculated expected spend vs actual total charges
    expected_charges = data["tenure_months"] * data["monthly_charges"]
    data["charge_velocity"] = np.where(
        expected_charges > 0,
        data["total_charges"] / (expected_charges + 1e-5),
        1.0
    )
    data["charge_velocity"] = np.clip(data["charge_velocity"], 0.5, 2.0)

    # Average monthly charge per support ticket (Cost of serving customer)
    data["charges_per_ticket"] = data["monthly_charges"] / (data["support_tickets"] + 1)

    # 3. Engagement & Behavioral Risk Features
    # Recency-to-Frequency Ratio: High days inactive / low frequency = high risk
    data["inactivity_ratio"] = data["days_since_last_login"] / (data["login_frequency"] + 1)

    # Support Ticket Intensity: Tickets relative to tenure in years
    tenure_years = np.maximum(data["tenure_months"] / 12.0, 0.25)
    data["annualized_ticket_rate"] = data["support_tickets"] / tenure_years

    # High Risk Warning Flag: Negative usage drop + low satisfaction + support complaints
    data["churn_alarm_signal"] = (
        (data["usage_drop_rate"] > 20).astype(int) +
        (data["support_tickets"] >= 3).astype(int) +
        (data["satisfaction_score"] <= 2).fillna(False).astype(int)
    )

    # 4. Service Ecosystem Depth
    # Number of active security/support add-on services
    service_cols = ["online_security", "tech_support"]
    data["addon_services_count"] = 0
    for col in service_cols:
        if col in data.columns:
            data["addon_services_count"] += (data[col] == "Yes").astype(int)

    # 5. Long-term Commitment Flag
    data["is_long_term_contract"] = data["contract_type"].isin(["One year", "Two year"]).astype(int)

    return data
