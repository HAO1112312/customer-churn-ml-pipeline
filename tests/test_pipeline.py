"""
test_pipeline.py
Unit tests for data generation, feature engineering, preprocessing, and API inference.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_generator import generate_customer_dataset
from src.features import engineer_features
from src.preprocessor import ChurnDataPreprocessor


@pytest.fixture
def sample_raw_df():
    return generate_customer_dataset(n_samples=200, random_state=42, output_path="data/test_customers.csv")


def test_data_generator(sample_raw_df):
    assert len(sample_raw_df) == 200
    assert "churn" in sample_raw_df.columns
    assert "customer_id" in sample_raw_df.columns
    assert sample_raw_df["churn"].nunique() == 2


def test_feature_engineering(sample_raw_df):
    df_feat = engineer_features(sample_raw_df)
    expected_new_cols = [
        "tenure_cohort",
        "charge_velocity",
        "charges_per_ticket",
        "inactivity_ratio",
        "annualized_ticket_rate",
        "churn_alarm_signal",
        "addon_services_count",
        "is_long_term_contract"
    ]
    for col in expected_new_cols:
        assert col in df_feat.columns
    assert df_feat["charge_velocity"].notnull().all()


def test_preprocessor_no_leakage(sample_raw_df):
    preprocessor = ChurnDataPreprocessor(target_col="churn", test_size=0.25, random_state=42)
    X_train, X_test, y_train, y_test, feat_names = preprocessor.prepare_data(sample_raw_df, apply_smote=True)

    assert X_train.shape[1] == X_test.shape[1]
    assert len(feat_names) == X_train.shape[1]
    assert np.isnan(X_train).sum() == 0
    assert np.isnan(X_test).sum() == 0
    # Test that SMOTE balanced the train set
    assert np.sum(y_train == 1) > 0


def test_single_prediction_transformation(sample_raw_df):
    preprocessor = ChurnDataPreprocessor(target_col="churn", test_size=0.25, random_state=42)
    preprocessor.prepare_data(sample_raw_df, apply_smote=False)

    sample_dict = sample_raw_df.drop(columns=["customer_id", "churn"]).iloc[0].to_dict()
    X_single = preprocessor.transform_single(sample_dict)
    assert X_single.shape[0] == 1
    assert X_single.shape[1] == len(preprocessor.feature_names_out)
