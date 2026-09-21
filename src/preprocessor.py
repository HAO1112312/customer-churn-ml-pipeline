"""
preprocessor.py
Data preprocessing pipeline with ColumnTransformer, median/mode imputers,
One-Hot Encoding, RobustScaler, and SMOTE resampling for imbalanced classification.
Strictly ensures zero data leakage between train and test sets.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from imblearn.over_sampling import SMOTE

from src.features import engineer_features


class ChurnDataPreprocessor:
    """
    Production-ready Preprocessor for Customer Churn Modeling.
    Handles schema validation, feature engineering, missing values,
    scaling, encoding, and SMOTE class balancing.
    """

    def __init__(self, target_col: str = "churn", test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.preprocessor_pipeline: Optional[ColumnTransformer] = None
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.feature_names_out: List[str] = []

    def identify_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Identify numerical and categorical feature columns excluding IDs and target."""
        drop_cols = ["customer_id", self.target_col]
        feature_cols = [c for c in df.columns if c not in drop_cols]

        num_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df[feature_cols].select_dtypes(exclude=[np.number]).columns.tolist()

        return num_cols, cat_cols

    def build_transformer(self, numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
        """Create sklearn ColumnTransformer pipeline."""
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())  # Robust to outliers in charges and drop rates
        ])

        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))
        ])

        transformer = ColumnTransformer(
            transformers=[
                ("num", num_transformer, numeric_cols),
                ("cat", cat_transformer, categorical_cols)
            ],
            remainder="drop"
        )
        return transformer

    def prepare_data(
        self,
        df: pd.DataFrame,
        apply_smote: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Execute full end-to-end split, transform, and resample workflow.
        Returns:
            X_train_final, X_test_transformed, y_train_final, y_test, feature_names
        """
        # Step 1: Feature Engineering
        print("[*] Running domain feature engineering...")
        df_featured = engineer_features(df)

        # Step 2: Separate X and y
        X = df_featured.drop(columns=[self.target_col, "customer_id"], errors="ignore")
        y = df_featured[self.target_col].values

        self.numeric_features, self.categorical_features = self.identify_feature_types(
            df_featured.drop(columns=["customer_id"], errors="ignore")
        )

        # Step 3: Stratified Train-Test Split (BEFORE any transformations to prevent leakage)
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        print(f"[+] Train shape: {X_train_raw.shape} | Test shape: {X_test_raw.shape}")
        print(f"[+] Train churn rate: {y_train.mean():.2%} | Test churn rate: {y_test.mean():.2%}")

        # Step 4: Fit ColumnTransformer ONLY on Training set
        self.preprocessor_pipeline = self.build_transformer(self.numeric_features, self.categorical_features)
        X_train_transformed = self.preprocessor_pipeline.fit_transform(X_train_raw)
        X_test_transformed = self.preprocessor_pipeline.transform(X_test_raw)

        # Retrieve transformed feature names
        num_names = self.numeric_features
        cat_encoder = self.preprocessor_pipeline.named_transformers_["cat"].named_steps["onehot"]
        cat_names = cat_encoder.get_feature_names_out(self.categorical_features).tolist()
        self.feature_names_out = num_names + cat_names

        # Step 5: Class Balancing via SMOTE (Applied ONLY to X_train)
        if apply_smote:
            print("[*] Applying SMOTE oversampling on training fold to combat class imbalance...")
            smote = SMOTE(random_state=self.random_state, sampling_strategy=0.75)
            X_train_final, y_train_final = smote.fit_resample(X_train_transformed, y_train)
            print(f"[+] After SMOTE - Train samples: {X_train_final.shape[0]:,} (Class 1: {np.sum(y_train_final==1):,})")
        else:
            X_train_final, y_train_final = X_train_transformed, y_train

        return X_train_final, X_test_transformed, y_train_final, y_test, self.feature_names_out

    def transform_single(self, input_dict: dict) -> np.ndarray:
        """Transform a single customer record for online inference API."""
        if self.preprocessor_pipeline is None:
            raise ValueError("Preprocessor has not been fitted yet.")
        df_single = pd.DataFrame([input_dict])
        df_featured = engineer_features(df_single)
        df_ready = df_featured.drop(columns=[self.target_col, "customer_id"], errors="ignore")
        return self.preprocessor_pipeline.transform(df_ready)

    def save(self, filepath: str = "models/preprocessor.pkl"):
        """Save fitted preprocessor to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[+] Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/preprocessor.pkl") -> "ChurnDataPreprocessor":
        """Load fitted preprocessor from disk."""
        return joblib.load(filepath)
