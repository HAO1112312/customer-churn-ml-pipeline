"""
run_pipeline.py
Master orchestrator script: Runs the complete End-to-End Machine Learning Pipeline
from raw data generation to training, SHAP explainability, and customer segmentation.
"""

import os
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_generator import generate_customer_dataset
from src.eda import run_eda
from src.preprocessor import ChurnDataPreprocessor
from src.model_trainer import ModelBenchmarkEngine
from src.explainability import ChurnModelExplainer
from src.segmentation import CustomerValueSegmentation
import pandas as pd
import joblib


def main():
    start_time = time.time()
    print("\n" + "="*80)
    print("🚀 STARTING END-TO-END CUSTOMER CHURN MACHINE LEARNING PIPELINE")
    print("="*80)

    # 1. Generate Synthetic Data
    data_path = os.path.join(PROJECT_ROOT, "data", "raw_customers.csv")
    print("\n[STEP 1/5] DATA GENERATION & INGESTION")
    df = generate_customer_dataset(n_samples=10000, output_path=data_path)

    # 2. Automated EDA
    print("\n[STEP 2/5] AUTOMATED EXPLORATORY DATA ANALYSIS (EDA)")
    run_eda(data_path=data_path, output_dir=os.path.join(PROJECT_ROOT, "reports", "figures"))

    # 3. Preprocessing & SMOTE Balancing
    print("\n[STEP 3/5] FEATURE ENGINEERING & PREPROCESSING (WITH SMOTE)")
    preprocessor = ChurnDataPreprocessor(target_col="churn", test_size=0.20, random_state=42)
    X_train, X_test, y_train, y_test, feature_names = preprocessor.prepare_data(df, apply_smote=True)
    preprocessor.save(os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"))

    # 4. Multi-Model Benchmark & Optuna Tuning
    print("\n[STEP 4/5] MULTI-MODEL BENCHMARKING & OPTUNA HYPERPARAMETER TUNING")
    benchmarker = ModelBenchmarkEngine(random_state=42)
    df_benchmark = benchmarker.run_benchmark(X_train, y_train, X_test, y_test)
    
    os.makedirs(os.path.join(PROJECT_ROOT, "reports"), exist_ok=True)
    df_benchmark.to_csv(os.path.join(PROJECT_ROOT, "reports", "metrics_benchmark.csv"))

    # Optuna tuning for XGBoost
    best_xgb = benchmarker.tune_xgboost_with_optuna(X_train, y_train, X_test, y_test, n_trials=15)
    benchmarker.best_model = best_xgb
    final_metrics = benchmarker.evaluate_model(best_xgb, X_test, y_test)
    optimal_th, _, _ = benchmarker.optimize_decision_threshold(best_xgb, X_test, y_test)
    benchmarker.export_performance_charts(best_xgb, X_test, y_test, output_dir=os.path.join(PROJECT_ROOT, "reports", "figures"))

    # Save best model
    model_payload = {
        "model": best_xgb,
        "feature_names": feature_names,
        "optimal_threshold": optimal_th,
        "metrics": final_metrics
    }
    joblib.dump(model_payload, os.path.join(PROJECT_ROOT, "models", "best_model.pkl"))

    # 5. SHAP Explainability & Customer Segmentation
    print("\n[STEP 5/5] SHAP EXPLAINABILITY & CUSTOMER VALUE SEGMENTATION")
    explainer = ChurnModelExplainer(os.path.join(PROJECT_ROOT, "models", "best_model.pkl"))
    explainer.generate_global_explanations(X_test[:400], output_dir=os.path.join(PROJECT_ROOT, "reports", "figures"))

    # Compute predicted churn probabilities on full dataset for segmentation
    from src.features import engineer_features
    df_feat = engineer_features(df)
    X_full_proc = preprocessor.preprocessor_pipeline.transform(df_feat.drop(columns=["churn", "customer_id"], errors="ignore"))
    df["churn_prob"] = best_xgb.predict_proba(X_full_proc)[:, 1]

    segmenter = CustomerValueSegmentation(n_clusters=4, random_state=42)
    df_segmented = segmenter.fit_predict(df)
    segmenter.save(os.path.join(PROJECT_ROOT, "models", "segmentation_kmeans.pkl"))
    segmenter.export_segmentation_visual(df_segmented, output_path=os.path.join(PROJECT_ROOT, "reports", "figures", "08_segmentation_matrix.png"))

    roi_metrics = segmenter.calculate_business_roi(df_segmented)

    duration = time.time() - start_time
    print("\n" + "="*80)
    print(f"🎉 PIPELINE EXECUTION COMPLETED IN {duration:.1f} SECONDS!")
    print("="*80)
    print(f"🏆 Best Model: Tuned XGBoost")
    print(f"📊 Final ROC-AUC: {final_metrics['ROC-AUC']:.4f} | Recall: {final_metrics['Recall']:.4f} | F1-Score: {final_metrics['F1-Score']:.4f}")
    print(f"🎯 Optimal Business Threshold: {optimal_th:.2f}")
    print(f"💰 Projected Annual Revenue Saved: ${roi_metrics['annual_revenue_saved']:,.2f} (ROI: {roi_metrics['roi_percentage']:.1f}%)")
    print(f"📁 Artifacts exported to /models and reports to /reports/figures")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
