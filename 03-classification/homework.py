"""Machine Learning Zoomcamp 2026 - Homework 3: Classification

This script provides complete, reproducible solutions for Homework 3 of ML Zoomcamp:
  - Dataset: 2026 Lead Scoring dataset
  - Categorical Features: lead_source, industry, employment_status, location
  - Numerical Features: annual_income, number_of_courses_viewed, interaction_count, lead_score
  - Target: converted (binary classification: 0 or 1)
"""

from typing import Dict, List, Tuple
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mutual_info_score
from sklearn.model_selection import train_test_split

# =============================================================================
# Configuration & Constants
# =============================================================================
DATA_URL = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv"
LOCAL_FILE = os.path.join(os.path.dirname(__file__), "course_lead_scoring_2026.csv")

CATEGORICAL_FEATURES: List[str] = [
    "lead_source",
    "industry",
    "employment_status",
    "location",
]

NUMERICAL_FEATURES: List[str] = [
    "annual_income",
    "number_of_courses_viewed",
    "interaction_count",
    "lead_score",
]

ALL_FEATURES: List[str] = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET: str = "converted"


# =============================================================================
# Core Utility & Data Preparation Functions
# =============================================================================
def load_data(url: str = DATA_URL, local_path: str = LOCAL_FILE) -> pd.DataFrame:
    """Load dataset from local file if available, otherwise fetch from URL."""
    if os.path.exists(local_path):
        print(f"Loading dataset from local cache: {local_path}...")
        df = pd.read_csv(local_path)
    else:
        print(f"Fetching dataset from {url}...")
        df = pd.read_csv(url)
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data according to instructions:
    - Check missing values
    - Categorical features: fill missing values with 'NA'
    - Numerical features: fill missing values with 0.0
    """
    df = df.copy()

    print("Missing values before imputation:")
    missing = df[ALL_FEATURES].isnull().sum()
    for col, count in missing.items():
        if count > 0:
            print(f"  - {col}: {count} missing")

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("NA")

    for col in NUMERICAL_FEATURES:
        df[col] = df[col].fillna(0.0)

    return df


def split_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into 60% train, 20% validation, and 20% test using exact specifications:
    - df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    - df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
    - Separate target 'converted' and ensure it is not present in feature dataframes.
    """
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train[TARGET].values
    y_val = df_val[TARGET].values
    y_test = df_test[TARGET].values

    del df_train[TARGET]
    del df_val[TARGET]
    del df_test[TARGET]

    return df_train, df_val, df_test, y_train, y_val, y_test


# =============================================================================
# Homework Question Solvers
# =============================================================================
def solve_q1(df: pd.DataFrame) -> str:
    """Question 1: What is the most frequent observation (mode) for the column `industry`?"""
    counts = df["industry"].value_counts()
    print("Value counts for 'industry':")
    for val, count in counts.items():
        print(f"  - {val}: {count}")

    mode_val = df["industry"].mode()[0]
    print(f"\n--> Question 1 Answer: {mode_val}\n")
    return mode_val


def solve_q2(df: pd.DataFrame) -> str:
    """Question 2: Which two numerical features have the biggest correlation?"""
    corr = df[NUMERICAL_FEATURES].corr()
    print("Correlation matrix:")
    print(corr.round(4))

    candidate_pairs = [
        ("interaction_count", "lead_score"),
        ("number_of_courses_viewed", "lead_score"),
        ("number_of_courses_viewed", "interaction_count"),
        ("annual_income", "interaction_count"),
    ]

    print("\nCandidate feature correlations:")
    corr_results = {}
    for f1, f2 in candidate_pairs:
        val = corr.loc[f1, f2]
        corr_results[f"{f1} and {f2}"] = val
        print(f"  - {f1} and {f2}: {val:.4f}")

    best_pair = max(corr_results, key=corr_results.get)
    print(f"\n--> Question 2 Answer: {best_pair} (correlation = {corr_results[best_pair]:.4f})\n")
    return best_pair


def solve_q3(df_train: pd.DataFrame, y_train: np.ndarray) -> str:
    """Question 3: Which categorical variable has the biggest mutual information score with converted?"""
    print("Mutual Information scores (rounded to 2 decimals):")
    mi_scores: Dict[str, float] = {}

    for col in CATEGORICAL_FEATURES:
        score = mutual_info_score(df_train[col], y_train)
        rounded_score = round(score, 2)
        mi_scores[col] = rounded_score
        print(f"  - {col}: raw = {score:.4f}, rounded = {rounded_score:.2f}")

    biggest_mi = max(mi_scores, key=mi_scores.get)
    print(f"\n--> Question 3 Answer: {biggest_mi} (score = {mi_scores[biggest_mi]})\n")
    return biggest_mi


def solve_q4(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_val: pd.DataFrame,
    y_val: np.ndarray,
) -> Tuple[float, float]:
    """Question 4: Logistic regression validation accuracy."""
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train[ALL_FEATURES].to_dict(orient="records"))
    X_val = dv.transform(df_val[ALL_FEATURES].to_dict(orient="records"))

    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    raw_acc = accuracy_score(y_val, y_pred)
    rounded_acc = round(raw_acc, 2)

    print(f"Validation Accuracy: raw = {raw_acc:.4f}, rounded (2 decimals) = {rounded_acc}")
    print(f"--> Question 4 Answer: {rounded_acc}\n")
    return raw_acc, rounded_acc


def solve_q5(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_val: pd.DataFrame,
    y_val: np.ndarray,
    orig_acc: float,
) -> str:
    """Question 5: Feature elimination - which feature has the smallest difference?"""
    candidate_features = [
        "lead_source",
        "number_of_courses_viewed",
        "interaction_count",
    ]

    print(f"Original accuracy without rounding: {orig_acc:.4f}")
    print("Testing feature elimination:")

    diffs: Dict[str, float] = {}
    for feature in candidate_features:
        subset = [f for f in ALL_FEATURES if f != feature]
        dv = DictVectorizer(sparse=False)
        X_tr = dv.fit_transform(df_train[subset].to_dict(orient="records"))
        X_va = dv.transform(df_val[subset].to_dict(orient="records"))

        model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
        model.fit(X_tr, y_train)

        acc = accuracy_score(y_val, model.predict(X_va))
        diff = orig_acc - acc
        diffs[feature] = diff
        print(f"  - Without '{feature}': acc = {acc:.4f}, difference (orig - without) = {diff:+.6f}")

    smallest_feature = min(diffs, key=diffs.get)
    print(f"\n--> Question 5 Answer: {smallest_feature} (diff = {diffs[smallest_feature]:+.6f})\n")
    return smallest_feature


def solve_q6(
    df_train: pd.DataFrame,
    y_train: np.ndarray,
    df_val: pd.DataFrame,
    y_val: np.ndarray,
) -> float:
    """Question 6: Parameter tuning for C - which C leads to the best validation accuracy?"""
    c_candidates = [0.000001, 0.00001, 0.0001, 0.001]

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(df_train[ALL_FEATURES].to_dict(orient="records"))
    X_val = dv.transform(df_val[ALL_FEATURES].to_dict(orient="records"))

    print("Evaluating regularization parameter C:")
    scores: Dict[float, float] = {}
    for c in c_candidates:
        model = LogisticRegression(solver="liblinear", C=c, max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        raw_acc = accuracy_score(y_val, model.predict(X_val))
        rounded_acc = round(raw_acc, 3)
        scores[c] = rounded_acc
        print(f"  - C = {c:<8}: raw_acc = {raw_acc:.5f}, rounded (3 decimals) = {rounded_acc:.3f}")

    best_score = max(scores.values())
    best_c_list = [c for c, score in scores.items() if score == best_score]
    best_c = min(best_c_list)  # As per instructions: if multiple, choose smallest

    print(f"\n--> Question 6 Answer: {best_c} (best accuracy = {best_score:.3f})\n")
    return best_c


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    print("=" * 68)
    print("ML Zoomcamp 2026 - Homework 3: Classification Solution Pipeline")
    print("=" * 68 + "\n")

    raw_df = load_data()
    print(f"Dataset successfully loaded. Shape: {raw_df.shape}\n")

    print("-" * 68)
    print("Data Preparation (Handling Missing Values)")
    print("-" * 68)
    df = prepare_data(raw_df)
    print("Data preparation complete.\n")

    print("-" * 68)
    print("Running Questions...")
    print("-" * 68 + "\n")

    # Q1 & Q2
    q1_ans = solve_q1(df)
    q2_ans = solve_q2(df)

    # Data split for Q3-Q6
    df_train, df_val, df_test, y_train, y_val, y_test = split_data(df)
    print(f"Dataset split completed:")
    print(f"  - Train: {len(df_train)} samples")
    print(f"  - Val:   {len(df_val)} samples")
    print(f"  - Test:  {len(df_test)} samples\n")

    # Q3, Q4, Q5, Q6
    q3_ans = solve_q3(df_train, y_train)
    orig_acc, q4_ans = solve_q4(df_train, y_train, df_val, y_val)
    q5_ans = solve_q5(df_train, y_train, df_val, y_val, orig_acc)
    q6_ans = solve_q6(df_train, y_train, df_val, y_val)

    print("=" * 68)
    print("SUMMARY TABLE OF HOMEWORK 3 ANSWERS")
    print("=" * 68)
    print(f"  1. Mode for industry:       {q1_ans}")
    print(f"  2. Biggest correlation:     {q2_ans}")
    print(f"  3. Biggest MI:              {q3_ans}")
    print(f"  4. Accuracy:                {q4_ans}")
    print(f"  5. Feature selection:       {q5_ans}")
    print(f"  6. Parameter tuning:        {q6_ans}")
    print("=" * 68)


if __name__ == "__main__":
    main()
