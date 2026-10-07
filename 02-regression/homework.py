"""Machine Learning Zoomcamp 2026 - Homework 2: Linear Regression

This script provides complete, reproducible solutions for Homework 2 of ML Zoomcamp:
  - Dataset: 2026 Car Fuel Efficiency dataset
  - Features: engine_displacement, horsepower, vehicle_weight, model_year
  - Target: fuel_efficiency_mpg
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


# =============================================================================
# Configuration & Constants
# =============================================================================
DATA_URL = "https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv"

FEATURES: List[str] = [
    "engine_displacement",
    "horsepower",
    "vehicle_weight",
    "model_year",
]
TARGET: str = "fuel_efficiency_mpg"


# =============================================================================
# Core Utility & Modeling Functions
# =============================================================================
def load_data(url: str = DATA_URL) -> pd.DataFrame:
    """Load and filter dataset to required features and target."""
    print(f"Fetching dataset from {url}...")
    df = pd.read_csv(url)
    return df[FEATURES + [TARGET]].copy()


def split_data(
    df: pd.DataFrame, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset into 60% train, 20% validation, and 20% test using lecture shuffle."""
    n = len(df)
    n_val = int(n * 0.2)
    n_test = int(n * 0.2)
    n_train = n - n_val - n_test

    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)

    train = df.iloc[idx[:n_train]].copy().reset_index(drop=True)
    val = df.iloc[idx[n_train : n_train + n_val]].copy().reset_index(drop=True)
    test = df.iloc[idx[n_train + n_val :]].copy().reset_index(drop=True)

    return train, val, test


def train_linear_regression(
    X: np.ndarray, y: np.ndarray, r: float = 0.0
) -> Tuple[float, np.ndarray]:
    """Train linear regression with optional L2 regularization (Ridge).

    Bias term (w0) is not regularized.
    """
    ones = np.ones(X.shape[0])
    X_full = np.column_stack([ones, X])

    XTX = X_full.T.dot(X_full)
    if r > 0.0:
        reg = r * np.eye(XTX.shape[0])
        reg[0, 0] = 0.0  # Preserve bias term unpenalized
        XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    weights = XTX_inv.dot(X_full.T).dot(y)

    w0 = weights[0]
    w = weights[1:]
    return w0, w


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    error = y_pred - y_true
    mse = (error**2).mean()
    return float(np.sqrt(mse))


# =============================================================================
# Homework Question Solvers
# =============================================================================
def solve_q1(df: pd.DataFrame) -> str:
    """Question 1: Find column with missing values."""
    missing_counts = df[FEATURES].isnull().sum()
    print("Missing values per column:")
    for col, count in missing_counts.items():
        print(f"  - {col}: {count} missing")

    missing_cols = missing_counts[missing_counts > 0].index.tolist()
    answer = missing_cols[0]
    print(f"--> Question 1 Answer: {answer}\n")
    return answer


def solve_q2(df: pd.DataFrame) -> int:
    """Question 2: Compute median (50th percentile) for horsepower."""
    median_val = df["horsepower"].median()
    print(f"Horsepower median: {median_val}")
    answer = int(median_val)
    print(f"--> Question 2 Answer: {answer}\n")
    return answer


def solve_q3(df: pd.DataFrame) -> Tuple[str, float, float]:
    """Question 3: Compare filling missing values with 0 vs training set mean."""
    df_train, df_val, _ = split_data(df, seed=42)
    y_train = df_train[TARGET].values
    y_val = df_val[TARGET].values

    # Imputation Option 1: Fill with 0
    X_train_0 = df_train[FEATURES].fillna(0).values
    X_val_0 = df_val[FEATURES].fillna(0).values
    w0_0, w_0 = train_linear_regression(X_train_0, y_train, r=0.0)
    rmse_0 = round(rmse(y_val, w0_0 + X_val_0.dot(w_0)), 3)

    # Imputation Option 2: Fill with mean (from train set only)
    train_hp_mean = df_train["horsepower"].mean()
    X_train_mean = df_train[FEATURES].fillna(train_hp_mean).values
    X_val_mean = df_val[FEATURES].fillna(train_hp_mean).values
    w0_m, w_m = train_linear_regression(X_train_mean, y_train, r=0.0)
    rmse_mean = round(rmse(y_val, w0_m + X_val_mean.dot(w_m)), 3)

    print(f"Validation RMSE filling with 0:    {rmse_0}")
    print(f"Validation RMSE filling with mean: {rmse_mean}")

    if rmse_mean < rmse_0:
        answer = "With mean"
    elif rmse_0 < rmse_mean:
        answer = "With 0"
    else:
        answer = "Both are equally good"

    print(f"--> Question 3 Answer: {answer}\n")
    return answer, rmse_0, rmse_mean


def solve_q4(df: pd.DataFrame) -> float:
    """Question 4: Identify regularization parameter r that minimizes validation RMSE."""
    df_train, df_val, _ = split_data(df, seed=42)
    y_train = df_train[TARGET].values
    y_val = df_val[TARGET].values

    X_train = df_train[FEATURES].fillna(0).values
    X_val = df_val[FEATURES].fillna(0).values

    candidates = [0, 0.01, 0.1, 1, 5, 10, 100]
    scores = {}

    print("Evaluating regularization values:")
    for r in candidates:
        w0, w = train_linear_regression(X_train, y_train, r=r)
        y_pred = w0 + X_val.dot(w)
        score = round(rmse(y_val, y_pred), 4)
        scores[r] = score
        print(f"  r = {r:<6} -> Validation RMSE = {score:.4f}")

    # Tie-break rule: select smallest r if multiple give same rounded score
    best_r = min(candidates, key=lambda r: (scores[r], r))
    print(f"--> Question 4 Answer: {best_r}\n")
    return best_r


def solve_q5(df: pd.DataFrame) -> float:
    """Question 5: Measure model stability across 10 random seeds."""
    seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    scores = []

    print("Evaluating across split seeds [0-9]:")
    for s in seeds:
        tr, va, _ = split_data(df, seed=s)
        X_tr = tr[FEATURES].fillna(0).values
        y_tr = tr[TARGET].values
        X_va = va[FEATURES].fillna(0).values
        y_va = va[TARGET].values

        w0, w = train_linear_regression(X_tr, y_tr, r=0.0)
        y_pred = w0 + X_va.dot(w)
        score = rmse(y_va, y_pred)
        scores.append(score)
        print(f"  Seed {s}: Validation RMSE = {score:.4f}")

    std_val = round(float(np.std(scores)), 3)
    print(f"Standard deviation across seeds: {std_val}")
    print(f"--> Question 5 Answer: {std_val}\n")
    return std_val


def solve_q6(df: pd.DataFrame) -> float:
    """Question 6: Evaluate final model on test set using seed 9 and r=0.001."""
    tr, va, te = split_data(df, seed=9)
    full_train = pd.concat([tr, va]).reset_index(drop=True)

    X_full = full_train[FEATURES].fillna(0).values
    y_full = full_train[TARGET].values
    X_test = te[FEATURES].fillna(0).values
    y_test = te[TARGET].values

    w0, w = train_linear_regression(X_full, y_full, r=0.001)
    y_pred_test = w0 + X_test.dot(w)
    test_rmse = round(rmse(y_test, y_pred_test), 3)

    print(f"Final Test RMSE (seed=9, r=0.001): {test_rmse}")
    print(f"--> Question 6 Answer: {test_rmse}\n")
    return test_rmse


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    print("=" * 68)
    print("ML Zoomcamp 2026 - Homework 2: Linear Regression Solution Pipeline")
    print("=" * 68 + "\n")

    df = load_data()
    print(f"Dataset successfully loaded. Shape: {df.shape}\n")

    print("-" * 68)
    print("Running Questions...")
    print("-" * 68 + "\n")

    q1_ans = solve_q1(df)
    q2_ans = solve_q2(df)
    q3_ans, _, _ = solve_q3(df)
    q4_ans = solve_q4(df)
    q5_ans = solve_q5(df)
    q6_ans = solve_q6(df)

    print("=" * 68)
    print("SUMMARY TABLE OF HOMEWORK ANSWERS")
    print("=" * 68)
    print(f"  1. Missing values:           {q1_ans}")
    print(f"  2. Median for horsepower:    {q2_ans}")
    print(f"  3. Filling NAs:              {q3_ans}")
    print(f"  4. Best regularization:      {q4_ans}")
    print(f"  5. RMSE standard deviation:  {q5_ans}")
    print(f"  6. Evaluation on test:       {q6_ans}")
    print("=" * 68)


if __name__ == "__main__":
    main()
