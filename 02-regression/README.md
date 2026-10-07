# Machine Learning Zoomcamp 2026 - Homework 2: Linear Regression

This directory contains the code and solutions for **Homework 2: Machine Learning for Regression**.

## Dataset
- **Name:** Car Fuel Efficiency dataset (2026 cohort release)
- **URL:** [car_fuel_efficiency_2026.csv](https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/car_fuel_efficiency_2026.csv)
- **Target:** `fuel_efficiency_mpg`
- **Features used:** `engine_displacement`, `horsepower`, `vehicle_weight`, `model_year`

---

## Questions & Verified Answers

| Question | Topic | Chosen Option | Details / Result |
|---|---|---|---|
| **Q1** | Missing values | **`horsepower`** | `horsepower` has 877 missing values; all other features have 0 |
| **Q2** | Median for horsepower | **`254`** | `df['horsepower'].median() = 254.0` |
| **Q3** | Filling NAs | **`With mean`** | RMSE filling with mean is `2.202` vs filling with 0 is `2.205` |
| **Q4** | Best regularization | **`0`** | Ties for lowest RMSE (`2.2053`), smallest $r$ is `0` |
| **Q5** | RMSE standard deviation | **`0.029`** | `np.std(seed_scores) = 0.02878...` rounded to 3 digits is `0.029` |
| **Q6** | Evaluation on test | **`2.236`** | Seed 9, train+val, $r=0.001$: Test RMSE is `2.23577...` rounded to `2.236` |

---

## Files in this Folder

1. `homework.ipynb` - Jupyter Notebook containing the full end-to-end analysis, markdown explanations, formulas, and executed outputs.
2. `homework.py` - Standalone executable Python script calculating all homework answers.
3. `README.md` - Problem descriptions and summary table.

## Running the Code

Run the Python script directly:
```bash
python homework.py
```
Or open and execute `homework.ipynb` in VS Code / Jupyter Lab.
