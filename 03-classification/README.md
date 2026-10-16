# Machine Learning Zoomcamp - Week 3 Homework: Classification

## Dataset
Dataset URL: `https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv`

Target variable: `converted` (has the client signed up to the platform or not).

## Data preparation
* Check if missing values are present in the features.
* For categorical features, replace them with `'NA'`.
* For numerical features, replace them with `0.0`.

## Questions & Options
### Question 1: Mode for industry
- `NA`
- `technology`
- `healthcare`
- `retail`

### Question 2: Biggest correlation
- `interaction_count` and `lead_score`
- `number_of_courses_viewed` and `lead_score`
- `number_of_courses_viewed` and `interaction_count`
- `annual_income` and `interaction_count`

### Data Split
- Split the data with:
```python
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=42)
```
- Ensure `converted` is not in the features dataframe.

### Question 3: Biggest Mutual Information (MI)
- `industry`
- `location`
- `lead_source`
- `employment_status`

### Question 4: Accuracy
Model: `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`
- `0.55`
- `0.65`
- `0.75`
- `0.85`

### Question 5: Feature selection (least useful feature by smallest difference)
- `lead_source`
- `number_of_courses_viewed`
- `interaction_count`

### Question 6: Parameter tuning (best C)
Values: `[0.000001, 0.00001, 0.0001, 0.001]`
- `0.000001`
- `0.00001`
- `0.0001`
- `0.001`
