# Smartphone Addiction Prediction

Predicting smartphone addiction probability using behavioral and 
demographic features. Built for Kaggle Playground Series Season 6 
Episode 8.

## Competition
[Kaggle Playground Series S6E8 — Predicting Smartphone Addiction](https://www.kaggle.com/competitions/playground-series-s6e8)

## Problem Statement
Given behavioral patterns like daily screen time, social media usage, 
app categories, and demographic information, predict the probability 
of a user being addicted to their smartphone.

## Dataset
- **Source:** Kaggle Playground Series S6E8
- **Target variable:** Addicted (probability 0-1)
- **Task:** Binary classification

## What I Did

### 1. Exploratory Data Analysis
- Checked class distribution, feature distributions and correlations
- Identified missing values and data types
- Visualized key relationships between features and target

### 2. Feature Engineering
Created 13 new features including:
- **screen_time_bin** — binned daily screen time into 4 categories 
  (low/moderate/high/very high) using pd.cut
- **weekend_screen_bin** — binned weekend screen time usage
- **high_social_bin** — binned social media hours into risk categories
- Additional interaction and ratio features between related columns

### 3. Preprocessing Pipeline
Built a proper ColumnTransformer pipeline with separate sub-pipelines:
- **Numeric pipeline** — SimpleImputer (median strategy)
- **Ordinal pipeline** — SimpleImputer + OrdinalEncoder for 
  ordered categoricals
- **OHE pipeline** — SimpleImputer + OneHotEncoder for 
  nominal categoricals

### 4. Modeling
Trained and evaluated two gradient boosting models:
- XGBoost Classifier
- LightGBM Classifier

Used 5-fold cross validation with AUC and F1 scoring.

**Cross Validation Results:**

| Model | AUC | F1 |
|---|---|---|
| XGBoost | 0.961 | 0.928 |
| LightGBM | 0.956 | 0.924 |

### 5. Hyperparameter Tuning
Applied GridSearchCV on XGBoost pipeline with 5-fold CV.

Best parameters:
- learning_rate: 0.01
- max_depth: 4
- n_estimators: 100

### 6. Final Submission
Used predict_proba to generate addiction probabilities for submission.

**Kaggle Leaderboard Score: 0.95825**

## Key Findings
- Screen time and social media usage were the strongest predictors 
  of addiction
- Binning continuous features into categories improved model 
  performance
- XGBoost slightly outperformed LightGBM on this dataset
- ColumnTransformer pipeline prevented data leakage between 
  train and test sets

## What I'd Improve Next
- Apply Optuna hyperparameter tuning for smarter search
- Ensemble XGBoost and LightGBM predictions
- Engineer more interaction features between screen time 
  and social media columns
- Try CatBoost as a third model

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, 
Matplotlib, Seaborn

## Competition
[Kaggle Playground Series S6E8](https://www.kaggle.com/competitions/playground-series-s6e8)
