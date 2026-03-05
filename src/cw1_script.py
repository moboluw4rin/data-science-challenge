"""
CW1 Data Science Challenge 

What this script does:
1) Loads CW1_train.csv and CW1_test.csv
2) Ordinal-encodes ordered categorical variables: cut, color, clarity
3) Builds preprocessing pipelines
4) Compares 5 models via cross-validated R²:
   - Linear Regression
   - Ridge
   - Random Forest
   - XGBoost 
   - Support Vector Regression (SVR)
5) Picks the best model by mean CV R², refits on all training data
6) Predicts on test data and saves: CW1_test_predictions.csv

Note:
- You may need to adjust train_data_path & test_data_path, it is currently set to the location of the csv files on my personal laptop
- Run: python cw1_script.py
"""
#imports
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


#config
train_data_path = r"C:\Users\DELL\Downloads\CW1_train.csv"
test_data_path = r"C:\Users\DELL\Downloads\CW1_test.csv"

RANDOM_STATE = 42
N_SPLITS = 5

#output file
predictions_path = "CW1_test_predictions.csv"


#ordinal mappings (based on domain information about diamonds)
ORDINAL_MAPS = {
    "cut": {
        # worst -> best
        "Fair": 1,
        "Good": 2,
        "Very Good": 3,
        "Premium": 4,
        "Ideal": 5,
    },
    "color": {
        # worst -> best
        "J": 1, "I": 2, "H": 3, "G": 4, "F": 5, "E": 6, "D": 7
    },
    "clarity": {
        # worst -> best
        "I1": 1, "SI2": 2, "SI1": 3, "VS2": 4, "VS1": 5, "VVS2": 6, "VVS1": 7, "IF": 8
    }
}

#this function applies the ordinal encodings 
#used this instead of one-hot encoding to preserve ranking information & reduce unnecessary dimensionality expansion

def apply_ordinal_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies deterministic ordinal mapping to cut/color/clarity (if columns exist).
    """
    df = df.copy()
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

# this function handles all pre-processing
def build_preprocessors(X: pd.DataFrame):
    """
    Builds two preprocessors:
    - tree_preprocess: numeric impute + categorical impute+ OHE (no scaling)
    - linear_preprocess: same but adds scaling for numeric columns
    After ordinal encoding, cut/color/clarity become numeric and will be in numerical_columns.
    """
    categorical_columns = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_columns = [c for c in X.columns if c not in categorical_columns]

    tree_preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numerical_columns),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_columns),
        ],
        remainder="drop"
    )

    linear_preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numerical_columns),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_columns),
        ],
        remainder="drop"
    )

    return tree_preprocess, linear_preprocess

# this function computes both mean & standard deviation of cross-validated r2
def mean_std_r2(estimator, X, y, cv):
    
    scores = cross_val_score(estimator, X, y, scoring="r2", cv=cv, n_jobs=-1)
    return float(scores.mean()), float(scores.std())

# main function
def main():

    # Load data
    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    if "outcome" not in train.columns:
        raise ValueError("Training file must contain 'outcome' column as the target.")

    # Apply ordinal encoding (cut/color/clarity -> ordered integers)
    train = apply_ordinal_encoding(train)
    test = apply_ordinal_encoding(test)

    y = train["outcome"]
    X = train.drop(columns=["outcome"])

    # Test may or may not contain outcome; drop if present
    X_test = test.drop(columns=["outcome"], errors="ignore")

  
    # Build preprocessors and CV
    
    tree_preprocess, linear_preprocess = build_preprocessors(X)
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    results = []  # list of dicts: model name, best_cv_mean, best_cv_std, fitted_search_or_model

    
    # 1) Linear Regression (baseline)
    
    lin = Pipeline([("prep", linear_preprocess), ("model", LinearRegression())])
    
    lin_mean, lin_std = mean_std_r2(lin, X, y, cv)
    results.append({
        "name": "LinearRegression",
        "cv_mean_r2": lin_mean,
        "cv_std_r2": lin_std,
        "estimator": lin
    })
    print(f"[LinearRegression] CV R²: {lin_mean:.6f} ± {lin_std:.6f}")

    
    # 2) Ridge (tuned)
    
    ridge = Pipeline([("prep", linear_preprocess), ("model", Ridge(random_state=RANDOM_STATE))])

    ridge_grid = {"model__alpha": np.logspace(-4, 4, 30)}

    ridge_search = GridSearchCV(
        ridge,
        ridge_grid,
        scoring="r2",
        cv=cv,
        n_jobs=-1
    )
    ridge_search.fit(X, y)

    results.append({
        "name": "Ridge",
        "cv_mean_r2": float(ridge_search.best_score_),
        "cv_std_r2": float(np.std(ridge_search.cv_results_["mean_test_score"])),  # rough indicator
        "estimator": ridge_search.best_estimator_
    })
    print(f"[Ridge] Best CV R²: {ridge_search.best_score_:.6f} | Best params: {ridge_search.best_params_}")

    
    # 3) Random Forest (tuned)
    
    rf = Pipeline([
        ("prep", tree_preprocess),
        ("model", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))
    ])

    rf_params = {
        "model__n_estimators": [400, 800, 1200],
        "model__max_depth": [None, 10, 20, 40],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": ["sqrt", "log2", 0.5, 0.8],
    }

    rf_search = RandomizedSearchCV(
        rf,
        rf_params,
        n_iter=30,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    rf_search.fit(X, y)

    results.append({
        "name": "RandomForest",
        "cv_mean_r2": float(rf_search.best_score_),
        "cv_std_r2": float(np.std(rf_search.cv_results_["mean_test_score"])),  # rough indicator
        "estimator": rf_search.best_estimator_
    })
    print(f"[RandomForest] Best CV R²: {rf_search.best_score_:.6f} | Best params: {rf_search.best_params_}")

    
    # 4) XGBoost (tuned)
    xgb_available = True
    try:
        from xgboost import XGBRegressor
    except Exception:
        xgb_available = False
        print("[XGBoost] xgboost is not installed; skipping XGBoost model.")

    if xgb_available:
        xgb = Pipeline([
            ("prep", tree_preprocess),
            ("model", XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_estimators=2000,
                n_jobs=-1
            ))
        ])

        #xgboost hyperparameters
        xgb_params = {
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 6, 8],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_weight": [1, 5, 10, 20],
            "model__reg_alpha": [0, 0.1, 1, 5],
            "model__reg_lambda": [1, 2, 5, 10, 20],
            "model__gamma": [0, 0.5, 1, 5],
        }

        xgb_search = RandomizedSearchCV(
            xgb,
            xgb_params,
            n_iter=50,
            scoring="r2",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE
        )
        xgb_search.fit(X, y)

        results.append({
            "name": "XGBoost",
            "cv_mean_r2": float(xgb_search.best_score_),
            "cv_std_r2": float(np.std(xgb_search.cv_results_["mean_test_score"])),  # rough indicator
            "estimator": xgb_search.best_estimator_
        })
        print(f"[XGBoost] Best CV R²: {xgb_search.best_score_:.6f} | Best params: {xgb_search.best_params_}")

    
    # 5) Support Vector Regression (tuned)

    svr = Pipeline([("prep", linear_preprocess), ("model", SVR(kernel="rbf"))])
    svr_params = {
        "model__C": np.logspace(-1, 3, 20),
        "model__gamma": np.logspace(-4, 0, 20),
        "model__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5],
    }

    svr_search = RandomizedSearchCV(
        svr,
        svr_params,
        n_iter=40,
        scoring="r2",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    svr_search.fit(X, y)

    results.append({
        "name": "SVR",
        "cv_mean_r2": float(svr_search.best_score_),
        "cv_std_r2": float(np.std(svr_search.cv_results_["mean_test_score"])),  # rough indicator
        "estimator": svr_search.best_estimator_
    })
    print(f"[SVR] Best CV R²: {svr_search.best_score_:.6f} | Best params: {svr_search.best_params_}")


    # selects best model by mean CV R²

    best = max(results, key=lambda d: d["cv_mean_r2"])

    print("\n====================")
    print("Model comparison (mean CV R²):")

    for r in sorted(results, key=lambda d: d["cv_mean_r2"], reverse=True):
        print(f"  - {r['name']}: {r['cv_mean_r2']:.6f}")
    print("====================")
    print(f"Chosen best model: {best['name']} (CV R² = {best['cv_mean_r2']:.6f})\n")


    # fits best model on full training data and predict test

    best_estimator = best["estimator"]
    best_estimator.fit(X, y)

    # computes training R²
    train_r2 = best_estimator.score(X, y)
    print("Training R²:", train_r2)

    preds = best_estimator.predict(X_test)

    # saves the predictions
    pd.DataFrame({"prediction": preds}).to_csv(predictions_path, index=False)
    print(f"Saved predictions to: {predictions_path}")


if __name__ == "__main__":
    main()