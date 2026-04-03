import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

from features import add_features_np, log1p_selected_np, num_cols


def build_pipeline() -> Pipeline:
    """Build and return the full preprocessing + model pipeline."""
    cat_cols = ["ocean_proximity"]

    num_pipeline = Pipeline(steps=[
        ("imputer",  SimpleImputer(strategy="median")),
        ("features", FunctionTransformer(add_features_np)),
        ("log1p",    FunctionTransformer(log1p_selected_np)),
    ])

    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols),
        ],
        remainder="drop",
    )

    model = HistGradientBoostingRegressor(random_state=42)

    return Pipeline(steps=[
        ("prep",  preprocess),
        ("model", model),
    ])


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return MAE, RMSE, R2 metrics."""
    return {
        "MAE":  mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2":   r2_score(y_true, y_pred),
    }


if __name__ == "__main__":
    DATA_PATH = os.getenv("DATA_PATH", "data/housing.csv")
    hous_df = pd.read_csv(DATA_PATH)

    # --- التحسين الثاني: حذف البيوت المـ capped ---
    # كل بيت سعره 500,001 هو في الحقيقة "أكثر من 500k" مجهول القيمة
    before = len(hous_df)
    hous_df = hous_df[hous_df["median_house_value"] < 500_001]
    after  = len(hous_df)
    print(f"Removed {before - after} capped rows | Remaining: {after}")

    X = hous_df.drop("median_house_value", axis=1)
    y = hous_df["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- التحسين الأول: log1p على الـ target ---
    # بيحوّل الأسعار لتوزيع أقرب لـ normal
    y_train_log = np.log1p(y_train)

    pipe = build_pipeline()

    # --- التحسين الثالث: GridSearchCV ---
    param_grid = {
        "model__max_iter":         [200, 500],
        "model__learning_rate":    [0.05, 0.1],
        "model__max_depth":        [4, 6, None],
        "model__min_samples_leaf": [20, 50],
    }

    search = GridSearchCV(
        estimator = pipe,
        param_grid = param_grid,
        cv         = 5,
        scoring    = "neg_mean_absolute_error",
        n_jobs     = -1,       # استخدم كل الـ CPU cores
        verbose    = 2,
    )

    print("\nStarting GridSearchCV — this may take a few minutes...")
    search.fit(X_train, y_train_log)

    print("\nBest parameters found:")
    for param, val in search.best_params_.items():
        print(f"  {param}: {val}")

    # التنبؤ + رجّع log1p بـ expm1
    best_pipe  = search.best_estimator_
    pred_log   = best_pipe.predict(X_test)
    pred       = np.expm1(pred_log) 

    metrics = evaluate(y_test, pred)
    print("\nTest Set Metrics:")
    for name, val in metrics.items():
        print(f"  {name}: {val:.4f}")

    os.makedirs("outputs", exist_ok=True)
    joblib.dump(best_pipe, "outputs/housing_model.joblib")
    print("\nSaved model -> outputs/housing_model.joblib")