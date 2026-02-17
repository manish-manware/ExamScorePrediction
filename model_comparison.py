import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def load_data(path):
    df = pd.read_csv(path)
    return df


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # create OneHotEncoder in a way that's compatible across scikit-learn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )
    return preprocessor


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return {"r2": r2, "mae": mae, "mse": mse, "rmse": rmse}


def main(csv_path="Exam_Score_Prediction.csv", output_csv="model_results.csv"):
    df = load_data(csv_path)

    # target
    target_col = "exam_score"

    # drop identifier
    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"]) 

    # define feature sets
    numeric_features = [c for c in df.select_dtypes(include=["number"]).columns if c != target_col]
    categorical_features = [c for c in df.columns if c not in numeric_features + [target_col]]

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "KNN": KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
    }

    results = []

    for name, estimator in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", estimator)])

        # evaluate on hold-out test
        metrics = evaluate_model(pipe, X_train, X_test, y_train, y_test)

        # cross-val R2 (5-fold)
        try:
            cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=-1)
            cv_r2_mean = np.mean(cv_scores)
            cv_r2_std = np.std(cv_scores)
        except Exception:
            cv_r2_mean = np.nan
            cv_r2_std = np.nan

        results.append({
            "model": name,
            "test_r2": metrics["r2"],
            "test_mae": metrics["mae"],
            "test_mse": metrics["mse"],
            "test_rmse": metrics["rmse"],
            "cv_r2_mean": cv_r2_mean,
            "cv_r2_std": cv_r2_std,
        })

    results_df = pd.DataFrame(results).sort_values(by="test_r2", ascending=False)
    print("Model comparison results:\n")
    print(results_df.to_string(index=False))

    results_df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    main()
