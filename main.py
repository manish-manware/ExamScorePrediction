"""
Main script for Exam Score Prediction project.

This script provides a command-line interface to train and evaluate
the linear regression model for predicting exam scores.

Usage:
    python main.py --train              # Train the model
    python main.py --predict            # Make predictions
    python main.py --evaluate           # Evaluate on test set
    python main.py --all                # Train and evaluate
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from linear_regression import LinearRegressionNumPy, mean_squared_error, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_and_preprocess_data(csv_path: str) -> tuple:
    """
    Load and preprocess the exam score prediction dataset.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test, feature_names, scaler)
    """
    print("\n" + "="*60)
    print("Loading and Preprocessing Data")
    print("="*60)
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"\n✓ Loaded data shape: {df.shape}")
    
    # Separate features and target
    X = df.drop('exam_score', axis=1)
    y = df['exam_score'].values
    
    feature_names = X.columns.tolist()
    
    # Encode categorical variables
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    print(f"✓ Encoded {len(categorical_cols)} categorical features")
    
    # Convert to numpy arrays
    X = X.values.astype(np.float32)
    y = y.astype(np.float32)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"✓ Features normalized (mean≈0, std≈1)")
    
    return X_train, X_test, y_train, y_test, feature_names, scaler


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegressionNumPy:
    """
    Train the linear regression model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training target values
    
    Returns:
    --------
    LinearRegressionNumPy
        Trained model
    """
    print("\n" + "="*60)
    print("Training Model")
    print("="*60)
    
    model = LinearRegressionNumPy()
    model.fit(X_train, y_train)
    
    print(f"\n✓ Model trained successfully!")
    print(f"  Intercept: {model.intercept:.4f}")
    print(f"  Number of features: {len(model.coefficients)}")
    
    return model


def evaluate_model(model: LinearRegressionNumPy, 
                   X_train: np.ndarray, y_train: np.ndarray,
                   X_test: np.ndarray, y_test: np.ndarray,
                   feature_names: list) -> dict:
    """
    Evaluate the model on train and test sets.
    
    Parameters:
    -----------
    model : LinearRegressionNumPy
        Trained model
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training targets
    X_test : np.ndarray
        Testing features
    y_test : np.ndarray
        Testing targets
    feature_names : list
        Names of features
    
    Returns:
    --------
    dict
        Evaluation metrics
    """
    print("\n" + "="*60)
    print("Model Evaluation")
    print("="*60)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = {
        'MSE': mean_squared_error(y_train, y_train_pred),
        'RMSE': root_mean_squared_error(y_train, y_train_pred),
        'MAE': mean_absolute_error(y_train, y_train_pred),
        'R2': model.score(X_train, y_train)
    }
    
    test_metrics = {
        'MSE': mean_squared_error(y_test, y_test_pred),
        'RMSE': root_mean_squared_error(y_test, y_test_pred),
        'MAE': mean_absolute_error(y_test, y_test_pred),
        'R2': model.score(X_test, y_test)
    }
    
    # Print results
    print("\nTraining Set Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nTesting Set Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Feature importance
    print("\n" + "-"*60)
    print("Top 5 Important Features:")
    print("-"*60)
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coefficients
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    for idx, row in coef_df.head(5).iterrows():
        print(f"  {row['Feature']:20s}: {row['Coefficient']:8.4f}")
    
    return {
        'train': train_metrics,
        'test': test_metrics,
        'y_test_actual': y_test,
        'y_test_pred': y_test_pred
    }


def print_summary(results: dict):
    """Print final summary of model performance."""
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    train_r2 = results['train']['R2']
    test_r2 = results['test']['R2']
    test_rmse = results['test']['RMSE']
    test_mae = results['test']['MAE']
    
    print(f"\n📊 Model Performance:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²:  {test_r2:.4f}")
    print(f"  Test RMSE:   {test_rmse:.4f} points")
    print(f"  Test MAE:    {test_mae:.4f} points")
    
    quality = "EXCELLENT" if test_r2 > 0.8 else "GOOD" if test_r2 > 0.6 else "MODERATE" if test_r2 > 0.4 else "POOR"
    print(f"\n  Model Quality: {quality}")
    
    overfitting = abs(train_r2 - test_r2) < 0.1
    print(f"  Overfitting: {'No' if overfitting else 'Possible'}")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("EXAM SCORE PREDICTION - LINEAR REGRESSION")
    print("="*60)
    
    # Set paths
    csv_path = project_root / 'Exam_Score_Prediction.csv'
    
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found!")
        return
    
    # Load and preprocess
    X_train, X_test, y_train, y_test, feature_names, scaler = load_and_preprocess_data(str(csv_path))
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate
    results = evaluate_model(model, X_train, y_train, X_test, y_test, feature_names)
    
    # Print summary
    print_summary(results)
    
    print("✓ Project execution completed successfully!")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
