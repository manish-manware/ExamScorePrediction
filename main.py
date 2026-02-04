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


def get_user_input(feature_names: list, scaler: StandardScaler, encoding_map: dict = None) -> np.ndarray:
    """
    Get user input for prediction.
    
    Parameters:
    -----------
    feature_names : list
        Names of features to input
    scaler : StandardScaler
        Scaler fitted on training data
    encoding_map : dict
        Mapping of categorical features to their encodings
    
    Returns:
    --------
    np.ndarray
        Scaled input features
    """
    print("\n" + "="*60)
    print("EXAM SCORE PREDICTION")
    print("="*60)
    
    # Feature details for user guidance
    feature_guidance = {
        'age': 'Age of student (e.g., 18-25)',
        'study_hours': 'Daily study hours (e.g., 2-8)',
        'class_attendance': 'Class attendance percentage (0-100)',
        'sleep_hours': 'Average sleep hours (4-10)',
        'facility_rating': 'Facility rating (low/medium/high)',
        'exam_difficulty': 'Exam difficulty (easy/moderate/hard)',
    }
    
    user_input = []
    
    for feature in feature_names:
        while True:
            try:
                if feature == 'gender':
                    print(f"\n{feature.upper()}:")
                    print("  Options: male, female, other")
                    value = input(f"Enter {feature}: ").strip().lower()
                    if value not in ['male', 'female', 'other']:
                        print("  ❌ Invalid input. Please enter: male, female, or other")
                        continue
                    # Encode gender
                    gender_map = {'female': 0, 'male': 1, 'other': 2}
                    user_input.append(float(gender_map[value]))
                
                elif feature == 'course':
                    print(f"\n{feature.upper()}:")
                    print("  Options: bca, diploma, b.sc, b.tech, b.com, b.a")
                    value = input(f"Enter {feature}: ").strip().lower()
                    if value not in ['bca', 'diploma', 'b.sc', 'b.tech', 'b.com', 'b.a']:
                        print("  ❌ Invalid input. Please enter a valid course")
                        continue
                    # Encode course
                    course_map = {'b.a': 0, 'b.com': 1, 'b.sc': 2, 'b.tech': 3, 'bca': 4, 'diploma': 5}
                    user_input.append(float(course_map[value]))
                
                elif feature == 'internet_access':
                    print(f"\n{feature.upper()}:")
                    print("  Options: yes, no")
                    value = input(f"Enter {feature}: ").strip().lower()
                    if value not in ['yes', 'no']:
                        print("  ❌ Invalid input. Please enter: yes or no")
                        continue
                    user_input.append(float(1 if value == 'yes' else 0))
                
                elif feature == 'sleep_quality':
                    print(f"\n{feature.upper()}:")
                    print("  Options: poor, average, good")
                    value = input(f"Enter {feature}: ").strip().lower()
                    if value not in ['poor', 'average', 'good']:
                        print("  ❌ Invalid input. Please enter: poor, average, or good")
                        continue
                    quality_map = {'average': 0, 'good': 1, 'poor': 2}
                    user_input.append(float(quality_map[value]))
                
                elif feature == 'study_method':
                    print(f"\n{feature.upper()}:")
                    print("  Options: coaching, online videos, self-study")
                    value = input(f"Enter {feature}: ").strip().lower()
                    if value not in ['coaching', 'online videos', 'self-study']:
                        print("  ❌ Invalid input. Please enter a valid study method")
                        continue
                    study_map = {'coaching': 0, 'online videos': 1, 'self-study': 2}
                    user_input.append(float(study_map[value]))
                
                elif feature == 'facility_rating':
                    print(f"\n{feature.upper()}:")
                    print("  Options: low, medium, high")
                    value = input(f"Enter {feature}: ").strip().lower()
                    if value not in ['low', 'medium', 'high']:
                        print("  ❌ Invalid input. Please enter: low, medium, or high")
                        continue
                    rating_map = {'high': 0, 'low': 1, 'medium': 2}
                    user_input.append(float(rating_map[value]))
                
                elif feature == 'exam_difficulty':
                    print(f"\n{feature.upper()}:")
                    print("  Options: easy, moderate, hard")
                    value = input(f"Enter {feature}: ").strip().lower()
                    if value not in ['easy', 'moderate', 'hard']:
                        print("  ❌ Invalid input. Please enter: easy, moderate, or hard")
                        continue
                    difficulty_map = {'easy': 0, 'hard': 1, 'moderate': 2}
                    user_input.append(float(difficulty_map[value]))
                
                else:
                    # Numeric features
                    guidance = feature_guidance.get(feature, '')
                    prompt = f"Enter {feature}" + (f" ({guidance})" if guidance else "") + ": "
                    value = float(input(prompt))
                    user_input.append(value)
                
                break
                
            except ValueError:
                print(f"  ❌ Invalid input. Please enter a valid value for {feature}")
            except Exception as e:
                print(f"  ❌ Error: {str(e)}")
    
    # Convert to numpy array and scale
    user_array = np.array(user_input).reshape(1, -1).astype(np.float32)
    user_scaled = scaler.transform(user_array)
    
    return user_scaled


def predict_score(model: LinearRegressionNumPy, user_input: np.ndarray) -> float:
    """
    Predict exam score for user input.
    
    Parameters:
    -----------
    model : LinearRegressionNumPy
        Trained model
    user_input : np.ndarray
        Scaled user input features
    
    Returns:
    --------
    float
        Predicted exam score
    """
    prediction = model.predict(user_input)[0]
    return max(0, min(100, prediction))  # Clip to valid score range


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
    
    # Interactive prediction loop
    while True:
        try:
            user_choice = input("\n🎯 Would you like to predict an exam score? (yes/no): ").strip().lower()
            
            if user_choice == 'yes':
                user_input = get_user_input(feature_names, scaler)
                predicted_score = predict_score(model, user_input)
                
                print("\n" + "="*60)
                print("PREDICTION RESULT")
                print("="*60)
                print(f"\n📈 Predicted Exam Score: {predicted_score:.2f}/100")
                
                if predicted_score >= 80:
                    print("   Status: 🌟 Excellent")
                elif predicted_score >= 60:
                    print("   Status: ✅ Good")
                elif predicted_score >= 40:
                    print("   Status: ⚠️  Average")
                else:
                    print("   Status: ❌ Poor")
                print("\n" + "="*60)
                
            elif user_choice == 'no':
                print("\n✓ Thank you for using the prediction system!")
                break
            else:
                print("❌ Please enter 'yes' or 'no'")
                
        except KeyboardInterrupt:
            print("\n\n✓ Program interrupted by user. Exiting...")
            break
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
    
    print("✓ Project execution completed successfully!")
    
    return model, results


if __name__ == "__main__":
    model, results = main()
