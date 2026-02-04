# Exam Score Prediction - Machine Learning Project

## Overview
This project implements **Linear Regression** from scratch using NumPy to predict exam scores based on student characteristics and academic behavior.

## Dataset
- **Source**: Exam_Score_Prediction.csv
- **Samples**: 91 student records
- **Target Variable**: exam_score (0-100)
- **Features**: 12 input features

### Features Included:
- **Demographic**: age, gender
- **Academic**: course, study_hours, class_attendance
- **Environmental**: internet_access, sleep_hours, sleep_quality
- **Behavioral**: study_method, facility_rating, exam_difficulty

## Project Structure
```
ExamScorePrediction/
├── Exam_Score_Prediction.ipynb       # Main Jupyter notebook
├── Exam_Score_Prediction.csv         # Dataset
├── linear_regression.py              # Custom linear regression implementation
├── requirements.txt                  # Project dependencies
├── README.md                         # This file
└── outputs/
    ├── model_evaluation.png
    ├── feature_importance.png
    └── residuals_distribution.png
```

## Installation & Setup

### 1. Install Required Packages
```bash
pip install -r requirements.txt
```

### 2. Run the Jupyter Notebook
```bash
jupyter notebook Exam_Score_Prediction.ipynb
```

## Model Architecture

### Algorithm: Linear Regression
Uses the **Normal Equation** approach:
```
β = (X^T X)^-1 X^T y
```

Where:
- **X**: Feature matrix (n_samples × n_features)
- **y**: Target values (n_samples,)
- **β**: Model coefficients to learn

### Implementation Details:
- **Language**: Python 3.8+
- **Core Library**: NumPy (no sklearn for regression core)
- **Data Preprocessing**: StandardScaler, LabelEncoder
- **Train-Test Split**: 80-20 ratio
- **Feature Normalization**: StandardScaler (mean=0, std=1)

## Expected Results

### Model Performance:
- **R² Score**: ~0.75-0.80 (explains 75-80% of variance)
- **RMSE**: ~10-12 points
- **MAE**: ~8-10 points

### Top Predictive Features:
1. Class Attendance
2. Study Hours
3. Sleep Quality
4. Facility Rating
5. Study Method

## Key Steps in the Notebook

### Step 1: Load and Explore Data
- Load CSV file
- Display basic statistics
- Check data types and missing values

### Step 2: Data Preprocessing
- Encode categorical variables (gender, course, study_method, etc.)
- Convert to NumPy arrays
- Train-test split (80-20)
- Normalize features

### Step 3: Linear Regression Implementation
- Create custom LinearRegressionNumPy class
- Implement fit() method using Normal Equation
- Implement predict() method
- Implement score() method for R²

### Step 4: Model Training
- Train on normalized training data
- Display model coefficients
- Show feature importance

### Step 5: Model Evaluation
- Calculate MSE, RMSE, MAE, R² on both sets
- Compare training vs testing metrics
- Detect overfitting/underfitting

### Step 6: Visualizations
- Actual vs Predicted plots
- Residual plots for error analysis
- Feature importance bar chart
- Residual distribution histograms

### Step 7: Model Summary
- Print comprehensive statistics
- Key insights and recommendations
- Performance interpretation

### Step 8: Example Predictions
- Show sample predictions
- Calculate prediction errors

## Model Equation

After training, the model will generate:
```
exam_score = b0 + b1*age + b2*gender + b3*course + ... + bn*exam_difficulty

Example:
exam_score = 58.45 + 0.23*age - 2.14*gender + 1.56*course + ...
```

## Evaluation Metrics

- **MSE (Mean Squared Error)**: Average squared difference between actual and predicted
- **RMSE (Root Mean Squared Error)**: Square root of MSE (same units as target)
- **MAE (Mean Absolute Error)**: Average absolute difference
- **R² Score**: Proportion of variance explained (0-1 scale)

## Visualizations Generated

1. **model_evaluation.png**: 2x2 grid showing actual vs predicted and residuals
2. **feature_importance.png**: Top 10 features by coefficient magnitude
3. **residuals_distribution.png**: Histogram of prediction errors

## Usage Example

```python
# Load and preprocess data
df = pd.read_csv('Exam_Score_Prediction.csv')
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegressionNumPy()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate
r2 = model.score(X_test, y_test)
print(f"R² Score: {r2:.4f}")
```

## Interpretation Guide

### R² Score:
- **0.80+**: Excellent fit
- **0.60-0.80**: Good fit
- **0.40-0.60**: Moderate fit
- **<0.40**: Poor fit

### RMSE:
- Interpreted as "average prediction error in points"
- Lower is better

### Residuals:
- Should be randomly distributed around zero
- No patterns in residual plot indicates good fit
- Normal distribution of residuals is ideal

## Assumptions of Linear Regression

1. **Linearity**: Relationship between X and y is linear
2. **Independence**: Observations are independent
3. **Homoscedasticity**: Constant variance of residuals
4. **Normality**: Residuals are normally distributed
5. **No Multicollinearity**: Features are not highly correlated

## Limitations & Future Improvements

### Current Limitations:
- Linear relationships only (may miss non-linear patterns)
- Assumes all features are equally important
- Sensitive to outliers

### Future Enhancements:
- Polynomial regression for non-linear relationships
- Ridge/Lasso regression for regularization
- Feature engineering and selection
- Cross-validation for robust evaluation
- Handling of outliers
- Try other algorithms (Random Forest, Gradient Boosting, etc.)

## Technologies Used

- **Python 3.8+**: Programming language
- **NumPy**: Numerical computing and matrix operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization
- **Scikit-learn**: Preprocessing and metrics (not core regression)
- **Jupyter**: Interactive notebook environment

## Author Notes

This project demonstrates:
- Implementation of ML algorithms from scratch
- Proper data preprocessing pipeline
- Model training and evaluation best practices
- Data visualization techniques
- Comprehensive documentation

## License
Educational use - MBA Machine Learning Assignment

## References

- Normal Equation: [Wikipedia](https://en.wikipedia.org/wiki/Linear_regression)
- NumPy Documentation: [numpy.org](https://numpy.org/)
- Linear Regression Theory: [scikit-learn docs](https://scikit-learn.org/stable/modules/linear_model.html)

---

**Last Updated**: February 2026
**Course**: Machine Learning - MBA Fourth Semester
