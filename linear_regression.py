"""
Linear Regression Implementation using NumPy

This module provides a custom implementation of Linear Regression
using the Normal Equation method: β = (X^T X)^-1 X^T y

Author: Machine Learning Project
Date: February 2026
"""

import numpy as np
from typing import Tuple, Optional


class LinearRegressionNumPy:
    """
    Linear Regression implementation using NumPy only.
    
    Uses the Normal Equation approach to calculate optimal coefficients:
    β = (X^T X)^-1 X^T y
    
    Attributes:
        coefficients (np.ndarray): Feature weights/coefficients
        intercept (float): Bias term (y-intercept)
        X_train (np.ndarray): Training features (stored for reference)
        y_train (np.ndarray): Training target values (stored for reference)
    """
    
    def __init__(self):
        """Initialize the Linear Regression model."""
        self.coefficients = None
        self.intercept = None
        self.X_train = None
        self.y_train = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionNumPy':
        """
        Fit the linear regression model using the Normal Equation.
        
        The Normal Equation provides a closed-form solution for linear regression:
        β = (X^T X)^-1 X^T y
        
        This is computationally efficient and avoids the need for gradient descent.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features of shape (n_samples, n_features)
        y : np.ndarray
            Training target values of shape (n_samples,)
        
        Returns:
        --------
        self : LinearRegressionNumPy
            Returns self to allow method chaining
        
        Raises:
        -------
        ValueError
            If X and y have incompatible shapes
        numpy.linalg.LinAlgError
            If (X^T X) is singular (non-invertible)
        """
        # Validate inputs
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y must have same number of samples. "
                           f"Got X: {X.shape[0]}, y: {y.shape[0]}")
        
        # Add bias term (column of ones) to X
        # This allows the Normal Equation to directly solve for intercept
        X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
        
        # Calculate X^T X (transpose of X multiplied by X)
        XTX = X_with_bias.T @ X_with_bias
        
        # Calculate X^T y (transpose of X multiplied by y)
        XTy = X_with_bias.T @ y
        
        # Solve for coefficients: β = (X^T X)^-1 X^T y
        # Using np.linalg.solve is more numerically stable than matrix inversion
        try:
            coefficients = np.linalg.solve(XTX, XTy)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                "Matrix (X^T X) is singular. "
                "This may be due to multicollinearity or insufficient samples."
            )
        
        # Store intercept (first coefficient) and feature coefficients
        self.intercept = coefficients[0]
        self.coefficients = coefficients[1:]
        
        # Store training data for reference
        self.X_train = X
        self.y_train = y
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Uses the linear regression equation:
        y_pred = b0 + b1*x1 + b2*x2 + ... + bn*xn
        
        Parameters:
        -----------
        X : np.ndarray
            Features for prediction of shape (n_samples, n_features)
        
        Returns:
        --------
        np.ndarray
            Predicted values of shape (n_samples,)
        
        Raises:
        -------
        ValueError
            If model has not been fitted yet
        """
        if self.coefficients is None:
            raise ValueError("Model has not been fitted yet! "
                           "Call fit() method first.")
        
        if X.shape[1] != len(self.coefficients):
            raise ValueError(f"Expected {len(self.coefficients)} features, "
                           f"got {X.shape[1]}")
        
        # Linear regression equation: y = b0 + b1*x1 + b2*x2 + ... + bn*xn
        predictions = self.intercept + X @ self.coefficients
        
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² (coefficient of determination) score.
        
        R² measures the proportion of variance in y explained by the model.
        Formula: R² = 1 - (SS_res / SS_tot)
        
        Where:
        - SS_res = Σ(y_actual - y_pred)² (sum of squared residuals)
        - SS_tot = Σ(y_actual - y_mean)² (total sum of squares)
        
        Parameters:
        -----------
        X : np.ndarray
            Features of shape (n_samples, n_features)
        y : np.ndarray
            Actual target values of shape (n_samples,)
        
        Returns:
        --------
        float
            R² score between 0 and 1 (higher is better)
            - 1.0: Perfect fit
            - 0.5: 50% variance explained
            - 0.0: Model explains no variance
            - <0.0: Model performs worse than horizontal line
        """
        y_pred = self.predict(X)
        
        # Sum of squared residuals
        ss_res = np.sum((y - y_pred) ** 2)
        
        # Total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        # Calculate R² score
        if ss_tot == 0:
            return 1.0 if np.allclose(y_pred, y) else 0.0
        
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def get_params(self) -> dict:
        """
        Get model parameters.
        
        Returns:
        --------
        dict
            Dictionary containing 'intercept' and 'coefficients'
        """
        return {
            'intercept': self.intercept,
            'coefficients': self.coefficients.copy() if self.coefficients is not None else None
        }
    
    def set_params(self, **params):
        """
        Set model parameters.
        
        Parameters:
        -----------
        intercept : float
            Bias term
        coefficients : np.ndarray
            Feature coefficients
        """
        if 'intercept' in params:
            self.intercept = params['intercept']
        if 'coefficients' in params:
            self.coefficients = params['coefficients']
        return self
    
    def __repr__(self) -> str:
        """String representation of the model."""
        if self.coefficients is None:
            return "LinearRegressionNumPy(unfitted)"
        return (f"LinearRegressionNumPy(intercept={self.intercept:.4f}, "
                f"n_features={len(self.coefficients)})")


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Squared Error.
    
    MSE = (1/n) * Σ(y_true - y_pred)²
    
    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    
    Returns:
    --------
    float
        Mean squared error
    """
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.
    
    RMSE = √(MSE) = √[(1/n) * Σ(y_true - y_pred)²]
    
    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    
    Returns:
    --------
    float
        Root mean squared error
    """
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.
    
    MAE = (1/n) * Σ|y_true - y_pred|
    
    Parameters:
    -----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    
    Returns:
    --------
    float
        Mean absolute error
    """
    return np.mean(np.abs(y_true - y_pred))


if __name__ == "__main__":
    # Example usage
    print("Linear Regression Module")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(100, 3)
    true_coef = np.array([2.5, -1.3, 0.8])
    y = 3.0 + X @ true_coef + np.random.randn(100) * 0.1
    
    # Split data
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    model = LinearRegressionNumPy()
    model.fit(X_train, y_train)
    
    # Evaluate
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    print(f"\nModel: {model}")
    print(f"Train R² Score: {train_r2:.4f}")
    print(f"Test R² Score: {test_r2:.4f}")
    print(f"\nTrue coefficients: {true_coef}")
    print(f"Learned coefficients: {model.coefficients}")
