# In a Colab environment, you can install the necessary libraries by running:
# !pip install pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def house_price_regression_pipeline():
    """
    A complete pipeline for a house price prediction task using an improved regression model.
    """
    print("--- Starting House Price Prediction Pipeline (with Improvements) ---")

    # 1. Import and load the dataset
    print("\n[Step 1] Loading and Preprocessing Data...")
    
    # The dataset will be automatically downloaded from this URL.
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    df = pd.DataFrame(data, columns=feature_names)
    df['PRICE'] = target

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
    print("Missing values count:", df.isnull().sum().sum())

    # Define features (X) and target (y)
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']

    # 2. Split data into train-test sets (Done before scaling to prevent data leakage)
    print("\n[Step 2] Splitting Data into Training and Testing Sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # IMPROVEMENT 1: Add Feature Scaling and Polynomial Features
    print("\n[Improvement] Applying Feature Scaling and Creating Polynomial Features...")
    
    # Create polynomial features (degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    # Scale the polynomial features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_poly)
    X_test_scaled = scaler.transform(X_test_poly)
    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Number of features after adding polynomial terms: {X_train_scaled.shape[1]}")

    # 3. Fit a Linear Regression model (with improved features)
    print("\n[Step 3] Fitting the Linear Regression Model...")
    model = LinearRegression() 
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    # Make predictions on the scaled test set
    y_pred = model.predict(X_test_scaled)

    # 4. Evaluate the model
    print("\n[Step 4] Evaluating the Improved Model...")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  - Mean Squared Error (MSE): {mse:.2f}")
    print(f"  - R-squared (R²): {r2:.2f}")
    print("-" * 50)
    print("**Interpretation of Metrics:**")
    print(f"The R² of {r2:.2f} means that our improved model explains approximately {r2:.0%} of the variance in house prices.")
    print(f"The MAE indicates that, on average, our model's predictions are now off by about ${mae:.2f}k.")

    # 5. Plot regression line and interpret results
    print("\n[Step 5] Visualizing Results...")
    
    # Scatter plot of Actual vs. Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Prices ($1000s)")
    plt.ylabel("Predicted Prices ($1000s)")
    plt.title("Actual vs. Predicted House Prices (Improved Model)", fontsize=16)
    plt.grid(True)
    plt.show()

    print("\n**Interpretation of the Plot:**")
    print("The points are now generally closer to the red dashed line, indicating that the model's predictions have improved and have less error compared to the basic version.")
    
    # Note on interpreting coefficients with polynomial features
    print("\n**Note on Coefficients:**")
    print("With polynomial features, the model has many more coefficients (one for each original feature, plus interaction and squared terms).")
    print("Directly interpreting a single coefficient is no longer straightforward, as its effect depends on other features. The focus shifts to the overall predictive accuracy of the model, which we've improved.")

    print("\n--- Pipeline Complete ---")

if __name__ == '__main__':
    house_price_regression_pipeline()

