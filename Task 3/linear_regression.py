# In a Colab environment, you can install the necessary libraries by running:
# !pip install pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def house_price_regression_pipeline():
    """
    A complete pipeline for a house price prediction task using Linear Regression.
    """
    print("--- Starting House Price Prediction Pipeline ---")

    # 1. Import and preprocess the dataset
    print("\n[Step 1] Loading and Preprocessing Data...")
    
    # The dataset will be automatically downloaded from this URL. No manual intervention needed.
    # This is the Boston Housing dataset.
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    # Feature names for the Boston Housing dataset
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    
    df = pd.DataFrame(data, columns=feature_names)
    df['PRICE'] = target

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)
    
    # Check for missing values (this dataset is clean)
    print("Missing values count:", df.isnull().sum().sum())

    # Define features (X) and target (y)
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']

    # 2. Split data into train-test sets
    print("\n[Step 2] Splitting Data into Training and Testing Sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # 3. Fit a Linear Regression model
    print("\n[Step 3] Fitting the Linear Regression Model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # 4. Evaluate the model
    print("\n[Step 4] Evaluating the Model...")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"  - Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  - Mean Squared Error (MSE): {mse:.2f}")
    print(f"  - R-squared (R²): {r2:.2f}")
    print("-" * 50)
    print("**Interpretation of Metrics:**")
    print(f"The R² of {r2:.2f} means that our model explains approximately {r2:.0%} of the variance in house prices.")
    print(f"The MAE indicates that, on average, our model's predictions are off by about ${mae:.2f}k.")

    # 5. Plot regression line and interpret coefficients
    print("\n[Step 5] Visualizing Results and Interpreting Coefficients...")
    
    # Scatter plot of Actual vs. Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Prices ($1000s)")
    plt.ylabel("Predicted Prices ($1000s)")
    plt.title("Actual vs. Predicted House Prices", fontsize=16)
    plt.grid(True)
    plt.show()

    print("\n**Interpretation of the Plot:**")
    print("The points should ideally fall on the red dashed line (Perfect Prediction Line).")
    print("Our model's predictions follow the trend, but there is noticeable scatter, indicating prediction errors.")
    
    # Interpret Coefficients
    coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
    coefficients.sort_values('Coefficient', ascending=False, inplace=True)
    
    print("\n**Model Coefficients:**")
    print(coefficients)
    
    print("\n**Interpretation of Coefficients:**")
    print(" - **Positive Coefficients (e.g., RM):** For each one-unit increase in this feature, the house price is predicted to increase by the coefficient's value (in $1000s), holding all other features constant.")
    print(" - **Negative Coefficients (e.g., LSTAT):** For each one-unit increase in this feature, the house price is predicted to decrease by the coefficient's value (in $1000s).")
    print(f" - **Example:** An increase of one room ('RM') is associated with a predicted price increase of approximately ${coefficients.loc['RM'].values[0]*1000:.2f}.")

    print("\n--- Pipeline Complete ---")

if __name__ == '__main__':
    house_price_regression_pipeline()
