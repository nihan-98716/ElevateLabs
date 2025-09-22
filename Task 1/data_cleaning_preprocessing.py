# In a Colab environment, you can install the necessary libraries by running:
# !pip install pandas numpy seaborn matplotlib scikit-learn

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def preprocess_titanic_data(url):
    """
    A complete function to load and preprocess the Titanic dataset.

    Args:
        url (str): The URL to the Titanic CSV dataset.

    Returns:
        pandas.DataFrame: The cleaned and preprocessed DataFrame.
    """
    print("--- Starting Data Preprocessing ---")

    # 1. Import the dataset and explore basic info
    print("\n[Step 1] Importing and Exploring Dataset...")
    
    # The following line automatically downloads the dataset from the provided URL.
    # This is the most direct way to load data in Colab without manual uploads.
    print(f"Loading dataset automatically from URL: {url}")
    df = pd.read_csv(url)

    # --- ALTERNATIVE: UPLOAD FROM YOUR COMPUTER ---
    # If you want to upload a file from your own computer, comment out the line
    # above (`df = pd.read_csv(url)`) and uncomment the lines below.
    # from google.colab import files
    # import io
    #
    # print("\nPlease upload the titanic.csv file:")
    # uploaded = files.upload()
    # file_name = next(iter(uploaded)) # Get the name of the uploaded file
    # df = pd.read_csv(io.BytesIO(uploaded[file_name]))
    # print(f"\nSuccessfully loaded {file_name}")
    # -----------------------------------------------

    print("Original Dataset Shape:", df.shape)
    print("Original Dataset Info:")
    df.info()
    print("\nOriginal Missing Values Count:")
    print(df.isnull().sum())
    print("-" * 35)

    # 2. Handle missing values
    print("\n[Step 2] Handling Missing Values...")
    # Using the median for 'Age' is robust to outliers
    median_age = df['Age'].median()
    df['Age'].fillna(median_age, inplace=True)
    print("Missing values in 'Age' column filled with median value:", median_age)
    print("\nMissing Values Count After Handling:")
    print(df.isnull().sum())
    print("-" * 35)

    # 3. Convert categorical features into numerical
    print("\n[Step 3] Encoding Categorical Features...")
    # Convert 'Sex' into a numerical format (0 for female, 1 for male)
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    
    # Drop columns that are not useful for typical modeling tasks
    df.drop('Name', axis=1, inplace=True)
    print("Encoded 'Sex' column and dropped 'Name' column.")
    print("\nData Head After Encoding:")
    print(df.head())
    print("-" * 35)

    # 4. Normalize/standardize the numerical features
    print("\n[Step 4] Standardizing Numerical Features...")
    numerical_features = ['Age', 'Fare']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    print("Standardized 'Age' and 'Fare' columns using StandardScaler.")
    print("\nData Head After Standardization:")
    print(df.head())
    print("-" * 35)

    # 5. Visualize and remove outliers
    print("\n[Step 5] Visualizing and Removing Outliers...")
    
    # Visualize outliers before removal
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df['Fare'])
    plt.title('Boxplot of Fare (Before Outlier Removal)')
    plt.ylabel('Standardized Fare')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['Age'])
    plt.title('Boxplot of Age (Before Outlier Removal)')
    plt.ylabel('Standardized Age')
    plt.suptitle('Visualizing Outliers Before Removal', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Identify and remove outliers using the IQR method
    print("\nRemoving outliers based on IQR...")
    Q1 = df[numerical_features].quantile(0.25)
    Q3 = df[numerical_features].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out outliers
    df_no_outliers = df[~((df[numerical_features] < lower_bound) | (df[numerical_features] > upper_bound)).any(axis=1)]
    
    print(f"\nOriginal dataset shape: {df.shape}")
    print(f"Dataset shape after removing outliers: {df_no_outliers.shape}")
    print(f"Removed {df.shape[0] - df_no_outliers.shape[0]} rows.")

    # Visualize data after removing outliers
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(y=df_no_outliers['Fare'])
    plt.title('Boxplot of Fare (After Outlier Removal)')
    plt.ylabel('Standardized Fare')

    plt.subplot(1, 2, 2)
    sns.boxplot(y=df_no_outliers['Age'])
    plt.title('Boxplot of Age (After Outlier Removal)')
    plt.ylabel('Standardized Age')
    plt.suptitle('Data Distribution After Removing Outliers', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("-" * 35)
    
    print("\n--- Data Preprocessing Complete ---")
    print("\nFinal Data Head:")
    print(df_no_outliers.head())
    
    return df_no_outliers

if __name__ == '__main__':
    # URL for the Titanic dataset
    TITANIC_URL = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
    
    # Run the preprocessing pipeline
    cleaned_df = preprocess_titanic_data(TITANIC_URL)

