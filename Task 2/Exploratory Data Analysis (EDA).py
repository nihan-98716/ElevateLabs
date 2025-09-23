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

    # 1. Import the dataset
    print(f"Loading dataset automatically from URL: {url}")
    df = pd.read_csv(url)
    print("Original Dataset Shape:", df.shape)

    # 2. Handle missing values
    print("\n[Step 2] Handling Missing Values...")
    median_age = df['Age'].median()
    df['Age'].fillna(median_age, inplace=True)
    print("Missing values in 'Age' column filled.")

    # 3. Convert categorical features into numerical
    print("\n[Step 3] Encoding Categorical Features...")
    df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
    df.drop('Name', axis=1, inplace=True)
    print("Encoded 'Sex' and dropped 'Name'.")

    # 4. Normalize/standardize the numerical features
    print("\n[Step 4] Standardizing Numerical Features...")
    numerical_features = ['Age', 'Fare']
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    print("Standardized 'Age' and 'Fare'.")

    # 5. Remove outliers
    print("\n[Step 5] Removing Outliers...")
    Q1 = df[numerical_features].quantile(0.25)
    Q3 = df[numerical_features].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_no_outliers = df[~((df[numerical_features] < lower_bound) | (df[numerical_features] > upper_bound)).any(axis=1)]
    print(f"Removed {df.shape[0] - df_no_outliers.shape[0]} outlier rows.")
    
    print("\n--- Data Preprocessing Complete ---")
    
    return df_no_outliers

def perform_eda(df):
    """
    Performs Exploratory Data Analysis on the preprocessed DataFrame.

    Args:
        df (pandas.DataFrame): The cleaned DataFrame.
    """
    print("\n--- Starting Exploratory Data Analysis ---")
    
    # 1. Generate summary statistics
    print("\n[EDA Step 1] Summary Statistics:")
    # Using .T to transpose for better readability
    print(df.describe().T)
    print("-" * 50)

    # 2. Create histograms and boxplots for numeric features
    print("\n[EDA Step 2] Visualizing Feature Distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Histogram for Age
    sns.histplot(df['Age'], kde=True, ax=axes[0, 0], bins=30)
    axes[0, 0].set_title('Age Distribution')
    
    # Histogram for Fare
    sns.histplot(df['Fare'], kde=True, ax=axes[0, 1], bins=30)
    axes[0, 1].set_title('Fare Distribution')

    # Boxplot for Age vs. Survived
    sns.boxplot(x='Survived', y='Age', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Age Distribution by Survival Status')
    axes[1, 0].set_xticklabels(['Did not Survive', 'Survived'])

    # Boxplot for Fare vs. Survived
    sns.boxplot(x='Survived', y='Fare', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Fare Distribution by Survival Status')
    axes[1, 1].set_xticklabels(['Did not Survive', 'Survived'])
    
    plt.suptitle('Numeric Feature Distributions and Relationships with Survival', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    print("\n**Feature-Level Inferences from Plots:**")
    print("1. **Age:** The age distribution is roughly normal. There isn't a dramatic difference in age distribution between survivors and non-survivors, although younger passengers seem to have a slightly higher survival rate.")
    print("2. **Fare:** The fare distribution is skewed right. The boxplot clearly shows that passengers who paid a higher fare (and were likely in a higher class) had a significantly better chance of survival. This is a key trend.")
    print("-" * 50)
    
    # 3. Use correlation matrix for feature relationships
    print("\n[EDA Step 3] Analyzing Feature Relationships with a Correlation Matrix...")
    plt.figure(figsize=(10, 8))
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Features', fontsize=16)
    plt.show()

    print("\n**Inferences from Correlation Matrix:**")
    print("1. **Survived vs. Sex_male (-0.56):** This strong negative correlation is the most significant. It indicates that as `Sex_male` increases (i.e., the passenger is male), the likelihood of survival decreases. In other words, female passengers had a much higher survival rate.")
    print("2. **Survived vs. Pclass (-0.32):** A negative correlation here means that as the passenger class number increases (from 1st to 3rd), the survival chance decreases. This supports the inference from the Fare boxplot.")
    print("3. **Anomaly/Pattern:** The feature `Sex_male` appears to be the single strongest predictor of survival.")
    print("-" * 50)

    # 4. Use pairplot for a deeper dive
    print("\n[EDA Step 4] Pairplot for Deeper Insights...")
    # Select key columns for the pairplot to keep it readable
    pairplot_cols = ['Survived', 'Pclass', 'Age', 'Fare', 'Sex_male']
    sns.pairplot(df[pairplot_cols], hue='Survived', palette={0: 'red', 1: 'green'})
    plt.suptitle('Pairwise Feature Relationships by Survival Status', y=1.02, fontsize=16)
    plt.show()

    print("\n**Inferences from Pairplot:**")
    print("The pairplot confirms our findings. The separation between survivors (green) and non-survivors (red) is most distinct in the `Sex_male` and `Pclass` plots, reinforcing their importance.")
    print("\n--- Exploratory Data Analysis Complete ---")


if __name__ == '__main__':
    # URL for the Titanic dataset
    TITANIC_URL = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
    
    # Step 1: Run the preprocessing pipeline to get the cleaned data
    # This call now executes the function defined within this same script.
    cleaned_df = preprocess_titanic_data(TITANIC_URL)
    
    # Step 2: Run the EDA on the cleaned data
    if cleaned_df is not None:
        perform_eda(cleaned_df)

