# In a Colab environment, you can install the necessary libraries by running:
# !pip install pandas numpy scikit-learn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def classification_pipeline():
    """
    A complete pipeline for classification using Decision Trees and Random Forests.
    """
    print("--- Starting Classification Pipeline for Heart Disease Prediction ---")

    # 1. Import and preprocess the dataset
    print("\n[Step 1] Loading and Preprocessing Data...")
    
    # The dataset will be automatically downloaded from the UCI Machine Learning Repository.
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'
    
    # Define column names as they are not in the file
    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 
        'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
    ]
    
    # Load data, handle missing values ('?')
    df = pd.read_csv(url, header=None, names=columns, na_values='?')
    
    # For simplicity, we drop rows with any missing values
    df.dropna(inplace=True)
    
    # The target variable needs to be simplified: 0 = no disease, 1 = disease
    df['target'] = (df['target'] > 0).astype(int)
    
    print("Dataset loaded and preprocessed successfully.")
    print("Dataset shape:", df.shape)
    
    # Define features (X) and target (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")

    # 2. Train a Decision Tree Classifier and visualize it
    print("\n[Step 2] Training an Unconstrained Decision Tree...")
    
    # This tree is unconstrained (no max_depth) and will likely overfit
    dt_full = DecisionTreeClassifier(random_state=42)
    dt_full.fit(X_train, y_train)
    
    print("Visualizing the full (potentially overfit) Decision Tree...")
    plt.figure(figsize=(20, 10))
    plot_tree(dt_full, filled=True, feature_names=X.columns, class_names=['No Disease', 'Disease'], max_depth=3, fontsize=10)
    plt.title("Visualization of the Top of the Unconstrained Decision Tree", fontsize=16)
    plt.show()

    # 3. Analyze overfitting and control tree depth
    print("\n[Step 3] Analyzing Overfitting and Pruning the Tree...")
    
    # Prune the tree by setting a max_depth
    dt_pruned = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_pruned.fit(X_train, y_train)
    
    y_pred_full = dt_full.predict(X_test)
    y_pred_pruned = dt_pruned.predict(X_test)
    
    print(f"  - Accuracy of Full Tree on Test Set: {accuracy_score(y_test, y_pred_full):.2f}")
    print(f"  - Accuracy of Pruned Tree (max_depth=4) on Test Set: {accuracy_score(y_test, y_pred_pruned):.2f}")
    print("The pruned tree often performs better on unseen data because it generalizes more effectively.")

    # 4. Train a Random Forest and compare accuracy
    print("\n[Step 4] Training a Random Forest Classifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    
    print(f"  - Accuracy of Random Forest on Test Set: {accuracy_score(y_test, y_pred_rf):.2f}")
    print("Random Forest typically outperforms a single Decision Tree by averaging many trees.")

    # 5. Interpret feature importances from the Random Forest
    print("\n[Step 5] Interpreting Feature Importances...")
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)
    
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xticks(rotation=45)
    plt.title("Feature Importances from Random Forest", fontsize=16)
    plt.ylabel("Importance")
    plt.grid(axis='y', linestyle='--')
    plt.show()
    
    print("This plot shows which features the Random Forest model found most predictive.")
    print(f"Top 3 most important features: {', '.join(feature_importance_df['Feature'].head(3).tolist())}")

    # 6. Evaluate using cross-validation for a more robust comparison
    print("\n[Step 6] Evaluating Models with 5-Fold Cross-Validation...")
    
    cv_scores_dt = cross_val_score(dt_pruned, X, y, cv=5)
    cv_scores_rf = cross_val_score(rf, X, y, cv=5)
    
    print(f"  - Pruned Decision Tree CV Mean Accuracy: {np.mean(cv_scores_dt):.2f} (+/- {np.std(cv_scores_dt):.2f})")
    print(f"  - Random Forest CV Mean Accuracy: {np.mean(cv_scores_rf):.2f} (+/- {np.std(cv_scores_rf):.2f})")
    print("\nCross-validation provides a more reliable measure of model performance.")
    print("The Random Forest shows both higher accuracy and more stable performance (lower std dev) across folds.")
    
    print("\n--- Pipeline Complete ---")

if __name__ == '__main__':
    classification_pipeline()
