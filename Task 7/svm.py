# In a Colab environment, you can install the necessary libraries by running:
# !pip install pandas numpy scikit-learn matplotlib seaborn

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

def plot_decision_boundary(X, y, model, title):
    """
    Visualizes the decision boundary for a given model on 2D data.
    """
    # Create a mesh grid
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict for each point in the mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and the data points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Feature 1 (Scaled)')
    plt.ylabel('Feature 2 (Scaled)')
    plt.legend(handles=scatter.legend_elements()[0], labels=['Benign', 'Malignant'])
    plt.show()


def svm_classification_pipeline():
    """
    A complete pipeline for SVM classification using the Breast Cancer dataset.
    """
    print("--- Starting SVM Classification Pipeline for Breast Cancer Prediction ---")

    # 1. Load and Prepare the Dataset
    print("\n[Step 1] Loading and Preprocessing Data...")
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    print(f"Dataset loaded successfully. Shape: {X.shape}")
    print(f"Classes: {', '.join(cancer.target_names)}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features have been scaled.")

    # 2. Train an SVM with a Linear Kernel
    print("\n[Step 2] Training SVM with a Linear Kernel...")
    svm_linear = SVC(kernel='linear', random_state=42)
    svm_linear.fit(X_train_scaled, y_train)
    y_pred_linear = svm_linear.predict(X_test_scaled)
    
    print(f"  - Accuracy of Linear SVM: {accuracy_score(y_test, y_pred_linear):.2f}")

    # 3. Train an SVM with an RBF Kernel (and Tune Hyperparameters)
    print("\n[Step 3 & 4] Tuning Hyperparameters (C, gamma) for RBF Kernel using GridSearchCV...")
    
    # Define an expanded parameter grid to search for better performance
    param_grid = {
        'C': [0.1, 1, 10, 100, 1000],              # Regularization parameter
        'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001]  # Kernel coefficient
    }
    
    # Use GridSearchCV to find the best parameters with 5-fold cross-validation
    grid = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid, refit=True, verbose=0, cv=5)
    grid.fit(X_train_scaled, y_train)
    
    print(f"  - Best hyperparameters found: {grid.best_params_}")
    
    # Use the best estimator to make predictions
    best_svm_rbf = grid.best_estimator_
    y_pred_rbf = best_svm_rbf.predict(X_test_scaled)
    
    print(f"  - Accuracy of Tuned RBF SVM: {accuracy_score(y_test, y_pred_rbf):.2f}")
    
    # 5. Evaluate Final Model using Cross-Validation
    print("\n[Step 5] Final Evaluation using Cross-Validation on the Full Dataset...")
    
    # Create a pipeline to prevent data leakage during cross-validation
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', best_svm_rbf) # Use the best estimator found by GridSearchCV
    ])
    
    # Perform 5-fold cross-validation on the entire dataset using the pipeline
    cv_scores = cross_val_score(pipeline, X, y, cv=5)
    print(f"  - Mean CV Accuracy: {np.mean(cv_scores):.3f} (+/- {np.std(cv_scores):.3f})")
    print("This provides a more robust estimate of the model's performance on unseen data.")

    print("\nFinal Model Performance on the Test Set:")
    print(classification_report(y_test, y_pred_rbf, target_names=cancer.target_names))
    
    cm = confusion_matrix(y_test, y_pred_rbf)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cancer.target_names, yticklabels=cancer.target_names)
    plt.title('Confusion Matrix for Tuned RBF SVM', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # 6. Visualize Decision Boundaries using the first two features
    print("\n[Step 6] Visualizing Decision Boundaries (using first two features)...")
    
    # Prepare 2D data for visualization
    X_vis = X[:, :2]
    scaler_vis = StandardScaler()
    X_vis_scaled = scaler_vis.fit_transform(X_vis)
    
    # Train a linear SVM on 2D data
    svm_linear_vis = SVC(kernel='linear', random_state=42)
    svm_linear_vis.fit(X_vis_scaled, y)
    plot_decision_boundary(X_vis_scaled, y, svm_linear_vis, 'Decision Boundary for Linear SVM')

    # Train a tuned RBF SVM on 2D data
    # We need to re-tune for the 2D data
    grid_vis = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid, cv=5)
    grid_vis.fit(X_vis_scaled, y)
    best_svm_rbf_vis = grid_vis.best_estimator_
    print(f"  - Best params for 2D data: {grid_vis.best_params_}")
    plot_decision_boundary(X_vis_scaled, y, best_svm_rbf_vis, 'Decision Boundary for RBF SVM')

    print("\n--- Pipeline Complete ---")

if __name__ == '__main__':
    svm_classification_pipeline()

