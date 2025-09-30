# In a Colab environment, you can install the necessary libraries by running:
# !pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap

def knn_classification_pipeline():
    """
    A complete and polished pipeline for KNN classification using the Iris dataset.
    """
    print("--- Starting KNN Classification Pipeline for Iris Dataset ---")

    # 1. Load Dataset and Normalize Features
    print("\n[Step 1] Loading and Normalizing Data...")
    
    # Load the classic Iris dataset from scikit-learn
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Create a DataFrame for easier exploration (optional, but good practice)
    df = pd.DataFrame(X, columns=iris.feature_names)
    print("Dataset loaded successfully. Features:", ", ".join(iris.feature_names))
    
    # Split data *before* scaling to prevent data leakage from the test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Features have been normalized.")

    # 2. Experiment with different values of K
    print("\n[Step 2] Finding the Optimal Value of K...")
    
    k_range = range(1, 26)
    accuracies = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        accuracies.append(accuracy_score(y_test, y_pred))
        
    # Plotting the results to find the "elbow"
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies, marker='o', linestyle='--')
    plt.title('Elbow Method for Finding Optimal K', fontsize=16)
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()
    
    # Find the optimal K
    optimal_k = k_range[np.argmax(accuracies)]
    print(f"The optimal value of K was found to be {optimal_k} with an accuracy of {max(accuracies):.2f}.")

    # 3. Train the Final Model and Evaluate
    print(f"\n[Step 3] Training Final KNN Model with K={optimal_k}...")
    
    # Use the best K value to train the final model
    knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_final.fit(X_train_scaled, y_train)
    y_pred_final = knn_final.predict(X_test_scaled)
    
    print("\n[Step 4] Evaluating the Final Model...")
    
    # Accuracy Score
    final_accuracy = accuracy_score(y_test, y_pred_final)
    print(f"  - Final Model Accuracy: {final_accuracy:.2f}")
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_final, target_names=iris.target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_final)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # 5. Visualize Decision Boundaries
    print("\n[Step 5] Visualizing Decision Boundaries...")
    print("Note: Visualization is done using the first two features (Sepal Length and Sepal Width) for a 2D plot.")
    
    # We will use only the first two features for this visualization
    X_vis = X[:, :2]
    
    # We need to re-train the scaler and model on these two features
    scaler_vis = StandardScaler()
    X_vis_scaled = scaler_vis.fit_transform(X_vis)
    knn_vis = KNeighborsClassifier(n_neighbors=optimal_k)
    knn_vis.fit(X_vis_scaled, y) # Train on the full dataset for a better boundary map
    
    # Create a mesh grid
    h = .02  # step size in the mesh
    x_min, x_max = X_vis_scaled[:, 0].min() - 1, X_vis_scaled[:, 0].max() + 1
    y_min, y_max = X_vis_scaled[:, 1].min() - 1, X_vis_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict for each point in the mesh
    Z = knn_vis.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create color maps and color list
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    colors_bold = ['#FF0000', '#00FF00', '#0000FF']
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=cmap_light)
    
    # Plot the training points
    sns.scatterplot(x=X_vis_scaled[:, 0], y=X_vis_scaled[:, 1], hue=iris.target_names[y],
                    palette=colors_bold, alpha=1.0, edgecolor="black")
    
    plt.title(f'KNN Decision Boundaries for Iris Dataset (K={optimal_k})', fontsize=16)
    plt.xlabel(f'Scaled {iris.feature_names[0]}')
    plt.ylabel(f'Scaled {iris.feature_names[1]}')
    plt.legend()
    plt.show()
    
    print("\n--- Pipeline Complete ---")

if __name__ == '__main__':
    knn_classification_pipeline()

