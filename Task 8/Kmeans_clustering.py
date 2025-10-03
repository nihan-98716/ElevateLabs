# In a Colab environment, you can install the necessary libraries by running:
# !pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

def find_optimal_k(scaled_data, k_range):
    """
    Calculates inertia and silhouette scores for a range of k values.
    """
    inertia_scores = []
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init='auto')
        kmeans.fit(scaled_data)
        inertia_scores.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
    
    # Plotting the results
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia_scores, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K', fontsize=16)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o', linestyle='--')
    plt.title('Silhouette Scores for Optimal K', fontsize=16)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"Based on Silhouette Score analysis, the optimal number of clusters is K = {optimal_k}.")
    return optimal_k

def interpret_clusters(df_with_clusters, scaler, numerical_features):
    """
    Provides a summary of each cluster's characteristics.
    """
    print("\n--- Wholesale Customer Segment Interpretation ---")
    # Invert scaling to get original feature values back for interpretation
    original_features = scaler.inverse_transform(df_with_clusters[numerical_features])
    df_original_features = pd.DataFrame(original_features, columns=numerical_features, index=df_with_clusters.index)
    df_with_clusters[numerical_features] = df_original_features
    
    # Create a summary of each cluster's mean spending
    agg_dict = {col: ['mean'] for col in numerical_features}
    cluster_summary = df_with_clusters.groupby('Cluster').agg(agg_dict).round(0)
    
    print(cluster_summary)

def main():
    """
    Main function to run the optimized K-Means clustering pipeline.
    """
    print("--- Starting Optimized K-Means Clustering Pipeline ---")

    # 1. Load and Prepare Data
    # Using the "Wholesale customers" dataset from the UCI repository
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
    df = pd.read_csv(url)
    
    # For this clustering analysis, we focus on the spending features
    numerical_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    df_features = df[numerical_features]

    # Preprocessing: All features are numerical, so we only need to scale them
    scaler = StandardScaler()
    X_processed = scaler.fit_transform(df_features)
    print("\n[Step 1] Data loaded and features preprocessed successfully.")

    # 2. Find Optimal K
    k_range = range(2, 11)
    optimal_k = find_optimal_k(X_processed, k_range)

    # 3. Fit Final K-Means Model
    kmeans_final = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init='auto')
    df['Cluster'] = kmeans_final.fit_predict(X_processed)
    print(f"\n[Step 2] Final K-Means model fitted with K={optimal_k}.")

    # 4. Interpret the Clusters
    print("\n[Step 3] Analyzing cluster characteristics...")
    interpret_clusters(df.copy(), scaler, numerical_features)

    # 5. Visualize Clusters using PCA
    print("\n[Step 4] Visualizing clusters using PCA for dimensionality reduction...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    df['PCA1'] = X_pca[:, 0]
    df['PCA2'] = X_pca[:, 1]
    
    centroids_processed = kmeans_final.cluster_centers_
    centroids_pca = pca.transform(centroids_processed)

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', 
                    palette=sns.color_palette('viridis', n_colors=optimal_k), 
                    s=100, alpha=0.8, legend='full')
    
    plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], s=300, c='red', 
                marker='X', label='Centroids', edgecolors='black')
    
    plt.title('Wholesale Customer Segments (Visualized with PCA)', fontsize=18)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("\n--- Pipeline Complete ---")

if __name__ == '__main__':
    main()

