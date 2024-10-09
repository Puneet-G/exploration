# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data with clear clustering profiles
data_clear_clusters = pd.DataFrame({
    'Customer_ID': range(1, 51),
    'Monthly_Spend_Essentials': np.concatenate([
        np.random.normal(700, 50, 15),  # High spenders
        np.random.normal(300, 30, 20),  # Low spenders
        np.random.normal(500, 40, 15)   # Moderate spenders
    ]).round(2),
    'Monthly_Spend_Discretionary': np.concatenate([
        np.random.normal(600, 40, 15),  # High discretionary
        np.random.normal(200, 20, 20),  # Low discretionary
        np.random.normal(400, 30, 15)   # Moderate discretionary
    ]).round(2),
    'Monthly_Spend_Luxury': np.concatenate([
        np.random.normal(400, 30, 15),  # High luxury
        np.random.normal(100, 20, 20),  # Low luxury
        np.random.normal(200, 25, 15)   # Moderate luxury
    ]).round(2),
    'Transaction_Frequency': np.concatenate([
        np.random.randint(15, 20, 15),  # High frequency
        np.random.randint(5, 10, 20),   # Low frequency
        np.random.randint(10, 15, 15)   # Moderate frequency
    ])
})

# Standardize data for clustering
scaler = StandardScaler()
data_scaled_clear = scaler.fit_transform(data_clear_clusters[['Monthly_Spend_Essentials', 
                                                              'Monthly_Spend_Discretionary', 
                                                              'Monthly_Spend_Luxury', 
                                                              'Transaction_Frequency']])

# Apply k-means clustering for 3 clusters
kmeans_clear = KMeans(n_clusters=3, random_state=42)
data_clear_clusters['Cluster'] = kmeans_clear.fit_predict(data_scaled_clear)

# Calculate silhouette scores for each sample
silhouette_values_clear = silhouette_samples(data_scaled_clear, data_clear_clusters['Cluster'])
data_clear_clusters['Silhouette_Score'] = silhouette_values_clear

# Calculate overall and per-cluster silhouette scores
silhouette_avg_clear = silhouette_score(data_scaled_clear, data_clear_clusters['Cluster'])
silhouette_by_cluster_clear = data_clear_clusters.groupby('Cluster')['Silhouette_Score'].mean()

# Elbow Method for WCSS
wcss_clear = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_scaled_clear)
    wcss_clear.append(kmeans.inertia_)

# Plot WCSS for the elbow method
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss_clear, marker='o')
plt.title("Elbow Method for Optimal k (Clear Clusters)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
plt.show()

# Visualize the clusters by Spend Essentials vs. Discretionary
plt.figure(figsize=(14, 6))

# Essentials vs. Discretionary Spend
plt.subplot(1, 2, 1)
sns.scatterplot(x='Monthly_Spend_Essentials', y='Monthly_Spend_Discretionary', hue='Cluster', data=data_clear_clusters, palette='viridis')
plt.title('Essentials vs. Discretionary Spend by Cluster (Clear Separation)')

# Essentials vs. Luxury Spend
plt.subplot(1, 2, 2)
sns.scatterplot(x='Monthly_Spend_Essentials', y='Monthly_Spend_Luxury', hue='Cluster', data=data_clear_clusters, palette='viridis')
plt.title('Essentials vs. Luxury Spend by Cluster (Clear Separation)')

plt.tight_layout()
plt.show()

# 3D plot of clusters using Essentials, Discretionary, and Luxury spend
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotting each cluster with a different color
scatter = ax.scatter(
    data_clear_clusters['Monthly_Spend_Essentials'], 
    data_clear_clusters['Monthly_Spend_Discretionary'], 
    data_clear_clusters['Monthly_Spend_Luxury'], 
    c=data_clear_clusters['Cluster'], cmap='viridis', marker='o'
)

# Labeling the axes
ax.set_xlabel('Monthly Spend on Essentials')
ax.set_ylabel('Monthly Spend on Discretionary')
ax.set_zlabel('Monthly Spend on Luxury')
plt.title("3D Cluster Plot: Essentials vs. Discretionary vs. Luxury Spend")

# Show color legend
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

plt.show()

# Calculate descriptive statistics for each cluster to summarize cluster characteristics
cluster_characteristics = data_clear_clusters.groupby('Cluster').agg({
    'Monthly_Spend_Essentials': ['mean', 'std', 'min', 'max'],
    'Monthly_Spend_Discretionary': ['mean', 'std', 'min', 'max'],
    'Monthly_Spend_Luxury': ['mean', 'std', 'min', 'max'],
    'Transaction_Frequency': ['mean', 'std', 'min', 'max']
}).round(2)

# Display results
print("Overall Silhouette Score:", silhouette_avg_clear)
print("\nSilhouette Scores by Cluster:\n", silhouette_by_cluster_clear)
print("\nCluster Characteristics:\n", cluster_characteristics)
