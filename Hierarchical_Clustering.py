import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"Clustering\hierarchical_customer_segmentation.csv")

print(data.head())

X = data[['AnnualIncome', 'SpendingScore']]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage

linked = linkage(
    X_scaled,
    method='ward'
)

plt.figure(figsize=(12,6))

dendrogram(linked)

plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")

plt.show()

from sklearn.cluster import AgglomerativeClustering

model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

clusters = model.fit_predict(X_scaled)

data['Cluster'] = clusters

cluster_names = {
    0: "Budget Customers",
    1: "Premium Customers",
    2: "Medium Customers"
}

data['ClusterName'] = data['Cluster'].map(cluster_names)

print(data.head())

from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, clusters)

print("Silhouette Score:", score)

plt.figure(figsize=(10,6))

for cluster in data['Cluster'].unique():

    cluster_data = data[data['Cluster'] == cluster]

    plt.scatter(
        cluster_data['AnnualIncome'],
        cluster_data['SpendingScore'],
        label=cluster_names[cluster],
        s=60
    )

plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using Hierarchical Clustering")

plt.legend()

plt.show()

print(data)