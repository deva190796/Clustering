import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances

data = {
    'Point': ['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12'],
    'X': [3,4,5,6,7,6,7,8,3,2,3,2],
    'Y': [7,6,5,4,3,2,2,4,3,6,5,4]
}

df = pd.DataFrame(data)

print(df)

X = df[['X', 'Y']]

distance_matrix = euclidean_distances(X)

eps = 1.9
min_samples = 4

print("\nNeighbors within eps = 1.9\n")

for i in range(len(df)):

    neighbors = []

    for j in range(len(df)):

        if i != j and distance_matrix[i][j] <= eps:

            neighbors.append(df['Point'][j])

    print(f"{df['Point'][i]} : {neighbors}")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

model = DBSCAN(
    eps=1.9,
    min_samples=4
)

clusters = model.fit_predict(X_scaled)

df['Cluster'] = clusters

print("\nClustered Data")
print(df)

plt.figure(figsize=(10,6))

for cluster in df['Cluster'].unique():

    cluster_data = df[df['Cluster'] == cluster]

    if cluster == -1:

        plt.scatter(
            cluster_data['X'],
            cluster_data['Y'],
            label='Noise',
            s=100,
            marker='x'
        )

    else:

        plt.scatter(
            cluster_data['X'],
            cluster_data['Y'],
            label=f'Cluster {cluster}',
            s=100
        )

for i in range(len(df)):

    plt.text(
        df['X'][i],
        df['Y'][i],
        df['Point'][i]
    )

plt.xlabel("X")
plt.ylabel("Y")
plt.title("DBSCAN Clustering")

plt.legend()

plt.show()