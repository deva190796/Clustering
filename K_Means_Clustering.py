import pandas as pd
import numpy as np

data = {
    'Object': [1, 2, 3, 4, 5],
    'X': [2, 4, 6, 10, 12],
    'Y': [4, 6, 8, 4, 4]
}

df = pd.DataFrame(data)

print(df)

X = df[['X', 'Y']]

initial_centroids = np.array([
    [4, 6],
    [12, 4]
])

from sklearn.cluster import KMeans

model = KMeans(
    n_clusters=2,
    init=initial_centroids,
    n_init=1,
    random_state=42
)

model.fit(X)

df['Cluster'] = model.labels_

print("\nClustered Data")
print(df)

print("\nFinal Centroids")
print(model.cluster_centers_)

import matplotlib.pyplot as plt

plt.scatter(df['X'], df['Y'], c=df['Cluster'], s=100)

centroids = model.cluster_centers_

plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker='X',
    s=300
)

for i in range(len(df)):
    plt.text(df['X'][i], df['Y'][i], df['Object'][i])

plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-Means Clustering")

plt.show()

from sklearn.metrics import silhouette_score

score = silhouette_score(X, model.labels_)

print(score)