import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(42)
X = np.vstack([np.random.randn(50,2)+np.array([0,0]), np.random.randn(50,2)+np.array([5,5])])
kmeans = KMeans(n_clusters=2, random_state=42).fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

plt.scatter(X[:,0], X[:,1], c=labels, cmap='viridis')
plt.scatter(centers[:,0], centers[:,1], marker='x', color='red', s=100)
plt.title('KMeans Clusters'); plt.show()
print('Inertia (WCSS)', round(kmeans.inertia,3))
