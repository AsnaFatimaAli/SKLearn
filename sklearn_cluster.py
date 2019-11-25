from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wine = datasets.load_wine()
x = wine.data

distance = []
for clusts in range(1,16):
    kmeans = KMeans(n_clusters = clusts).fit(x)
    distance.append(kmeans.inertia_)

plt.plot(range(1,16),distance)
plt.xticks(range(1,16))
plt.title("Sum of Squared Distance vs. Number of Clusters", fontsize= 11)
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Distance from Cluster Point")
# Using Elbow Heuristic 
plt.axvline(x = 4, linestyle='dashed', color='black')
plt.show()

# Based on the elbow heuristic the assumed number of clusters for this data is 4
