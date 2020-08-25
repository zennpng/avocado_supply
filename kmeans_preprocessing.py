import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


# get region x and y coordinates (latitude and longitude) 
region = pd.read_csv("regions coordinates.csv")
region = region.drop('NAME',axis=1)
xlist = region['xcoord'].tolist()
ylist = region['ycoord'].tolist()

# Visualizing optimal K using silhouette method
sil = []   
kmax = 10

for k in range(2, kmax+1):   # dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
    kmeans = KMeans(n_clusters = k, random_state=102).fit(region)
    labels = kmeans.labels_
    sil.append(silhouette_score(region, labels, metric = 'euclidean'))
plt.plot(list(range(2, kmax+1)), sil)

# Visualizing optimal K using elbow method
distortions = []
K = range(1, 45)   # K range: 1 (max pooling) to 45 (no pooling)

for k in K:
    kmeanModel = KMeans(n_clusters=k, random_state=102)
    kmeanModel.fit(region)
    distortions.append(kmeanModel.inertia_)

z = plt.figure(figsize=(16, 8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('sum of square errors')
plt.title('The Elbow Method showing the optimal k')
plt.show()