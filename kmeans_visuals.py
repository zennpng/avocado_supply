import copy
import numpy as np
import pandas as pd 
from kmeans_preprocessing import xlist, ylist
import matplotlib.pyplot as plt


# repeat K means clustering process
df = pd.DataFrame({
    'x': xlist,
    'y': ylist
})
np.random.seed(18)
k = 9
centroids = {
    i+1: [np.random.randint(-120, -70), np.random.randint(25, 45)]
    for i in range(k)
}

# define color map
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'y', 5: 'm', 6: 'c', 7: 'lime', 8: 'darkred', 9: 'indigo'}

def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

df = assignment(df, centroids)
old_centroids = copy.deepcopy(centroids)

def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k

centroids = update(centroids)
# Repeat Assigment Stage
df = assignment(df, centroids)

while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break

# produce visualization 
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(-130, -70)
plt.ylim(20, 50)
plt.show()