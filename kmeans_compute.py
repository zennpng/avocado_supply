import copy
import numpy as np
import pandas as pd 
from kmeans_preprocessing import xlist, ylist


# main k means clustering function 
def kmeans(k):
    df = pd.DataFrame({
        'x': xlist,
        'y': ylist
    })
    np.random.seed(18)
    
    # initiate centroids
    centroids = {
        i+1: [np.random.randint(-120, -70), np.random.randint(25, 45)]
        for i in range(k)
    }
    
    # function for assignment step 
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
        #df['color'] = df['closest'].map(lambda x: colmap[x])
        return df
    
    # assign data to centroids (1st cycle)
    df = assignment(df, centroids)
    
    # copy centroids 
    old_centroids = copy.deepcopy(centroids)
    
    # update centroids function 
    def update(k):
        for i in centroids.keys():
            centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
            centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
        return k
      
    # update centroids and repeat assignment (2nd cycle)
    centroids = update(centroids)
    df = assignment(df, centroids)
  
    # keep repeating cycle till convergence occurs 
    while True:
        closest_centroids = df['closest'].copy(deep=True)
        centroids = update(centroids)
        df = assignment(df, centroids)
        if closest_centroids.equals(df['closest']):
            break
            
    # assignment (final cycle)
    df = assignment(df, centroids)
    
    # cleaning of result 
    df = df.rename(columns={'closest': 'k={}_group'.format(k)})
    df = df.drop(['x','y','distance_from_1','distance_from_2'],axis=1)
    dff = pd.DataFrame()
    dff['k={}_group'.format(k)] = df['k={}_group'.format(k)].tolist()
    return dff

# create groupings for different K values 
result = pd.DataFrame()
for i in range(2,46):
    result['k={}_group'.format(i)] = kmeans(i)['k={}_group'.format(i)]

# mapping region names to grouped data 
regions = ['Jacksonville','MiamiFtLauderdale','Tampa','HartfordSpringfield','Sacramento','Columbus','BaltimoreWashington','RichmondNorfolk','Denver','SanFrancisco',
           'CincinnatiDayton','Indianapolis','StLouis','Roanoke','Louisville','WestTexNewMexico','Charlotte','LasVegas','RaleighGreensboro','Portland','Spokane','Seattle',
           'BuffaloRochester','Syracuse','Albany','Boston','Detroit','NewYork','GrandRapids','HarrisburgScranton','Pittsburgh','GreatLakes','Chicago','Philadelphia','Atlanta',
           'Nashville','LosAngeles','SouthCarolina','PhoenixTucson','DallasFtWorth','Boise','SanDiego','Houston','Orlando','NewOrleansMobile']
result['region'] = regions
result = result.sort_values(by='region')
cols = result.columns.tolist()
cols = cols[-1:] + cols[:-1]
result = result[cols]

# exporting resulting dataframe
result.to_excel("groups.xlsx")