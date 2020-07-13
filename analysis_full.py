import math
import copy
import pandas as pd 
import pandasql as pdsql 
import numpy as np
from scipy.stats import norm
from scipy.stats import boxcox
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

pysql = lambda q: pdsql.sqldf(q, globals())



### 1) Data Preprocessing (After determining targeted regions using QGIS)

# import raw data
avo_raw = pd.read_csv("avocado.csv")

# removing these regions as they are not valuable to our analysis
avo = pysql('''SELECT * FROM avo_raw
                   WHERE region    
                   NOT IN ('California','Midsouth','Northeast','NorthernNewEngland','Plains','SouthCentral','Southeast','TotalUS','West')''')

# removing unimportant columns and renaming important columns
avo = avo.drop(['index','Total Bags','Small Bags','Large Bags','XLarge Bags','Total Volume','type'], axis=1)
avo = avo.rename(columns={'Date':'Previous_week_demand_date','4046':'Avo_4046_demand','4225':'Avo_4225_demand','4770':'Avo_4770_demand'})

# splitting avocado types: 1) organic and 2) conventional
avo_conventional = avo[0:7605]    
avo_organic = avo[7605:] 
avo_organic = avo_organic.reset_index(drop=True)

# performing train test split (training set -> 2015-2017 data, testing set -> 2018 data)
avo_organic_train = pysql(   
            """SELECT * from avo_organic    
             WHERE year != 2018
             ORDER BY region, year DESC""")
avo_organic_train = avo_organic_train.iloc[::-1].reset_index(drop = True)

avo_conventional_train = pysql(
            """SELECT * from avo_conventional 
             WHERE year != 2018
             ORDER BY region, year DESC""")
avo_conventional_train = avo_conventional_train.iloc[::-1].reset_index(drop = True)

avo_organic_test = pysql(
            """SELECT * from avo_organic 
             WHERE year = 2018
             ORDER BY region, year DESC""")
avo_organic_test = avo_organic_test.iloc[::-1].reset_index(drop = True)

avo_conventional_test = pysql(
            """SELECT * from avo_conventional 
             WHERE year = 2018
             ORDER BY region, year DESC""")
avo_conventional_test = avo_conventional_test.iloc[::-1].reset_index(drop = True)




### 2) Visualization of Demand Distribution (using Histograms)

# without trimming (avo4046_conventional example)
dfr = pd.concat([avo_conventional_train['Avo_4046_demand'], avo_conventional_train['region']], axis=1)
for (region, df_region) in dfr.groupby('region'):
    df_region_plot = df_region.plot.hist(bins=12)
    getfigure = df_region_plot.get_figure()
    getfigure.savefig("Demand_Histogram/avo4046_conventional_train/{}.png".format(region))

# with IQR trimming (avo4046_conventional example)
dfr = pd.concat([avo_conventional_train['Avo_4046_demand'], avo_conventional_train['region']], axis=1)

def remove_outlier(df_in, col_name):   
    q1 = df_in[col_name].quantile(0.25)
    q3 = df_in[col_name].quantile(0.75)
    iqr = q3-q1    #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
    return df_out

for region, df_region in dfr.groupby('region'):
    df_regionT = remove_outlier(df_region, 'Avo_4046_demand')
    df_region_plot = df_regionT.plot.hist(bins=50)
    getfigure = df_region_plot.get_figure()
    getfigure.savefig("NonNormal_Graphs/avo4046_conventional_train/{}.png".format(region))

# with IQR trimming and boxcox log transform (avo4046_conventional example)
dfr = pd.concat([avo_conventional_train['Avo_4046_demand'], avo_conventional_train['region']], axis=1)

for region, df_region in dfr.groupby('region'):
    df_regionT = remove_outlier(df_region,'Avo_4046_demand')
    data = df_regionT['Avo_4046_demand'].values
    data = data[data != 0]
    data = boxcox(data,0)
    plt.hist(data)
    plt.savefig("Demand_Histogram_TrimmedBoxcoxlog/avo4046_conventional_train/{}.jpg".format(region))

# with trim and boxcox square root transform (avo4046_conventional example)
dfr = pd.concat([avo_conventional_train['Avo_4046_demand'], avo_conventional_train['region']], axis=1)

for region, df_region in dfr.groupby('region'):
    df_regionT = remove_outlier(df_region,'Avo_4046_demand')
    data = df_regionT['Avo_4046_demand'].values
    data = data[data != 0]
    data = boxcox(data,0.5)
    plt.hist(data)
    plt.savefig("Demand_Histogram_TrimmedBoxcoxsquareroot/avo4046_conventional_train/{}.jpg".format(region))
    



### 3) Determining if demand can be considered normal under different transformations using:
#      - Chi^2 normality test
#      - Shapiro-Wilk test
#      - D'Agostino's k squared test 
#      The Final Q_star.xlsx file shows the results (data after transformation and transformation type for each data)




### 4) Data Trimming 

# conventional trimmed - each avocado type has a different dictionary, in each dict, each key = 'region' and each value = trimmed dataframe
avo4046_conventional_train_dict = {}
avo4225_conventional_train_dict = {}
avo4770_conventional_train_dict = {}

for region, df_region in avo_conventional_train.groupby('region'):
    df_region1 = remove_outlier(df_region,'Avo_4046_demand')
    df_region1 = df_region1.drop(['Previous_week_demand_date','AveragePrice','Avo_4225_demand','Avo_4770_demand','year'], axis=1)
    df_region2 = remove_outlier(df_region,'Avo_4225_demand')
    df_region2 = df_region2.drop(['Previous_week_demand_date','AveragePrice','Avo_4046_demand','Avo_4770_demand','year'], axis=1)
    df_region3 = remove_outlier(df_region,'Avo_4770_demand')
    df_region3 = df_region3.drop(['Previous_week_demand_date','AveragePrice','Avo_4225_demand','Avo_4046_demand','year'], axis=1)
    avo4046_conventional_train_dict.update({str(region): df_region1})
    avo4225_conventional_train_dict.update({str(region): df_region2})
    avo4770_conventional_train_dict.update({str(region): df_region3})

# organic trimmed - each avocado type has a different dictionary, in each dict, each key = 'region' and each value = trimmed dataframe
avo4046_organic_train_dict = {}
avo4225_organic_train_dict = {}
avo4770_organic_train_dict = {}

for region, df_region in avo_organic_train.groupby('region'):
    df_region1 = remove_outlier(df_region,'Avo_4046_demand')
    df_region1 = df_region1.drop(['Previous_week_demand_date','AveragePrice','Avo_4225_demand','Avo_4770_demand','year'], axis=1)
    df_region2 = remove_outlier(df_region,'Avo_4225_demand')
    df_region2 = df_region2.drop(['Previous_week_demand_date','AveragePrice','Avo_4046_demand','Avo_4770_demand','year'], axis=1)
    df_region3 = remove_outlier(df_region,'Avo_4770_demand')
    df_region3 = df_region3.drop(['Previous_week_demand_date','AveragePrice','Avo_4225_demand','Avo_4046_demand','year'], axis=1)
    avo4046_organic_train_dict.update({str(region): df_region1})
    avo4225_organic_train_dict.update({str(region): df_region2})
    avo4770_organic_train_dict.update({str(region): df_region3})



    
### 5) Calculating Alpha and Z values (for Q* calculations) 

# calculating mean
conventional_means = avo_conventional_train.groupby('region').mean().drop(['AveragePrice','year'], axis=1).rename(columns={'Avo_4046_demand':'Avo_4046_mean_demand','Avo_4225_demand':'Avo_4225_mean_demand','Avo_4770_demand':'Avo_4770_mean_demand'})  
organic_means = avo_organic_train.groupby('region').mean().drop(['AveragePrice','year'], axis=1).rename(columns={'Avo_4046_demand':'Avo_4046_mean_demand','Avo_4225_demand':'Avo_4225_mean_demand','Avo_4770_demand':'Avo_4770_mean_demand'})

# calculating standard deviation
conventional_std = avo_conventional_train.groupby('region').std().drop(['AveragePrice','year'], axis=1).rename(columns={'Avo_4046_demand':'Avo_4046_std_demand','Avo_4225_demand':'Avo_4225_std_demand','Avo_4770_demand':'Avo_4770_std_demand'}) 
organic_std = avo_organic_train.groupby('region').std().drop(['AveragePrice','year'], axis=1).rename(columns={'Avo_4046_demand':'Avo_4046_std_demand','Avo_4225_demand':'Avo_4225_std_demand','Avo_4770_demand':'Avo_4770_std_demand'})

# calculating overage cost
overage_cost_organic = pysql("""SELECT MIN(AveragePrice) as Minprice, year, region 
                        FROM avo_organic_train
                        GROUP BY region""")
overage_cost_conventional = pysql("""SELECT MIN(AveragePrice) as Minprice, year, region 
                        FROM avo_conventional_train
                        GROUP BY region""")

# calculating selling price
selling_price_organic = avo_organic_train.groupby('region').agg(lambda x: x.value_counts().index[0])
selling_price_organic = selling_price_organic.drop(['Previous_week_demand_date','Avo_4046_demand','Avo_4225_demand','Avo_4770_demand'],axis=1)
selling_price_organic = selling_price_organic.reset_index()

selling_price_conventional = avo_conventional_train.groupby('region').agg(lambda x:x.value_counts().index[0])
selling_price_conventional = selling_price_conventional.drop(['Previous_week_demand_date','Avo_4046_demand','Avo_4225_demand','Avo_4770_demand'],axis=1)
selling_price_conventional = selling_price_conventional.reset_index()

# calculating shortage cost
shortage_cost_organic = pd.DataFrame()
shortage_cost_organic['region'] = selling_price_organic['region']
shortage_cost_organic['AveragePrice'] = selling_price_organic['AveragePrice']-overage_cost_organic['Minprice']

shortage_cost_conventional = pd.DataFrame()
shortage_cost_conventional['region'] = selling_price_conventional['region']
shortage_cost_conventional['AveragePrice'] = selling_price_conventional['AveragePrice']-overage_cost_conventional['Minprice']

# alpha formula
def alpha(row):  
    return (row['AveragePrice'])/(row['AveragePrice']+row['Minprice'])
  
# z formula
def z_value(row):  
    return norm.ppf(row['alpha'])

# calculating alpha and z values organic
overage_cost_organic1 = overage_cost_organic.drop(['year','region'], axis=1)
alpha_organic = shortage_cost_organic.drop(['occurance','year'], axis=1)
alpha_organic = pd.concat([alpha_organic, overage_cost_organic1], axis=1)
alpha_organic['alpha'] = alpha_organic.apply (lambda row: alpha(row), axis=1)
alpha_organic = alpha_organic.drop(['AveragePrice','Minprice'], axis=1)
alpha_organic['z_value'] = alpha_organic.apply (lambda row: z_value(row), axis=1)

# alpha and z values conventional
overage_cost_conventional1 = overage_cost_conventional.drop(['year','region'], axis=1) 
alpha_conventional = shortage_cost_conventional.drop(['occurance','year'], axis=1)
alpha_conventional = pd.concat([alpha_conventional, overage_cost_conventional1], axis=1)
alpha_conventional['alpha'] = alpha_conventional.apply (lambda row: alpha(row), axis=1)
alpha_conventional = alpha_conventional.drop(['AveragePrice','Minprice'], axis=1)
alpha_conventional['z_value'] = alpha_conventional.apply (lambda row: z_value(row), axis=1)




### 6) Data transformation  

# normal sqrt transform function for Q*
def mnsqrt(region):   
    avo4225_organic_train_dict[region]['Avo_4225_demand'] = avo4225_organic_train_dict[region]['Avo_4225_demand'].apply(np.sqrt)
    mean = avo4225_organic_train_dict[region].groupby('region').mean()
    std = avo4225_organic_train_dict[region].groupby('region').std()
    d = mean+std*alpha_organic.loc[alpha_organic['region'] == region]['z_value'].iloc[0]
    return d**2

# normal log transform function for Q*
def mnl(region):   
    avo4225_conventional_train_dict[region] = avo4225_conventional_train_dict[region].replace(0,0.01)
    avo4225_conventional_train_dict[region]['Avo_4225_demand'] = avo4225_conventional_train_dict[region]['Avo_4225_demand'].apply(np.log)
    mean = avo4225_conventional_train_dict[region].groupby('region').mean()
    #print(mean)
    std = avo4225_conventional_train_dict[region].groupby('region').std()
    #print(std)
    d = mean['Avo_4225_demand'].iloc[0]+std['Avo_4225_demand'].iloc[0]*alpha_conventional.loc[alpha_conventional['region'] == region]['z_value'].iloc[0]
    return math.exp(d)
  
# non-normal function for Q* (percentile approximation method (refer to detailed report))
def mnn(region):   
    df = avo4046_conventional_train_dict[region]
    cf = alpha_conventional.loc[alpha_conventional['region'] == region]['alpha'].iloc[0]
    q = df.Avo_4046_demand.quantile(float(cf))
    df = df[df.Avo_4046_demand < q]
    return df['Avo_4046_demand'].max()

# apply functions to all the regions to get the Q* for that region (result stored in Final_Q_star.xlsx)
  

  
  
### 7) Profit Calculation Pre-Processing 

# import final Q* values 
Qstar_data = pd.read_excel("Final_Q_star.xlsx")       

# split Q* into conventional and organic
Qstar_conventional = pysql(
    """SELECT region, Type4225_conventional AS Avo_4225_demand,Type4046_conventional 
    AS Avo_4046_demand, Type4770_conventional AS Avo_4770_demand 
    FROM Qstar_data GROUP BY region""")

Qstar_organic = pysql(
    """SELECT region, Type4225_organic AS Avo_4225_demand,Type4046_organic 
    AS Avo_4046_demand,Type4770_organic AS Avo_4770_demand 
    FROM Qstar_data GROUP BY region""")

# clean overage and shortage cost
overage_cost_conventional = overage_cost_conventional.drop(columns = "year")
overage_cost_conventional.rename(columns = {"Minprice":"AveragePrice"}, inplace = True)
overage_cost_organic = overage_cost_organic.drop(columns = "year")
overage_cost_organic.rename(columns = {"Minprice":"AveragePrice"}, inplace = True)

shortage_cost_conventional = shortage_cost_conventional.drop(columns = ["occurance","year"])
shortage_cost_organic = shortage_cost_organic.drop(columns = ["occurance","year"])

# set up profits data for conventional and organic
profits_calculations_conventional = pysql(
    """SELECT avo_conventional_test.Previous_week_demand_date, avo_conventional_test.Avo_4225_demand,avo_conventional_test.Avo_4046_demand, avo_conventional_test.Avo_4770_demand, overage_cost_conventional.AveragePrice 
    AS cost_price, selling_price_conventional.AveragePrice as selling_price,avo_conventional_test.region 
    FROM avo_conventional_test 
    INNER JOIN overage_cost_conventional 
    ON overage_cost_conventional.region = avo_conventional_test.region 
    INNER JOIN selling_price_conventional on selling_price_conventional.region = avo_conventional_test.region 
    GROUP BY avo_conventional_test.region, avo_conventional_test.Previous_week_demand_date 
    ORDER BY avo_conventional_test.region, avo_conventional_test.Previous_week_demand_date""")

profits_calculations_organic = pysql(
    """SELECT avo_organic_test.Previous_week_demand_date, avo_organic_test.Avo_4225_demand, avo_organic_test.Avo_4046_demand, avo_organic_test.Avo_4770_demand, overage_cost_organic.AveragePrice 
    AS cost_price,selling_price_organic.AveragePrice as selling_price,avo_organic_test.region 
    FROM avo_organic_test INNER JOIN overage_cost_organic ON overage_cost_organic.region = avo_organic_test.region 
    INNER JOIN selling_price_organic on selling_price_organic.region = avo_organic_test.region 
    GROUP BY avo_organic_test.region, avo_organic_test.Previous_week_demand_date 
    ORDER BY avo_organic_test.region, avo_organic_test.Previous_week_demand_date""")

from itertools import cycle

# set up weekly profits data for conventional and organic
weeks = cycle([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
profits_calculations_conventional["week"] = [next(weeks) for week in range(len(profits_calculations_conventional))]
profits_calculations_organic["week"] = [next(weeks) for week in range(len(profits_calculations_conventional))]

# set up regions
cities = []
for i in list(range(45)):
    cities.append(Qstar_conventional.at[i,"region"])

# set up profit comparison dictionary, avocado types and weeks
profit_dict = {}
avo_types = ["4225", "4046", "4770"]
week = list(range(1,13))



    
### 8) Profits Calculation - Individual Regions (No Pooling) 

# formula to get weekly profits for individual regions 
def weekly_indiv_states_profits(df,Qstar_,avotypes,city,weeknum):
    sellingprice = pd.Series(df.loc[(df['region'] == city) & (df['week'] == weeknum)]['selling_price']).values[0]
    sales = pd.Series(df.loc[(df['region'] == city) & (df['week'] == weeknum)]["Avo_{}_demand".format(avotypes)]).values[0]
    revenue = min(Qstar_,sales)*sellingprice
    costprice = pd.Series(df.loc[(df['region'] == city) & (df['week'] == weeknum)]["cost_price"]).values[0]
    costs = Qstar_*costprice
    profit = revenue - costs
    return profit

profits = 0
counter = 0

# calculate total profits summed across different regions, avocado types and weeks in the testing set 
for city in cities:
    for avotypes in avo_types:
        Qstar_conv = Qstar_conventional.at[counter, "Avo_{}_demand".format(avotypes)]
        Qstar_org = Qstar_organic.at[counter, "Avo_{}_demand".format(avotypes)]

        for weeknum in week:
            profits += weekly_indiv_states_profits(profits_calculations_conventional, Qstar_conv, avotypes, city, weeknum)
            profits += weekly_indiv_states_profits(profits_calculations_organic, Qstar_org, avotypes, city, weeknum)
    counter += 1      
       
# add profits to profits comparison dictionary
profit_dict["individual"]= profits




### 9) Profits Calculation - Risk Pooling by States

# mapping regions to their state (27 states present - states with 1 or more cities)
city_states = {"California": ["LosAngeles","Sacramento","SanDiego","SanFrancisco"], 
               "Washington": ["Seattle","Spokane"], 
               "Texas":['DallasFtWorth',"Houston"], 
               "Illinois":["Chicago",'HartfordSpringfield'],
               "Ohio": ['CincinnatiDayton',"Columbus"],
               "North Carolina": ["Charlotte",'RaleighGreensboro'],
               "Florida":['MiamiFtLauderdale',"Jacksonville","Orlando","Tampa"],
               "Virginia": ['RichmondNorfolk','Roanoke'],
               "Pennsylvania": ['Philadelphia',"Pittsburgh",'HarrisburgScranton'],
               "New York": ["Albany",'BuffaloRochester',"NewYork","Syracuse"],
               "Michigan": ["Detroit","GrandRapids"],
               'Atlanta':['Atlanta'],
               'BaltimoreWashington':['BaltimoreWashington'],
               'Boise':['Boise'],
               'Boston':["Boston"],
               'Denver':['Denver'],
               'GreatLakes':['GreatLakes'],
               'Indianapolis':['Indianapolis'],
               'LasVegas':['LasVegas'],
               'Louisville':['Louisville'],
               'Nashville':['Nashville'],
               'NewOrleansMobile':['NewOrleansMobile'],
               'PhoenixTucson':['PhoenixTucson'],
               'Portland':['Portland'],
               'SouthCarolina':['SouthCarolina'],
               'StLouis':['StLouis'],
               'WestTexNewMexico':['WestTexNewMexico']
               }    

# formula for aggregating the regions Q* value
def aggregate_Qstar(Qstar_df,profit_df,cities,avotypes):
    Qstar = 0
    sellingprice = 0
    costprice = 0
    for city in cities:   # find optimal aggregate Q*
        indexval = Qstar_df.loc[Qstar_df['region'] == city].index[0]
        Qstar += Qstar_df.at[indexval, "Avo_{}_demand".format(avotypes)]
        sellingprice += pd.Series(profit_df.loc[(profit_df['region'] == city)]["selling_price"]).values[0]
        costprice += pd.Series(profit_df.loc[(profit_df['region'] == city)]["cost_price"]).values[0]
    sellingprice = sellingprice/len(cities)    # selling price here is the average across the state regions
    costprice = costprice/len(cities)          # cost price here is the average across the state regions
    return Qstar, sellingprice, costprice 

# formula to get weekly profits for the different states
def weekly_total_state_profit(df,cities,Qstar_,avotypes,sellingprice,costprice,weeknum):
    sales = 0
    for city in cities:
        sales += pd.Series(df.loc[(df['region'] == city) & (df['week'] == weeknum)]["Avo_{}_demand".format(avotypes)]).values[0]
    revenue = min(Qstar_,sales)*sellingprice
    costs = Qstar_*costprice
    profit = revenue - costs        
    return profit 

# calculate total profits summed across different states, avocado types and weeks in the testing set 
profits = 0
for avotypes in avo_types:
    for states,cities in city_states.items():   #loop over each state 
        Qstar_conv = aggregate_Qstar(Qstar_conventional,profits_calculations_conventional,cities,avotypes)[0]
        sellingprice_conv  = aggregate_Qstar(Qstar_conventional,profits_calculations_conventional,cities,avotypes)[1]
        costprice_conv  = aggregate_Qstar(Qstar_conventional,profits_calculations_conventional,cities,avotypes)[2]
        Qstar_org = aggregate_Qstar(Qstar_organic,profits_calculations_organic,cities,avotypes)[0]
        sellingprice_org  = aggregate_Qstar(Qstar_organic,profits_calculations_organic,cities,avotypes)[1]
        costprice_org  = aggregate_Qstar(Qstar_organic,profits_calculations_organic,cities,avotypes)[2]
        for weeknum in week:   #loop for 12 weeks
            profits += weekly_total_state_profit(profits_calculations_conventional,cities,Qstar_conv,avotypes,sellingprice_conv,costprice_conv,weeknum)
            profits += weekly_total_state_profit(profits_calculations_organic,cities,Qstar_org,avotypes,sellingprice_org,costprice_org,weeknum)

# add profits to profits comparison dictionary
profit_dict["By States"]= profits




### 10) Finding Optimal K Clusters (for risk pooling by distance)

### 10a) K-means Clustering Pre-Processing (and Visualization)

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
    

  
  
### 10b) K-means Clustering for different values of K

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
    import copy
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




### 10c) Determining best K based on profits and cost 

# calculating transport cost for different K
def transport_cost(k):  
    transport_cost = 0.65*distortions[k-1]*97.6   # $0.65 from transportprice.xlsx (market research), 
                                                  # 97.6 km ~ 1 unit longitude and 1 unit latitude (hypotenuse length)
    return transport_cost

# import region groupings for different K values 
result = pd.read_excel("groups.xlsx")
result = result.reset_index().drop('index', axis=1)

# set up grouping 
def grouping(k):
    grouping = {}
    ks = result[k].unique().tolist()
    for i in ks:
        groups = []
        for j in range(len(result['region'])):
            if result[k][j] == i:
                groups.append(result['region'][j])
        grouping[str(i)] = groups 
    return grouping

# format k string 
def k_string(k):
    return "k={}_group".format(k)

# formula to get weekly profits for the different k grouped regions
def weekly_k_total_states_profit_tran(df, kcentre, Qstar_, avotypes, sellingprice, costprice, weeknum):
    sales = 0
    for city in kcentre:
        sales += pd.Series(df.loc[(df['region'] == city) & (df['week'] == weeknum)]["Avo_{}_demand".format(avotypes)]).values[0]
    revenue = min(Qstar_, sales)*sellingprice
    costs = Qstar_*costprice + transport_cost(k)
    profits = revenue - costs        
    return profits 

# formula to calculate total profits for the different k groupings
def k_calculations_tran(Qstar_df, profit_df, k, avotypes):
    profits = 0 
    kname = k_string(k)
    for keys in grouping(kname):    #loop over different regional distribution centre
        kcentre = grouping(kname)[keys]
        Qstar = aggregate_Qstar(Qstar_df, profit_df, kcentre, avotypes)[0]
        sellingprice = aggregate_Qstar(Qstar_df, profit_df, kcentre, avotypes)[1]
        costprice = aggregate_Qstar(Qstar_df, profit_df, kcentre, avotypes)[2]
        for weeknum in week:    #loop for 12 weeks
            profits += weekly_k_total_states_profit_tran(profit_df, kcentre, Qstar, avotypes, sellingprice, costprice, weeknum)
    return profits

indiv_kprofits = {}

# calculate total profits for different values of K (1-10 as the K-means Algorithm breaks when K is too large)
possible_k = list(range(1,11)) 
for k in possible_k:   
    profits = 0 
    for avotypes in avo_types:
        profits += k_calculations_tran(Qstar_conventional,profits_calculations_conventional,k,avotypes)
        profits += k_calculations_tran(Qstar_organic,profits_calculations_organic,k,avotypes)
    indiv_kprofits[k] = profits

# finding optimal K value
optimalk = 1
optimalvalue = indiv_kprofits[1]
x=[]
y=[]
for key,value in indiv_kprofits.items():
    x.append(key)
    y.append(value)
    if (value >= optimalvalue):
        optimalk = key
        optimalvalue = value 
indiv_kprofits["optimalk"] = optimalk

# plotting Profit against K
plt.plot(x,y)
plt.title("profits vs k")
plt.xlabel("k")
plt.ylabel("profits")
plt.xticks([1,2,3,4,5,6,7,8,9,10])
plt.savefig("profits vs k.jpg")
plt.show()   # best k = 9




### 10d) Visualizing clusters when K = 9

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




### 11) Profits Calculation - Risk Pooling by Distance (K-means)

# formula to get weekly profits for the k regions
def weekly_k_total_states_profit(df, kcentre, Qstar_, avotypes, sellingprice, costprice, weeknum):
    sales = 0
    revenue = 0
    costs = 0
    for city in kcentre:
        sales += pd.Series(df.loc[(df['region'] == city) & (df['week'] == weeknum)]["Avo_{}_demand".format(avotypes)]).values[0]
    revenue = min(Qstar_, sales)*sellingprice
    costs = Qstar_*costprice 
    profits = revenue - costs        
    return profits 

# formula to get total profits summed across different k regions, avocado types and weeks in the testing set 
def k_calculations(Qstar_df, profit_df, k, avotypes):
    profits = 0 
    kname = k_string(k)
    for keys in grouping(kname):   #loop over different distribution centre
        kcentre = grouping(kname)[keys]
        Qstar = aggregate_Qstar(Qstar_df, profit_df, kcentre, avotypes)[0]
        sellingprice = aggregate_Qstar(Qstar_df, profit_df, kcentre, avotypes)[1]
        costprice = aggregate_Qstar(Qstar_df, profit_df, kcentre, avotypes)[2]
        for weeknum in week:   #loop for 12 weeks
            profits += weekly_k_total_states_profit(profit_df, kcentre, Qstar, avotypes, sellingprice, costprice, weeknum)
    return profits

revenue = 0
costs = 0
kname = k_string(optimalk)

# sum profits for organic and conventional avocado types
profits = 0 
for avotypes in avo_types:
    profits += k_calculations(Qstar_conventional,profits_calculations_conventional,k,avotypes)
    profits += k_calculations(Qstar_organic,profits_calculations_organic,k,avotypes)
    
# add profits to profits comparison dictionary
profit_dict["By k"]= profits




### 12) Comparing profits from the 3 methods of distribution implemented above

final = pd.DataFrame(data=profit_dict, index=[0])

# save results 
final.to_excel("profits_final.xlsx")
