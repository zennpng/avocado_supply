import pandas as pd
from preprocessing import avo_conventional_train, avo_conventional_test
from scipy.stats import boxcox
import matplotlib.pyplot as plt


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
    iqr = q3-q1    
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