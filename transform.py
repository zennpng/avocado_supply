import math
import numpy as np
from trim import *
from alphas import alpha_conventional, alpha_organic


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

## apply functions to all the regions to get the Q* for that region (result stored in Final_Q_star.xlsx)