from preprocessing import avo_conventional_train, avo_conventional_test, avo_organic_train, avo_organic_test
import pandas as pd
import pandasql as pdsql
pysql = lambda q: pdsql.sqldf(q, globals())
from scipy.stats import norm


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