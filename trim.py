from previsuals import remove_outlier 
from preprocessing import avo_conventional_train, avo_conventional_test, avo_organic_train, avo_organic_test

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