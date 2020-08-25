import pandas as pd 
from profit_preprocessing import cities, avo_types, week, profit_dict, profits_calculations_conventional, profits_calculations_organic, Qstar_conventional, Qstar_organic


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