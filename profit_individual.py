import pandas as pd 
from profit_preprocessing import cities, avo_types, week, profit_dict, profits_calculations_conventional, profits_calculations_organic, Qstar_conventional, Qstar_organic


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