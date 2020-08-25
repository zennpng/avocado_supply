import pandas as pd 
from profit_preprocessing import week, grouping, avo_types, profit_dict, Qstar_organic, Qstar_conventional, profits_calculations_conventional, profits_calculations_organic
from profit_states import aggregate_Qstar
from kmeans_best import k, k_string, optimalk


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