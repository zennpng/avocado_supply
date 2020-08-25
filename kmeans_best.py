import pandas as pd 
import matplotlib.pyplot as plt
from kmeans_preprocessing import distortions
from profit_preprocessing import week, avo_types, Qstar_conventional, Qstar_organic, profits_calculations_conventional, profits_calculations_organic
from profit_states import aggregate_Qstar


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