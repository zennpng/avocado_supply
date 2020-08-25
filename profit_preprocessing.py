import pandas as pd
import pandasql as pdsql
from itertools import cycle
pysql = lambda q: pdsql.sqldf(q, globals())
from alphas import overage_cost_conventional, overage_cost_organic, shortage_cost_conventional, shortage_cost_organic


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