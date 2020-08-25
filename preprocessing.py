import pandas as pd 
import pandasql as pdsql
pysql = lambda q: pdsql.sqldf(q, globals())


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