import pandas as pd 
from profit_preprocessing import profit_dict


final = pd.DataFrame(data=profit_dict, index=[0])
final.to_excel("profits_final.xlsx")