import os
import math
import numpy as np
import pandas as pd
from scipy.stats import shapiro
from scipy.stats import norm
from scipy.stats import boxcox
from scipy.stats import normaltest
from scipy.stats.distributions import chi2
import statsmodels.api as sm
import matplotlib.pyplot as plt


def shapirotest(dataframe, avoType, method, regionList, alpha=0.05):
    dd = {}
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        test_stat, p = shapiro(demand)
        if p > alpha:
            result = "Looks normal at " + str(alpha) + " level, do not reject H0"
        else:
            result = "Not normal at "+ str(alpha) + " level, reject H0"
            
        dd[reg] = result
        print ("Processing...")
    
    new_df = pd.DataFrame.from_dict(dd, orient='index', columns=['Normality Test Result'])
    
    file_name = "Data Analysis/Shapiro Normality Tests/" + "Shapiro" + avoType + method
    new_df.to_csv(file_name+'.csv')        

def D_Agostino(dataframe, avoType, method, regionList, alpha=0.05):
    dd = {}
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        test_stat, p = normaltest(demand)
        if p > alpha:
            result = "Looks normal at " + str(alpha) + " level, do not reject H0"
        else:
            result = "Not normal at "+ str(alpha) + " level, reject H0"
            
        dd[reg] = result
        print ("Processing...")
    
    new_df = pd.DataFrame.from_dict(dd, orient='index', columns=['Normality Test Result'])
    
    file_name = "Data Analysis/D_Agostino Normality Tests/" + "D_Agostino" + avoType + method
    new_df.to_csv(file_name+'.csv')

# QQ plots (no trim)
def qqplotdemand(dataframe, avoType, method, regionList):
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        demandSorted = pd.DataFrame(demand)
        sm.qqplot(demandSorted, line='s', alpha= 0.3)
        plt.title(reg+" "+method+" "+avoType)
        
        directory = "Data Analysis/QQplots/"
        if not os.path.isdir(directory+method+avoType+"/"):
            os.makedirs(directory+method+avoType+"/")
        
        plt.savefig(directory+method+avoType+"/"+reg+" "+method+" "+avoType)
        plt.show()
 
def qqplotbags(dataframe, bagType, method, regionList):
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[bagType].tolist()
        demand.sort()
        demandSorted = pd.DataFrame(demand)
        sm.qqplot(demandSorted, line='s', alpha= 0.3)
        plt.title(reg+" "+method+" "+bagType)
        
        directory = "Data Analysis/BAGS/QQplotsbags/"
        if not os.path.isdir(directory+method+bagType+"/"):
            os.makedirs(directory+method+bagType+"/")
        
        plt.savefig(directory+method+bagType+"/"+reg+" "+method+" "+bagType)
        plt.show()

# tests that trim
def shapirotesttrim(dataframe, avoType, method, regionList, alpha=0.05):
    dd = {}
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        Q1 = np.percentile(demand, 25, interpolation="midpoint")
        Q3 = np.percentile(demand, 75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        demand_trim = []
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                demand_trim.append(demand[i])
        
        test_stat, p = shapiro(demand_trim)
        if p > alpha:
            result = "Looks normal at " + str(alpha) + " level, do not reject H0"
        else:
            result = "Not normal at "+ str(alpha) + " level, reject H0"
            
        dd[reg] = result
        print ("Processing...")
    
    new_df = pd.DataFrame.from_dict(dd, orient='index', columns=['Normality Test Result'])
    
    file_name = "Data Analysis/Shapiro Normality Tests trimmed/" + "Shapiro" + avoType + method+ "trimmed"
    new_df.to_csv(file_name+'.csv')
    
    
def D_Agostinotrim(dataframe, avoType, method, regionList, alpha=0.05):
    dd = {}
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        Q1 = np.percentile(demand, 25, interpolation="midpoint")
        Q3 = np.percentile(demand, 75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        demand_trim = []
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                demand_trim.append(demand[i])
        
        test_stat, p = normaltest(demand_trim)
        if p > alpha:
            result = "Looks normal at " + str(alpha) + " level, do not reject H0"
        else:
            result = "Not normal at "+ str(alpha) + " level, reject H0"
            
        dd[reg] = result
        print ("Processing...")
    
    new_df = pd.DataFrame.from_dict(dd, orient='index', columns=['Normality Test Result'])
    
    file_name = "Data Analysis/D_Agostino Normality Tests trimmed/" + "D_Agostino" + avoType + method+ "trimmed"
    new_df.to_csv(file_name+'.csv')

# QQ plots (trimmed)
def qqplotdemandtrim(dataframe, avoType, method, regionList):
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        Q1 = np.percentile(demand, 25, interpolation="midpoint")
        Q3 = np.percentile(demand, 75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        demand_trim=[]
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                demand_trim.append(demand[i])
        
        demandSorted = pd.DataFrame(demand_trim)
        sm.qqplot(demandSorted, line='s', alpha= 0.3)
        plt.title(reg+" "+method+" "+avoType+ " trimmed")
        
        directory = "Data Analysis/QQplotstrim/"
        if not os.path.isdir(directory+method+avoType+"/"):
            os.makedirs(directory+method+avoType+"/")
        
        plt.savefig(directory+method+avoType+"/"+reg+" "+method+" "+avoType+" trimmed")
        plt.show()

def qqplotbagstrim(dataframe, bagType, method, regionList):
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[bagType].tolist()
        demand.sort()
        Q1 = np.percentile(demand, 25, interpolation="midpoint")
        Q3 = np.percentile(demand, 75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        demand_trim=[]
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                demand_trim.append(demand[i])
                
        demandSorted = pd.DataFrame(demand_trim)
        sm.qqplot(demandSorted, line='s', alpha= 0.3)
        plt.title(reg+" "+method+" "+bagType+ " trimmed")
        
        directory = "Data Analysis/BAGS/QQplotsbagstrim/"
        if not os.path.isdir(directory+method+bagType+"/"):
            os.makedirs(directory+method+bagType+"/")
        
        plt.savefig(directory+method+bagType+"/"+reg+" "+method+" "+bagType+ " trimmed")
        plt.show()

# QQ plots (trimmed and transformed)
def qqplotsqrttransform(dataframe, avoType, method, regionList):
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        Q1 = np.percentile(demand, 25, interpolation="midpoint")
        Q3 = np.percentile(demand, 75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        demand_trim=[]
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                demand_trim.append(math.sqrt(demand[i]))
        
        demandSorted = pd.DataFrame(demand_trim)
        sm.qqplot(demandSorted, line='s', alpha=0.3)
        plt.title(reg+" "+method+" "+avoType+ " trimmed sqrt")
        
        directory = "Data Analysis/QQplotssqrttransform/"
        if not os.path.isdir(directory+method+avoType+"/"):
            os.makedirs(directory+method+avoType+"/")
        
        plt.savefig(directory+method+avoType+"/"+reg+" "+method+" "+avoType+" trimmed sqrt")
        plt.show()
        
def qqplotlogtransform(dataframe, avoType, method, regionList):
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        Q1 = np.percentile(demand, 25, interpolation="midpoint")
        Q3 = np.percentile(demand, 75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        demand_trim=[]
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                if demand[i] == 0:      # check for log0 math error
                    val = 0.00000001
                    demand_trim.append(math.log(val))
                else:
                    demand_trim.append(math.log(demand[i]))
        
        demandSorted = pd.DataFrame(demand_trim)
        sm.qqplot(demandSorted, line='s', alpha= 0.3)
        plt.title(reg+" "+method+" "+avoType+ " trimmed log")
        
        directory = "Data Analysis/QQplotslogtransform/"
        if not os.path.isdir(directory+method+avoType+"/"):
            os.makedirs(directory+method+avoType+"/")
        
        plt.savefig(directory+method+avoType+"/"+reg+" "+method+" "+avoType+" trimmed log")
        plt.show()

# tests that trim and transform
def shapirotesttrimsqrt(dataframe, avoType, method, regionList, alpha=0.05):
    dd = {}
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        Q1 = np.percentile(demand, 25, interpolation="midpoint")
        Q3 = np.percentile(demand, 75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        demand_trim=[]
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                demand_trim.append(math.sqrt(demand[i]))
        
        test_stat, p = shapiro(demand_trim)
        if p > alpha:
            result = "Looks normal at " + str(alpha) + " level, do not reject H0"
        else:
            result = "Not normal at "+ str(alpha) + " level, reject H0"
            
        dd[reg] = result
    
    new_df = pd.DataFrame.from_dict(dd, orient='index', columns=['Normality Test Result'])
    
    file_name = "Data Analysis/Shapiro Normality Tests trimmed sqrt/" + "Shapiro" + avoType + method+ "trimmed sqrt"
    new_df.to_csv(file_name+'.csv')
    
def shapirotesttrimlog(dataframe, avoType, method, regionList, alpha=0.05):
    dd = {}
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        Q1 = np.percentile(demand, 25, interpolation="midpoint")
        Q3 = np.percentile(demand, 75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        demand_trim=[]
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                if demand[i] == 0:   # check for log0 math error
                    val = 0.00000001
                    demand_trim.append(math.log(val))
                else:
                    demand_trim.append(math.log(demand[i]))
        
        test_stat, p = shapiro(demand_trim)
        if p > alpha:
            result = "Looks normal at " + str(alpha) + " level, do not reject H0"
        else:
            result = "Not normal at "+ str(alpha) + " level, reject H0"
            
        dd[reg] = result
    
    new_df = pd.DataFrame.from_dict(dd, orient='index', columns=['Normality Test Result'])
    
    file_name = "Data Analysis/Shapiro Normality Tests trimmed log/" + "Shapiro" + avoType + method+ "trimmed log"
    new_df.to_csv(file_name+'.csv')
    
def D_Agostinotrimsqrt(dataframe, avoType, method, regionList, alpha=0.05):
    dd = {}
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        Q1 = np.percentile(demand, 25, interpolation="midpoint")
        Q3 = np.percentile(demand, 75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        demand_trim=[]
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                demand_trim.append(math.sqrt(demand[i]))
        
        test_stat, p = normaltest(demand_trim)
        if p > alpha:
            result = "Looks normal at " + str(alpha) + " level, do not reject H0"
        else:
            result = "Not normal at "+ str(alpha) + " level, reject H0"
            
        dd[reg] = result
    
    new_df = pd.DataFrame.from_dict(dd, orient='index', columns=['Normality Test Result'])
    
    file_name = "Data Analysis/D_Agostino Normality Tests trimmed sqrt/" + "D_Agostino" + avoType + method+ "trimmed sqrt"
    new_df.to_csv(file_name+'.csv')
    
def D_Agostinotrimlog(dataframe, avoType, method, regionList, alpha=0.05):
    dd = {}
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        Q1 = np.percentile(demand, 25, interpolation="midpoint")
        Q3 = np.percentile(demand, 75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        demand_trim=[]
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                if demand[i] == 0:     # check for log0 math error
                    val = 0.00000001
                    demand_trim.append(math.log(val))
                else:
                    demand_trim.append(math.log(demand[i]))
        
        test_stat, p = normaltest(demand_trim)
        if p > alpha:
            result = "Looks normal at " + str(alpha) + " level, do not reject H0"
        else:
            result = "Not normal at "+ str(alpha) + " level, reject H0"
            
        dd[reg] = result
    
    new_df = pd.DataFrame.from_dict(dd, orient='index', columns=['Normality Test Result'])
    
    file_name = "Data Analysis/D_Agostino Normality Tests trimmed log/" + "D_Agostino" + avoType + method+ "trimmed log"
    new_df.to_csv(file_name+'.csv')

def chi_gofsqrt(dataframe, avoType, method, regionList, alpha=0.05):
    dd = {}
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        Q1 = np.percentile(demand, 25, interpolation="midpoint")
        Q3 = np.percentile(demand, 75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        data = []
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                data.append(math.sqrt(demand[i]))
        
        n = len(data)
        bins = 2*(n**(2/5))
        bin_width = (max(data) - min(data)) / bins
        
        boundarylist=[]
        boundval = min(data)
        while boundval < max(data):
            boundarylist.append(boundval)
            boundval += bin_width
            # remember to account for missing last boundary
            
        lowerbound = boundarylist[:]
        upperbound = boundarylist[1:]
        
        obs_freq = []
        index = 0
        for b in range(len(upperbound)):
            count = 0    
            while data[index] < upperbound[b]:
                if lowerbound[b] <= data[index] < upperbound[b]:
                    count += 1
                    index += 1
            obs_freq.append(count)
    
        obs_freq.append(len(data) - index)   # count and add remaining data points
        upperbound.append(max(data))   # add last boundary  
        
        # check bin size
        ind = 0
        stop_value = len(obs_freq) - 1
        while ind < stop_value:
            if obs_freq[ind] < 5:
                del lowerbound[ind+1]
                del upperbound[ind]
                valtoadd = obs_freq.pop(ind)
                obs_freq[ind] += valtoadd    # add to next bin
                stop_value = len(obs_freq) - 1    # update because freqlist becomes shorter
            else:
                ind += 1

        # check last bin
        if obs_freq[ind] < 5:
            del lowerbound[ind]
            del upperbound[ind - 1]
            valtoadd = obs_freq.pop(ind)
            obs_freq[ind - 1] += valtoadd     # add to previous bin
        
        meanlist = []
        for b in range(len(upperbound)):
            if len(upperbound) != 1:
                ave = (lowerbound[b] + upperbound[b])/2
                meanlist.append(ave * obs_freq[b])
            else:
                meanlist.append(0)
            
        mean = sum(meanlist) / sum(obs_freq)
        
        varlist = []
        for b in range(len(upperbound)):
            if len(upperbound) != 1:
                ave = (lowerbound[b] + upperbound[b])/2
                sq = (ave - mean)**2
                varlist.append(sq * obs_freq[b])
            else:
                varlist.append(0)
            
        variance = sum(varlist) / (sum(obs_freq) - 1)
        
        expt_freq = []
        for b in range(len(upperbound)):
            if len(upperbound) != 1:
                prob = norm(mean,variance**0.5).cdf(upperbound[b])-norm(mean,variance**0.5).cdf(lowerbound[b])
                expt_freq.append(prob*sum(obs_freq))
            else:
                expt_freq.append(0)
            
        chi_list = []
        for i in range(len(obs_freq)):
            if len(obs_freq) != 1:
                chisq = ( obs_freq[i] - expt_freq[i] )**2
                chi_list.append(chisq/expt_freq[i])
            else:
                chi_list.append(((obs_freq[0] - expt_freq[0])**2)/1)  
        test_stat = sum(chi_list)
        crit_val = chi2.ppf(1-alpha, len(obs_freq)-3)
        
        if min(obs_freq) < 5:
            print ("Decrease number of bins, bin count less than 5")
        else:
            print ("Done")
            
        if test_stat >= crit_val:
            result = "Not normal at "+ str(alpha) + " level, reject H0"
        else:
            result = "Looks normal at " + str(alpha) + " level, do not reject H0"

        dd[reg] = result   
    new_df = pd.DataFrame.from_dict(dd, orient='index', columns=['Normality Test Result'])
    file_name = "Data Analysis/Chi2 trimmed sqrt/" + "Chi2" + avoType + method+ "trimmed sqrt"
    new_df.to_csv(file_name+'.csv')
    
def chi_goflog(dataframe, avoType, method, regionList, alpha=0.05, bins=10):
    dd = {}
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        Q1 = np.percentile(demand, 25, interpolation="midpoint")
        Q3 = np.percentile(demand, 75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        data=[]
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                if demand[i] == 0:     # check for log0 math error
                    val = 0.00000001
                    data.append(math.log(val))
                else:
                    data.append(math.log(demand[i]))
        
 
        n = len(data)
        bins = 2*(n**(2/5))
        bin_width = (max(data) - min(data))/bins

        boundarylist=[]
        boundval = min(data)
        while boundval < max(data):
            boundarylist.append(boundval)
            boundval += bin_width    # remember to account for missing last boundary
            
        lowerbound = boundarylist[:]
        upperbound = boundarylist[1:]
        
        obs_freq = []
        index = 0
        for b in range(len(upperbound)):
            count = 0    
            while data[index] < upperbound[b]:
                if lowerbound[b] <= data[index] < upperbound[b]:
                    count += 1
                    index += 1
            obs_freq.append(count)
    
        obs_freq.append(len(data) - index)   # count and add remaining data points
        upperbound.append(max(data))   # add last boundary  
        
        # check bin size
        ind = 0
        stop_value = len(obs_freq) - 1
        while ind < stop_value:
            if obs_freq[ind] < 5:
                del lowerbound[ind + 1]
                del upperbound[ind]
                valtoadd = obs_freq.pop(ind)
                obs_freq[ind] += valtoadd
                stop_value = len(obs_freq) - 1
            else:
                ind += 1

        # check last bin
        if obs_freq[ind] < 5:
            del lowerbound[ind]
            del upperbound[ind - 1]
            valtoadd = obs_freq.pop(ind)
            obs_freq[ind-1] += valtoadd

        meanlist = []
        for b in range(len(upperbound)):
            if len(upperbound) != 1:
                ave = (lowerbound[b] + upperbound[b])/2
                meanlist.append(ave * obs_freq[b])
            else:
                meanlist.append(0)
            
        mean = sum(meanlist)/sum(obs_freq)
        
        varlist = []
        for b in range(len(upperbound)):
            if len(upperbound) != 1:
                ave = (lowerbound[b] + upperbound[b])/2
                sq = (ave - mean)**2
                varlist.append(sq * obs_freq[b])
            else:
                varlist.append(0)
            
        variance = sum(varlist)/(sum(obs_freq)-1)
        
        expt_freq = []
        for b in range(len(upperbound)):
            if len(upperbound) != 1:
                prob = norm(mean,variance**0.5).cdf(upperbound[b])-norm(mean,variance**0.5).cdf(lowerbound[b])
                expt_freq.append(prob*sum(obs_freq))
            else:
                expt_freq.append(0)
            
        chi_list = []
        for i in range(len(obs_freq)):
            if len(obs_freq) != 1:
                chisq = (obs_freq[i] - expt_freq[i])**2
                chi_list.append(chisq/expt_freq[i])
            else:
                chi_list.append(((obs_freq[0] - expt_freq[0])**2)/1)
            
        test_stat = sum(chi_list)
        crit_val = chi2.ppf(1-alpha, len(obs_freq)-3)
        
        if min(obs_freq) < 5:
            print("Decrease number of bins, bin count less than 5")
        else:
            print("Done")
            
        if test_stat >= crit_val:
            result = "Not normal at "+ str(alpha) + " level, reject H0"
        else:
            result = "Looks normal at " + str(alpha) + " level, do not reject H0"
   
        dd[reg] = result
    
    new_df = pd.DataFrame.from_dict(dd, orient='index', columns=['Normality Test Result'])
    file_name = "Data Analysis/Chi2 trimmed log/" + "Chi2" + avoType + method+ "trimmed log"
    new_df.to_csv(file_name+'.csv')
    
def chi_goftrim(dataframe, avoType, method, regionList, alpha=0.05):
    dd={}
    for reg in regionList:
        df = dataframe.loc[(dataframe['region'] == reg) & (dataframe['type'] == method)]
        demand = df[avoType].tolist()
        demand.sort()
        Q1 = np.percentile(demand,25, interpolation="midpoint")
        Q3 = np.percentile(demand,75, interpolation="midpoint")
        IQR = Q3-Q1
        lower_bound = Q1 - 1.5*IQR
        upper_bound = Q3 + 1.5*IQR
        data = []
        for i in range(len(demand)):
            if lower_bound <= demand[i] <= upper_bound:
                data.append((demand[i]))
        
        n = len(data)
        bins = 2*(n**(2/5))
        bin_width = (max(data) - min(data)) / bins
        
        boundarylist = []
        boundval = min(data)
        while boundval < max(data):
            boundarylist.append(boundval)
            boundval += bin_width    # remember to account for missing last boundary
            
        lowerbound = boundarylist[:]
        upperbound = boundarylist[1:]
        
        obs_freq = []
        index = 0
        for b in range(len(upperbound)):
            count = 0    
            while data[index] < upperbound[b]:
                if lowerbound[b] <= data[index] < upperbound[b]:
                    count +=1
                    index += 1
            obs_freq.append(count)
    
        obs_freq.append(len(data) - index)   # count and add remaining data points
        upperbound.append(max(data))   # add last boundary  
        
        #check bin size
        ind = 0
        stop_value = len(obs_freq) - 1
        while ind < stop_value:
            if obs_freq[ind] < 5:
                del lowerbound[ind + 1]
                del upperbound[ind]
                valtoadd = obs_freq.pop(ind)
                obs_freq[ind] += valtoadd    # add to next bin
                stop_value = len(obs_freq) - 1    # update because freqlist becomes shorter
            else:
                ind += 1

        # check last bin
        if obs_freq[ind] < 5:
            del lowerbound[ind]
            del upperbound[ind-1]
            valtoadd = obs_freq.pop(ind)
            obs_freq[ind-1] += valtoadd    # add to previous bin
        
        meanlist = []
        for b in range(len(upperbound)):
            if len(upperbound) != 1:
                ave = (lowerbound[b] + upperbound[b])/2
                meanlist.append(ave * obs_freq[b])
            else:
                meanlist.append(0)
            
        mean = sum(meanlist)/sum(obs_freq)
        
        varlist = []
        for b in range(len(upperbound)):
            if len(upperbound) != 1:
                ave = (lowerbound[b] + upperbound[b])/2
                sq = (ave - mean)**2
                varlist.append(sq * obs_freq[b])
            else:
                varlist.append(0)
            
        variance = sum(varlist)/(sum(obs_freq)-1)
        
        expt_freq = []
        for b in range(len(upperbound)):
            if len(upperbound) != 1:
                prob = norm(mean,variance**0.5).cdf(upperbound[b])-norm(mean,variance**0.5).cdf(lowerbound[b])
                expt_freq.append(prob * sum(obs_freq))
            else:
                expt_freq.append(0)
            
        chi_list = []
        for i in range(len(obs_freq)):
            if len(obs_freq) != 1:
                chisq = (obs_freq[i] - expt_freq[i])**2
                chi_list.append(chisq/expt_freq[i])
            else:
                chi_list.append(((obs_freq[0] - expt_freq[0])**2)/1)
            
        test_stat = sum(chi_list)
        crit_val = chi2.ppf(1-alpha, len(obs_freq)-3)
        
        if min(obs_freq) < 5:
            print ("Decrease number of bins, bin count less than 5")
        else:
            print ("Done")
            
        if test_stat >= crit_val:
            result = "Not normal at "+ str(alpha) + " level, reject H0"
        else:
            result = "Looks normal at " + str(alpha) + " level, do not reject H0"
            
        dd[reg] = result
    
    new_df = pd.DataFrame.from_dict(dd, orient='index', columns=['Normality Test Result'])
    file_name = "Data Analysis/Chi2 trimmed/" + "Chi2" + avoType + method+ "trimmed"
    new_df.to_csv(file_name+'.csv')

# load data file
avoTrain = pd.read_csv("avotrain.csv")
pd.options.display.max_rows = 10
pd.options.display.max_columns = 12

def extractColumn(dataframe,heading):
    ls = []
    for i in dataframe[heading]:
        if i not in ls:
            ls.append(i)
    return ls

# create Region list
regionList = extractColumn(avoTrain,"region")

# execute tests
qqplotdemand(avoTrain, "Type4046", "conventional", regionList)    # qqplotdemand example 
shapirotest(avoTrain, "Type4046", "conventional", regionList, alpha=0.01)    # Shapiro test example
D_Agostino(avoTrain, "Type4046", "conventional", regionList, alpha=0.01)    # D_Agostino test example

## etc ..............................