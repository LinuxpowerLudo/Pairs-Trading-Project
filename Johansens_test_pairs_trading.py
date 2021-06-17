import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
warnings.filterwarnings('ignore')

%matplotlib inline


def ind_marker(stock): #function to identify start dates for each pair's backtest
    index_marker = 0
    data = pd.read_csv('/.../ingestable_csvs/daily/{}.csv'.format(stock)) #reading stock prices data file
    prices = data.close
    for i in range(len(prices)):
        if pd.isna(prices[i]) == False:
            index_marker = i
            break
    
    return index_marker

def half_life(spread): #function to calculate the halflfie of a mean reverting time series
    spread_lag = spread.shift(1)
    spread_lag.iloc[0] = spread_lag.iloc[1]
    
    spread_ret = spread - spread_lag
    spread_ret.iloc[0] = spread_ret.iloc[1]
    
    spread_lag2 = sm.add_constant(spread_lag)
     
    model = sm.OLS(spread_ret,spread_lag2)
    res = model.fit()
    halflife = int(round(-np.log(2) / res.params[1],0))
 
    if halflife <= 0:
        halflife = 1
    return halflife

def kt(spread): #Kendall’s Tau function to check if a time series is trending - used in the backtest for stoploss
    C, D = 0 , 0
    for i in range(len(spread)):
        base = spread.iloc[i]
        for a in range(i+1, len(spread)):
            if base <= spread.iloc[a]:
                C += 1
            else:
                D += 1
                
    kt = (C-D)/(C+D)
    return round(kt,2)


#manual backtest
def johansens_backtest(sym1, sym2):
    stock1 = pd.read_csv('/.../ingestable_csvs/daily/{}.csv'.format(sym1))
    stock2 = pd.read_csv('/.../ingestable_csvs/daily/{}.csv'.format(sym2))
    s1 = stock1.close
    s2 = stock2.close
    
    yr_ret = []
    yr_sharpe = []
    
    #define the time_line
    max_start = max(ind_marker(sym1), ind_marker(sym2))
    time_line = [*range(max_start+240,len(s1), 240)]
    
    #iterating each year and running the backtest
    main_df = pd.DataFrame(columns = ['x', 'y', 'spread', 'hr', 'zScore', 'long entry', 'long exit',
                                      'num units long', 'short entry', 'short exit', 'num units short',
                                      'numUnits'])
    count = 0
    for i in time_line:
        if i == time_line[-1]:
            start = i
            end = len(s1) - 1
        else: 
            start = i
            end = i+240
        tx, ty = s1[start-240:start], s2[start-240:start] #to train the hr
        x, y = s1[start-240:end], s2[start-240:end] #to calculate the adf, avg and zscores
        ax, ay = s1[start:end], s2[start:end] #the actual df to use for the trading and return calc
        df1 = pd.DataFrame({'x':ax, 'y':ay})
        df1 = df1.reset_index(drop = True)


        df = pd.DataFrame({'x':tx, 'y': ty})
        res = coint_johansen(df, 0, 1)

        hr = (res.evec[:,0]/res.evec[:,0][1])[0]
        test_spread = ((x*hr) + y) #used for adf, avg, and zscores
        spread = (ax*hr) + ay
        df1['spread'] = spread.reset_index(drop=True)

        hf = []
        hf = half_life(tx*hr + ty)

        m = test_spread.rolling(window = hf).mean() 
        m = m.loc[start:].reset_index(drop=True)
        s = test_spread.rolling(window = hf).std()
        s = s.loc[start:].reset_index(drop = True)
        
        zscore = (df1['spread'] - m)/s


        df1['hr'] = hr
        df1['zScore'] = zscore


        #####################################
        entryZscore = 1.5
        exitZscore = 0
        #stopZscore = 2.5

        # Set up num units long             
        df1['long entry'] = ((df1.zScore < - entryZscore) & ( df1.zScore.shift(1) > - entryZscore))
        df1['long exit'] = ((df1.zScore > - exitZscore) & (df1.zScore.shift(1) < - exitZscore)) 
        df1['num units long'] = np.nan 
        df1.loc[df1['long entry'],'num units long'] = 1 
        df1.loc[df1['long exit'],'num units long'] = 0
        df1['num units long'][0] = 0 
        df1['num units long'] = df1['num units long'].fillna(method='pad')

        # Set up num units short 
        df1['short entry'] = ((df1.zScore >  entryZscore) & ( df1.zScore.shift(1) < entryZscore))
        df1['short exit'] = ((df1.zScore < exitZscore) & (df1.zScore.shift(1) > exitZscore))
        df1.loc[df1['short entry'],'num units short'] = -1
        df1.loc[df1['short exit'],'num units short'] = 0
        df1['num units short'][0] = 0
        df1['num units short'] = df1['num units short'].fillna(method='pad')


        df1['numUnits'] = df1['num units long'] + df1['num units short']

        ###################
        ###STOP LOSS for extreme movments in spread 
        #### OPTION 1: using 1mon mov avg of 2mon adf pval of spread 
        adf = []
        for i in range(start-20,end):
            val = ts.adfuller(test_spread.loc[i-40:i])[1]
            adf.append(val)

        adf_m = pd.Series(adf[20:])
        avg = pd.Series(adf).rolling(window = 20).mean()# create avg
        avg_m = avg.iloc[19:].reset_index(drop=True)
        df1['adf_pval'] = adf_m
        df1['avg'] = avg_m
        #df1['stop_test'] = (avg_m >= 0.5) #(remove the hash to use this method)
        #df1.loc[df1['stop_test'], 'numUnits'] = 0 #(remove the hash to use this method)

        ###STOP LOSS for extreme movments in spread 
        #### OPTION 2: using Kendal's Tau to measure trending in spread time series
        kt_test = [] 
        for i in range(start,end):
            val = kt(test_spread.loc[i-60:i])
            kt_test.append(val)
        kt_test = pd.Series(kt_test) 
        df1['kt'] = kt_test
        df1['kt_stop'] = ((kt_test <= -0.5) | (kt_test >= 0.3))
        df1.loc[df1['kt_stop'], 'numUnits'] = 0
        ###################

        df1['spread pct ch'] = (df1['spread'] - df1['spread'].shift(1)) / ((df1['x'] * abs(df1['hr'])) + df1['y'])
        df1['port rets'] = df1['spread pct ch'] * df1['numUnits'].shift(1)

        df1['cum rets'] = df1['port rets'].cumsum()
        df1['cum rets'] = df1['cum rets'] + 1

        try:
            sharpe = ((df1['port rets'].mean() / df1['port rets'].std()) * sqrt(252)) 
        except ZeroDivisionError:
            sharpe = 0.0

        #add the ret and hr to a list
        yr_ret.append(df1['cum rets'].iloc[-1])
        yr_sharpe.append(sharpe)
        main_df = pd.concat([main_df, df1])
        
        count += 1
        print('Year {} done'.format(count))
        
    main_df = main_df.reset_index(drop=True)
    port_val = (main_df['port rets'].dropna()+1).cumprod()
    avg_daily_return = main_df['port rets'].mean()
    avg_daily_std = main_df['port rets'].std()
    annualised_sharpe = (avg_daily_return/avg_daily_std) * sqrt(252)
    total_return = port_val.iloc[-1]-1
    
    #refine port_val to fit the timeline
    port_val = port_val.reset_index(drop=True)
    shift_amt = len(s1)-len(port_val)
    port_val = port_val.reindex(range(len(s1))).shift(shift_amt)
    
    return main_df, port_val,total_return,annualised_sharpe, yr_sharpe, yr_ret 

def pairs_trade(pairs, chosen_list = None):
    
    #assign variables to output
    if chosen_list == None:
        chosen_list = [*range(len(pairs))]
    else:
        chosen_lsit = chosen_list
    
    #to get the size of data / index
    s1 = pd.read_csv('/.../ingestable_csvs/daily/OMC.csv').close
    port_val_df = pd.DataFrame(columns = [list(pairs.keys())[index] for index in chosen_list], 
                               index = range(len(s1))) #create a port_df
   
    
    #loop over a list of pairs
    for i in chosen_list:
        #assign stock names to variables
        stock1 = pairs['Pair ' + str(i)][0]
        stock2 = pairs['Pair ' + str(i)][1]
        
        #run manual backtest and save output 
        res = johansens_backtest(stock1, stock2)
        
        portfolio_value = res[1]
        
        port_val_df['Pair '+str(i)] = portfolio_value #add the portfolio value to the df
        
        print('Done backtesting pair {}'.format(i))
    
    total_val = []
    for row in range(len(s1)):
        num_null = port_val_df.loc[row].isnull().sum()
        if num_null == len(chosen_list):
            alloc = 0
            total_val.append(np.nan)
        else:
            alloc = 1/(len(chosen_list) - num_null)
            port_val = (port_val_df.loc[row] * alloc).sum()
            total_val.append(port_val)
    total_port_val = pd.Series(total_val)
    avg_daily_return = ((total_port_val/total_port_val.shift(1))-1).mean()
    avg_daily_std = ((total_port_val/total_port_val.shift(1))-1).std()
    overall_sharpe = (avg_daily_return/avg_daily_std) * sqrt(252)
    overall_vol = ((total_port_val/total_port_val.shift(1))-1).std() * sqrt(252) # annualised vol of strat
    overall_return = total_port_val.iloc[-1]-1
    
    result = [overall_return, overall_vol, overall_sharpe, total_port_val]
    return result


def read_data():

	pairs_data = pd.read_csv('/.../Mean Reversion Pairs.csv') #reading the main list of pairs
	pairs_data = pairs_data[['S1 ticker', 'S2 ticker']]

	pairs = {}
	for i in range(len(pairs_data)): #creating a dictionary of pairs data
	    pairs['Pair ' + str(i)] = pairs_data.loc[i].tolist()
	    
	chosen_list = [0,1,2,4,5,6,13,15,21,23,25,26,29] #list of stock pairs with strong cointegration 

	return pairs, chosen_list 

def read_results(result):
	overall_return, overall_vol, overall_sharpe, total_port_val = result
	plt.plot(total_port_val)
	print(f'total return: {round(overall_return,3)}')
	print(f'total vol: {round(overall_vol,3)}')
	print(f'total sharpe: {round(overall_sharpe,3)}')

	return

if __name__ == "__main__":
	print("Reading Data...")
	pairs, chosen_list = read_data()
	print("Running Johansen's Backtest...")
	result = pairs_trade(pairs, chosen_list = chosen_list)
	print("*********** Backtest Results ***********")
	read_results(result)

		            
