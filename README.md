# Pairs-Trading-Project

(The approaches in the project were inspired by the book published by Dr. Ernest P Chan - "Algorithmic Trading")
In this project, a select list of pairs were chosen for pairs trading, based on their individual stock characteristics, whose spread/difference has a tendency to mean revert over a give time period. An intial list of 30 pairs were chosen on random based on their individual market capitalisations and industries. A more focused subset of those pairs were chosen discretionarily, based on insights into company/industry dynamics and size (basically, assessing the tendency of the stocks in a pair to move in tandem). Please note, that this approach may not always lead to gains as it is exposed to personal biases. This drawback warrants further research into developing a more reliable systematic approach to selecting pairs. 

In this project, three methods were used to assess the spread between chosen stock pairs: Johansen's Test, modified Cointegrated Augmented Dickey Fuller Test (or as I reffered to in this project as "Cointegrated Kendall's Tau test", and Kalman Filters. 

## Method 1: Johansen's Test:
In this approach, for a given set of pairs, Johansen's test was used to assess the Hedge Ratio between two stocks in a pair and place trades when the spread deviates by a prespecified threshold from the mean. 

1. The trading logic is that the algorithm calculates the eigen vectors and eigen values using last year's prices, and uses the eigen vector with correspoding largest eigen value as the hedge ratio. 
2. Using this hedge ratio, it calculates the spread. 
3. With that spread, the halflife is calculated, which is used as a window measure for the rolling mean and standard deviation of the spread.
4. Lastly the Zscore is computed that assess the deviation of the spread from the mean spread as a multiple of the standard deviation.
5. Then the logic is similar to that of bollinger bands, where, if the Zscore goes above the prespecified threshold, we short the spread, and if it goes below the prespecified threshold we long the spread, on the basis that the spread would eventually mean revert. 

RESULTS:
A basic snapshot of this strategy's results are as follows:


## Method 2: "Cointegrated Kendall's Tau Test"
For this pairs trading project, the traditional Cointegrated Augmented Dickey Fuller (CADF) test has been modified, to include the Kendall's Tau test instead of the Augmented Dickey Fuller test to assess mean reversion / stationarity in cointegrating pairs.

1. The trading logic, as with the traditional CADF test pairs trading approach, Linear Regression is used to find the hedge ratio between two stocks in a pair for a specified time period of historical prices at any given time during the backtest. 
2. Using that hedge ratio, the spread is calculated.
3. Using the spread series, the halflife is calculated, and with the halflife the Zscore is calculated, which will indicate buy and sell signals, same as in that of Johansen's test. 
4. The Modification - in the traditional CADF approach, the AD Fuller test was used to assess stationarity in the spread series of a pair to get insight into the mean reverting tendency of the spread. In the modified approach, rather than to use the AD Fuller test, the Kendall's Tau test has been used to measure the trending/stationary behavior of historical spread series at any given point in time during the backtest. (More info on Kendall's Tau can be found [here](https://www.statisticshowto.com/kendalls-tau/).)
5. Using Kendall's Tau measure, if the measure indicated a trending behaviour (KT<-0.5 | KT>0.3), there would no trades placed. The threshold values '-0.5' and '0.3' can be looked at as hyperparameters that could be changed to maximise returns for any given pair, or a more systematic approach can be devised to justify the use of certain values as thresholds. In general, The Kendall's Tau measure has a maximum and a minimum of 1 and -1 respectively, if the series is trending the measure would divulge towards the max and min, and if it is not trending it would remain close to 0.

RESULTS: 
A basic snapshot of this strategy's results are as follows:


