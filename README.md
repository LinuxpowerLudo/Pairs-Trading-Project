# Pairs-Trading-Project

In this project, a select list of pairs were chosen for pairs trading, based on their individual stock characteristics, whose spread/difference has a tendency to mean revert over a give time period. An intial list of 30 pairs were chosen on random based on their individual market capitalisations and industries. A more focused subset of those pairs were chosen discretionarily, based on insights into company/industry dynamics and size (basically, assessing the tendency of the stocks in a pair to move in tandem). Please note, that this approach may not always lead to gains as it is exposed to personal biases. This drawback warrants further research into developing a more reliable systematic approach to selecting pairs. 

In this project, three methods were used to assess the spread between chosen stock pairs: Johansen's Test, modified Cointegrated Augmented Dickey Fuller Test (or as I reffered to in this project as "Cointegrated Kendall's Tau test", and Kalman Filters. 

##Method 1: Johansen's Test:
In this approach, for a given set of pairs, Johansen's test was used to assess the Hedge Ratio between two stocks in a pair and place trades when the spread deviates by a prespecified threshold from the mean. 

1. The trading logic is that the algorithm calculates the eigen vectors and eigen values using last year's prices, and uses the eigen vector with correspoding largest eigen value as the hedge ratio. 
2. Using this hedge ratio, it calculates the spread. 
3. With that spread, the halflife is calculated, which is used as a window measure for the rolling mean and standard deviation of the spread.
4. Lastly the Zscore is computed that assess the deviation of the spread from the mean spread as a multiple of the standard deviation.
5. Then the logic is similar to that of bollinger bands, where, if the Zscore goes above the prespecified threshold, we short the spread, and if it goes below the prespecified thresholf we long the spread, on the basis that the spread would eventually mean revert. 

RESULTS:
A basic snapshot of this strategy's results are as follows:



For this pairs trading project, the traditional Cointegrated Augmented Dickey Fuller (CADF) test has been modified, to include the Kendall's Tau test instead of the Augmented Dickey Fuller test to assess mean reversion / stationarity in cointegrating pairs.
