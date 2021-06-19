# Pairs-Trading-Project

In this project, a select list of pairs have been chosen for pairs trading, based on their cointegrating relationship, whose spread/difference mean reverts over any give time period. 

For this pairs trading project, for a given set of cointegrating pairs, we will use Johansen's test to assess the Hedge Ratio between two stocks in a pair and place trades when the spread deviates by a prespecified threshold from the mean.


For this pairs trading project, the traditional Cointegrated Augmented Dickey Fuller (CADF) test has been modified, to include the Kendall's Tau test instead of the Augmented Dickey Fuller test to assess mean reversion / stationarity in cointegrating pairs.
