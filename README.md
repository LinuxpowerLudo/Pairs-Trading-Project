# Pairs-Trading-Project

In this project, a select list of pairs have been chosen for pairs trading, based on their individual stock characteristics, whose spread/difference has a tendency to mean revert over a give time period. An intial list of 30 pairs were chosen on random based on their individual market capitalisations and industries. A more focused subset of those pairs were chosen discretionarily, based on insights into company/industry dynamics and size (basically assessing the tendency of the stocks in a pair to move in tandem). Please note, that this approach may not always lead to gains, this drawback warrants further research into developing a more reliable approach to selecting pairs. 

For this pairs trading project, for a given set of cointegrating pairs, we will use Johansen's test to assess the Hedge Ratio between two stocks in a pair and place trades when the spread deviates by a prespecified threshold from the mean.


For this pairs trading project, the traditional Cointegrated Augmented Dickey Fuller (CADF) test has been modified, to include the Kendall's Tau test instead of the Augmented Dickey Fuller test to assess mean reversion / stationarity in cointegrating pairs.
