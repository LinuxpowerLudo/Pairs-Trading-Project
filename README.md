# Pairs-Trading-Project

[![forthebadge made-with-python](https://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

[![GitHub license](https://img.shields.io/badge/License-MIT-brightgreen.svg?style=flat-square)](https://github.com/surelyourejoking/MachineLearningStocks/blob/master/LICENSE.txt)


(Credit: The approaches used in this project were inspired by the book published by Dr. Ernest P Chan - "Algorithmic Trading: Winning Strategies and their Rationale")

In this project, a select list of pairs were chosen for pairs trading backtest over the past 22 years, based on their individual stock characteristics, whose spread/difference has a tendency to mean revert over a given time period. An intial list of 30 pairs were chosen on random based on their individual market capitalisations and industries. A more focused subset of those pairs were chosen discretionarily, based on insights into company/industry dynamics and size (basically, assessing the tendency of the stocks in a pair to move in tandem). Please note, that this approach may not always lead to gains as it is exposed to personal biases. This drawback warrants further research into developing a more reliable systematic approach to selecting pairs. 

*Disclaimer: this is a purely educational project. Be aware that backtested performance may often be deceptive – trade at your own risk!*

In this project, three methods were used to assess the spread between chosen stock pairs: Johansen's Test, modified Cointegrated Augmented Dickey Fuller Test (or as I reffered to in this project, "Cointegrated Kendall's Tau test"), and Kalman Filters. 

The core buy and sell signal framework of the three approaches below are similar to that of a simple bollinger bands strategy, only we are trading the spread between two stocks in a pair:-

```txt
Long entry -> (cur Zscore < -entry Zscore)  & (prev Zscore > -entry Zscore)
Long exit -> (cur Zscore > -exit Zscore) & (prev Zscore < -exit Zscore)
Short entry -> (cur Zscore > entryZscore) & (prev Zscore < entry Zscore)
Short exit -> (cur Zscore < exit Zscore) & (prev Zscore > exit Zscore)

note: The entry and exit Zscores can be considered as hyperparameters, but in the below 
      approaches it is set to be between 1 to 1.5 (entry Zscore), and 0 (exit Zscore)
```

## Method 1: Johansen's Test:
In this approach, for a given set of pairs, Johansen's test was used to assess the Hedge Ratio between two stocks in a pair and place trades when the spread deviates by a prespecified threshold from the mean. 

1. The trading logic is that the algorithm calculates the eigen vectors and eigen values using last year's prices, and uses the corresponding eigen vector of the largest eigen value as the hedge ratio. 
2. Using this hedge ratio, it calculates the spread. 
3. With that spread, the halflife is calculated, which is used as a window measure for the rolling mean and standard deviation of the spread.
4. Lastly the Zscore is computed that assess the deviation of the spread from the mean spread as a multiple of the standard deviation.
5. Then the logic is similar to that of bollinger bands, where, if the Zscore goes above the prespecified threshold, we short the spread, and if it goes below the prespecified threshold we long the spread, on the basis that the spread would eventually mean revert. 

RESULTS:
A basic snapshot of this strategy's results over the past 22 years:

<img width="400" alt="Johansen" src="https://user-images.githubusercontent.com/30551461/122651086-344f4380-d154-11eb-9d71-3a2e41e8c1cd.png">

(_Scope for further imporvement_: The strategy assumes a constant hedge ratio for the whole of current year, based on the hedge ratio obtained from running the Johansen's test over the preceding year's prices. This can further be enhanced, by considering a more "evolving" hedge ratio from running the Johansen's test real time over a moving historical time interval. For example, for any given day in the backtest, Johansen's test would be run on the preceding 6month or 1year from that day, and then use the hedge ratio from it to identify the spread and Zscore.)  

## Method 2: "Cointegrated Kendall's Tau Test"
In this approach, the traditional Cointegrated Augmented Dickey Fuller (CADF) test has been modified, to include the Kendall's Tau test instead of the Augmented Dickey Fuller test to assess mean reversion / stationarity in cointegrating pairs.

1. The trading logic, as with the traditional CADF test pairs trading approach, Linear Regression is used to find the hedge ratio between two stocks in a pair for a specified time period of historical prices at any given time during the backtest. 
2. Using that hedge ratio, the spread is calculated.
3. Using the spread series, the halflife is calculated, and with the halflife the Zscore is calculated, which will indicate buy and sell signals, same as in that of Johansen's test. 
4. The Modification - in the traditional CADF approach, the AD Fuller test was used to assess stationarity in the spread series of a pair to get insight into the mean reverting tendency of the spread. In the modified approach, rather than to use the AD Fuller test, the Kendall's Tau test has been used to measure the trending/stationary behavior of historical spread series at any given point in time during the backtest. (More info on Kendall's Tau can be found [here](https://www.statisticshowto.com/kendalls-tau/).)
5. Using Kendall's Tau measure, if the measure indicated a trending behaviour (KT<-0.5 | KT>0.3), there would no trades placed. The threshold values '-0.5' and '0.3' can be looked at as hyperparameters that could be changed to maximise returns for any given pair, or a more systematic approach can be devised to justify the use of certain values as thresholds. In general, The Kendall's Tau measure has a maximum and a minimum of 1 and -1 respectively, if the series is trending the measure would divulge towards the max and min, and if it is not trending it would remain close to 0.

RESULTS: 
A basic snapshot of this strategy's results over the past 22 years:

<img width="400" alt="CKT" src="https://user-images.githubusercontent.com/30551461/122651460-97da7080-d156-11eb-921e-c6abe159749d.png">


## Method 3: Kalman Filters Approach
In this approach, the hedge ratio pertaining to a given pair has been calculated dynamically using Kalman Filters. The added benefit of using Kalman Filters instead of a simple Linear Regression, is that it is not fixed for a given time period. The hedge ratio dynamically evolves, by considering the new data inputs and arrives at a more realistic representation of the underpinning relationship between two stocks. Especially since they are continuously influenced by various factors and developments during any given time period. Hence it would be naive to simply label a fixed constant to represent the relationship between stocks. 

(For an intuitive and clear understanding of Kalman Filters, check out this [video](https://www.youtube.com/watch?v=mwn8xhgNpFY&t=4s))

1. The trading loging is similar to that of the above strategies. For any given day during the backtest, it calculates the dynamic hedge ratio series using the past one year historical prices from that day. 
2. With this series of hedge ratios, the spread series is calculated for the past 1year of prices from that day. 
3. With the spread series, the halflife and Zscore is calculated just as in the above strategies. 
4. If the current Zscore (or latest Zscore - if you are at any given day during the backtest) crosses a prespecified threshold of Zscore, then the trades are placed according to our core buy and sell signal framework mentioned above. 

RESULTS:
A basic snapshot of this strategy's results over the past 22 years:

<img width="400" alt="Kalman" src="https://user-images.githubusercontent.com/30551461/122651479-b3457b80-d156-11eb-9bac-ea84b94df83e.png">
