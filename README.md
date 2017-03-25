# QuantBox
QuantBox 是一款基于深度学习的股票量化分析软件。他可以通过学习和识别历史事件中的潜在规律来预测未来的股价变动。
小盒子现在可以自动寻找近日较为强势的股票集合A（采用的是基于Geometric Brownian Motion的蒙特卡罗模拟），
然后小盒子会对这些股票（集合A）的历史数据进行机器学习，然后他将学习到的结果应用到最新的数据中来预测未来一天
集合A中的股票的股价变动。最后他会在预测结束时返回一个列表列出下一个交易日集合A中所有股票的收益率排序和预测误差。

## Mechanism
1. Find the recent strong shares
2. Analyse these shares using Machine Learning

## Way to find recent strong shares
* Monte Carlo simulation of Geometric Brownian Motion

## Factors for machine learning (added)
* past daily returns on itself with lags (o,h,l,c)
* past daily returns on related shares with lags (o,h,l,c)
* past daily close price on itself with lags
* past daily close price on related shares with lags
* past volumes with lags
* day of week with lags

## Factors for machine learning (will be added)
* day of year with lags
* holidays with lags
* technical analysis
* daily big deals
* daily realized volatility using 5m（Taylor，CQF M2L6 P55）
