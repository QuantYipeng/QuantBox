# QuantBox
QuantBox 是一款基于深度学习的股票量化分析软件。他可以通过学习和识别历史事件中的潜在规律来预测未来的股价变动。
小盒子现在可以自动寻找近日较为强势的股票集合A（采用的是基于Geometric Brownian Motion的蒙特卡罗模拟），
然后小盒子会对这些股票（集合A）的历史数据进行机器学习，然后他将学习到的结果应用到最新的数据中来预测未来一天
集合A中的股票的股价变动。最后他会在预测结束时返回一个列表列出下一个交易日集合A中所有股票的收益率排序和预测误差。

## Recommendations 
* next trading day: 300420[2017/3/28]
* history recommendations: 

## Mechanism
1. Find the recent strong shares
2. Analyse these shares using Machine Learning

## Way to find recent strong shares
* Monte Carlo simulation of Geometric Brownian Motion

## Information Sets
* (added) past daily returns on itself with lags
   will be delete
* (added) past daily returns on related shares with lags
   will be delete
* (added) past daily close price on itself with lags
* (added) past daily close price on related shares with lags
* (added) past volumes with lags
* (added) day of week with lags
* (will be added) day of year with lags (will be added)
* (will be added) holidays with lags (will be added)
* (will be added) daily realized volatility using 5m (will be added)
   
   Taylor，CQF M2L6 P55.  
   Using ts.get_k_hist(code, ktype='5').  
   The Learning Pool is the Panel Data for all trading stocks within one week.  
   One proposal is to seperate the rising returns and the failing returns
* (will be added) past daily candle bar informations

   (h-max{o,c})/c_t-1  
   (min{o,c}-l)/c_t-1  
   (c-o)/c_t-1
* (will be added) past daily candle bar informations on related shares

   (h-max{o,c})/c_t-1  
   (min{o,c}-l)/c_t-1  
   (c-o)/c_t-1
* (will be added) openning price change

   (o_t-c_t-1)/c_t-1  
   which represeting the hesitating people's minds
