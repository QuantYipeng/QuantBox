# QuantBox
QuantBox 是一个聪明的机器人。他可以通过学习和识别历史事件中的潜在规律来预测未来的股价变动。
# Things the Box knows
目前小盒子已经能够看懂历史K线，找出高相关性的股票和识别简单的时间事件（比如历史上的这一天是周几）。
# Things the Box is studying
小盒子正在对多种技术分析手段进行识别和量化，同时也在学习日内盘口数据中隐藏的信息。
# Functions
小盒子现在可以自动寻找近日较为强势的股票集合A（采用的是基于Geometric Brownian Motion的蒙特卡罗模拟），
然后小盒子会对这些股票（集合A）的历史数据进行机器学习，然后他将学习到的结果应用到最新的数据中来预测未来一天
集合A中的股票的股价变动。最后他会在预测结束时返回一个列表列出下一个交易日集合A中所有股票的收益率排序和预测误差。
# Stylized
1. The distribution of returns is not normal. It's approximately symmetric and has fat tail as well as high peak.
2. There is almost no correlation between returns for different days
3. There is positive dependence between absolute returns on nearby days, and likewise for squared returns.
