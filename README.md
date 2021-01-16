# Overview

* Built a long short-term memory model that predicts the next hour's solar power output based on the previous 24 hours of 
1. Solar power (MWh)
3. Cloud coverage (%)

* The model achieved a mean squared error of .00403 vs the baseline model which achieved a mean squared error of .01227

![](https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/first_3_days.png)
![](https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/last_3_days.png)

# Feature Selection and Engineering


![](https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/June.png =10x10)
![](https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/Nov.png)
Not suprisingly, there is virtually no energy output from a solar farm during periods of the nighttime.
