# Overview

* Built a long short-term memory model that predicts the next hour's solar power output based on the previous 24 hours of 
1. Solar power (MWh)
2. The hour of the day
3. Cloud coverage (%)
4. Nighttime (0 = no, 1 = yes)

* The model achieved a mean squared error of .00403 vs the baseline model which achieved a mean squared error of .01227

![](/images/Last_3_Days.jpg)
![](/images/First_3_Days.jpg)

# Feature Selection and Engineering

![](https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/first_3_days.png)

Not suprisingly, there is virtually no energy output from a solar farm during periods of the nighttime.
