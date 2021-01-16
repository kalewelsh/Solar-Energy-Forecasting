# Overview

* Built a long short-term memory model that predicts the next hour's solar power output based on the previous 24 hours of 
1. Solar power
2. The hour of the day
3. Cloud coverage

* The model achieved a mean squared error of .00403 vs the baseline model which achieved a mean squared error of .01227

![](/images/Last_3_Days.jpg)
![](/images/First_3_Days.jpg)

# Feature Selection and Engineering

![](/images/Capture.PNG)

Not suprisingly, there is virtually no energy output from a solar farm during periods of the nighttime.
