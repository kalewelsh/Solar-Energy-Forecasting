# Overview

* Built a long short-term memory model that predicts the next hour's solar power output based on the previous 24 hours of 
1. Solar power
2. net solar raditation at the top of the atmosphere (J/m^2)
3. solar radiation on the surface (J/m^2)
4.surface solar radiatino down (J/m^2)
5. surface thermal radiation (J/m^2)
6. temperature (K)
7. Relative humidity at 1000 mbar (%)
8. Hour of the day (0-23)
9. Nighttime (0 = daytime, 1 = nighttime)  

* The model achieved a mean squared error of .00578 vs the baseline model which achieved a mean squared error of .01487

![](/images/Last_3_Days.jpg)
![](/images/First_3_Days.jpg)

# Feature Selection and Engineering

![](/images/Capture.PNG)

Not suprisingly, there is virtually no energy output from a solar farm during periods of the nighttime.
