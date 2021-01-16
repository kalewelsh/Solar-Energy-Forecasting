# Overview

* Built a long short-term memory model that predicts the next hour's solar power output based on the previous 24 hours of 
1. Solar power (MWh)
3. Cloud coverage (%)

* The model achieved a mean squared error of .00403 vs the baseline model which achieved a mean squared error of .01227

![](https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/first_3_days.png)
![](https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/last_3_days.png)

# Feature Selection and Engineering


<img src="https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/June.png" width="600" height="300"> <img src="https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/Nov.png" width="600" height="300">


Not suprisingly, there is virtually no energy output from a solar farm during periods of the nighttime. There are clearly different nighttime and daytime hours depending on the season. For this reason, I created a categorical nighttime feature (0 = daytime, 1 = nighttime) determined by the month and and hour of the observation. 

![](https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/Cloudy%20vs%20not%20cloudy.png)
![](https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/cloud%20cover%20scatter.png)

Although the scatter plot is somewhat unclear, when divided into groupd of cloud cover < 50% and cloud cover > 50% there is a noticable difference in solar power output. 

![](https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/Solar%20heatmap.png) Other than cloud cover, there are not too many remarkable relationships between weahter variables and power output. After testing the addition of weather features through some trial and error, I was unable to produce better results than with just cloud cover. 

# The Model
* I opted to use a long short-term memeory neural network (lstm). An lstm model is a type of recurrent neural network that is capable of processing entire sequences of data and does not suffer from the vanishing gradient problem.
* In order to use an lstm model for time series predictions I had to convert the feature data into a dataframe of tensors of features for the past 24 hours and a corresponding data frame of power output to be predicted. This method is also known as the sliding window method.
* I tuned the parameters of the model through a grid search method as well as tweaking the values myself. 

![](https://github.com/kalewelsh/Solar-Energy-Forecasting/blob/main/image.png)

