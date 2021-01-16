import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Function that creates training and test sets with fetures that are time laggs of the original features
def create_dataset(X,y, time_steps):
    Xs,ys = [],[]
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs).astype('float32'),np.array(ys).astype('float32')

#Function that makes scatter plots of Power output vs features
def make_feature_label_plots(feature_list):
    plot_index = 1
    plt.figure(figsize = (20,20))
    for feature in feature_list:
        plt.subplot(len(feature_list),1,plot_index)
        plot_df = energy_df.groupby(by = feature).mean().reset_index()
        plot_df.reset_index(inplace = True)
        sns.lineplot(plot_df[feature],plot_df['POWER'])
        plot_index += 1

#Baseline model for comparison. Predicts the next value in the time series based on it's current value.
def baseline_model(y_test):
    sse = .0
    for i in range(1,len(y_test)):
        sse += (float(y_test[i]) - float(y_test[i-1]))**2
    return sse/(len(y_test)-1)

#Creates a list of combinations of hyperparameters to be grid searched.
def model_configs():
    config = []
    nodes = [128]
    epochs = [25]
    time_steps = [24]
    dropout_rate = [0]
    for i in nodes:
        for j in epochs:
            for t in time_steps:
                for k in dropout_rate:
                    cfg = [i,j,t,k]
                    config.append(cfg)
    return config

#Given a training set, test set, and list of hyperparameter combinations this function compiles, fits,
#and returns the model and it's set of parameters that resulted in the lowes mse.
def compile_fit_lstm(train_df,test_df,config_list):
    parameter_list = []
    mse_list = []
    model_list = []
    history_list = []
    for i in range(len(config_list)):
        nodes, epochs, time_steps,dropout_rate = config_list[i]
    
        X_train, y_train = create_dataset(train_df,train_df['POWER'],time_steps)
        X_test, y_test = create_dataset(test_df,test_df['POWER'],time_steps)
        
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
        
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units = nodes, 
                                       input_shape = (X_train.shape[1],X_train.shape[2]),
                                       return_sequences=False,activation = tf.keras.activations.relu))
        model.add(tf.keras.layers.Dense(units = nodes))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        model.add(tf.keras.layers.Dense(1))
        
        model.compile(loss = 'mean_squared_error', optimizer = 'adam')
        
        config_list = model_configs()
        history = model.fit(X_train, y_train, epochs = epochs, batch_size = 100, validation_split = .1
                            ,callbacks = [callback])
        model_list.append(model)
        evaluation = model.evaluate(X_test,y_test)
        history_list.append(history)
        mse_list.append(evaluation)
        parameter_list.append(config_list[i])
        
        lowest_mse = min(mse_list)
        lowest_mse_index = mse_list.index(min(mse_list))
        best_model = model_list[lowest_mse_index]
        best_parameters = parameter_list[lowest_mse_index]
        best_history = history_list[lowest_mse_index]
    return lowest_mse,best_parameters,best_model,X_train,y_train,X_test,y_test,best_history

#Calculates the mean squared error.
def calculate_mse(actual,predicted):
    sse = 0
    for i in range(len(actual)):
        sse += (actual[i] - predicted[i][0])**2
    return sse/len(actual)


energy_df = pd.read_csv(r'C:/Users/Kyle/Documents/Solar_Zone_1.csv')
energy_df = energy_df[energy_df['ZONEID'] == 1]
energy_df.drop(columns = ['ZONEID.1','TIMESTAMP.1'],inplace = True)
energy_df.drop(columns = 'ZONEID',inplace = True)

#Converting Timestamp column to useful features that the model can learn from.
date_list = energy_df['TIMESTAMP'].tolist()
date_time_list = []
for date in date_list:
    date_time_list.append(datetime.datetime.strptime(str(date), '%Y%m%d %H:%M'))

energy_df['DATETIME'] = date_time_list

energy_df.drop(columns = 'TIMESTAMP',inplace = True)

energy_df['HOUR'] = pd.to_datetime(energy_df['DATETIME']).dt.hour
energy_df['DAY'] = pd.to_datetime(energy_df['DATETIME']).dt.day
energy_df['MONTH'] = pd.to_datetime(energy_df['DATETIME']).dt.month
energy_df['WEEKDAY'] = pd.to_datetime(energy_df['DATETIME']).dt.weekday

energy_df.drop(columns = 'DATETIME',inplace = True)

#Renaming features to more intuitive names
energy_df.columns = ['POWER','TCLW','TCIW','SURFACE PRESSURE','HUMIDITY','CLOUD COVER',
                     'U WIND COMPONENT','V WIND COMPONENT','TEMPERATURE','SSRD','STRD','TSR','PRECIPITATION',
                     'HOUR','DAY','MONTH','WEEKDAY']

#Creating a feature that determines whether the observation occured during night time based on the month
#and hour of the observation. 
energy_df['Nighttime'] = 0
for i in range(len(energy_df)):
    if energy_df['MONTH'][i] in range(1,3)  and energy_df['HOUR'][i] in range(10,20):
        energy_df['Nighttime'][i] = 1
    elif energy_df['MONTH'][i] == 3 and energy_df['HOUR'][i] in range(9,21):
        energy_df['Nighttime'][i] = 1
    elif energy_df['MONTH'][i] in range(4,9) and energy_df['HOUR'][i] in range(8,22):
        energy_df['Nighttime'][i] = 1  
    elif energy_df['MONTH'][i] in range(9,11) and energy_df['HOUR'][i] in range(9,21):
        energy_df['Nighttime'][i] = 1
    elif energy_df['MONTH'][i] in range(11,13) and energy_df['HOUR'][i] in range(10,20):
        energy_df['Nighttime'][i] = 1


#Selecting only certain features
energy_df = energy_df[['HOUR','POWER','Nighttime','CLOUD COVER','TCIW']]

#splitting original data into a training and testing set
n = len(energy_df)
training_length = int(n*(0.8))

train_df = energy_df.iloc[0:training_length]
test_df = energy_df.iloc[training_length:n]

#Scaling the continuous varaiables (Standardizing)
train_df_cat = train_df[['HOUR']]
train_df_cont = train_df.drop(columns = ['HOUR'])
test_df_cont = test_df.drop(columns = ['HOUR'])
test_df_cat = test_df[['HOUR']]

train_df_mean = train_df_cont.mean()
train_df_std = train_df_cont.std()

train_df_cont_scaled = (train_df_cont - train_df_mean)/train_df_std
test_df_cont_scaled = (test_df_cont - train_df_mean)/train_df_std

train_df = pd.concat(objs = [train_df_cont_scaled,train_df_cat],axis = 1)
test_df = pd.concat(objs = [test_df_cont_scaled,test_df_cat],axis = 1)

#Defining hyperparameters and training model
config_list = model_configs()
mse,parameters,model,X_train,y_train,X_test,y_test,history = compile_fit_lstm(train_df,test_df,config_list)


model.evaluate(X_test,y_test)

#plotting the test and validation set mse against the number of training epochs
plt.plot(history.history['loss'],label = 'training MSE')
plt.plot(history.history['val_loss'], label = 'validation data MSE')
plt.legend()
plt.savefig('tain_val_plot.jpg')
plt.show()


#Undoing the sclaing that was done previously so that the results are more interpretable. 
y_pred = model.predict(X_test)
y_pred_denormalized = y_pred*train_df_std['POWER'] + train_df_mean['POWER']
y_test_denormalized = y_test*train_df_std['POWER'] + train_df_mean['POWER']

#Converting all power predictions of < 0 to 0.
for i in range(len(y_pred_denormalized)):
    if y_pred_denormalized[i] < 0:
        y_pred_denormalized[i] = [0]
    
#Plotting predictions vs actual for first 3 days
plt.plot(y_pred_denormalized[24:72], label = 'Predicted Values')
plt.plot(y_test_denormalized[24:72], label = 'Actual Values')
plt.title('First 3 Days')
plt.legend()
plt.savefig('First 3 Days.jpg')
plt.show()

#Last day
plt.plot(y_pred_denormalized[-72:-1], label = 'Predicted Values')
plt.plot(y_test_denormalized[-72:-1], label = 'Actual Values')
plt.title('Last 3 Days')
plt.legend()
plt.savefig('Last 3 Days.jpg')
plt.show()

#First 1000 days
plt.plot(y_pred_denormalized[0:1000], label = 'Predicted Values')
plt.plot(y_test_denormalized[0:1000], label = 'Actual Values')
plt.title('First 1000 Days')
plt.legend()
plt.show()


#Baseline vs finished model MSE comparison
print('Baseline Model MSE')
print(baseline_model(y_test_denormalized))
print('Final Model MSE')
print(calculate_mse(y_test_denormalized,y_pred_denormalized))
    




