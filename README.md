### Developed By : Sri Varshan P
### Register No. : 212222240104
### Date : 

# Ex.No: 6               HOLT WINTERS METHOD


### AIM:

To create and implement Holt Winter's Method Model using python for tank losses in Russian War



### ALGORITHM:

1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions

### PROGRAM:

#### Import Neccesary Libraries
```py
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
```

#### Load the dataset

```py

data = pd.read_csv('russia_losses_equipment.csv', index_col='date', parse_dates=True)

```

#### Resample the data to a monthly frequency (beginning of the month)

```py
data = data['tank'].resample('MS').mean()

```
#### Scaling the Data using MinMaxScaler 


```py
scaler = MinMaxScaler()
data_scaled = pd.Series(scaler.fit_transform(data.values.reshape(-1, 1)).flatten(), index=data.index)
```

#### Split into training and testing sets (80% train, 20% test)

```py
train_data = data_scaled[:int(len(data_scaled) * 0.8)]
test_data = data_scaled[int(len(data_scaled) * 0.8):]
```

#### Fitting the model
```py

fitted_model_add = ExponentialSmoothing(
    train_data, trend='add', seasonal='add', seasonal_periods=12
).fit()


```

#### Forecast and evaluate

```py

test_predictions_add = fitted_model_add.forecast(len(test_data))

```

#### Evaluate performance

```py

print("MAE :", mean_absolute_error(test_data, test_predictions_add))
print("RMSE :", mean_squared_error(test_data, test_predictions_add, squared=False))

```


#### Plot predictions


```py
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='TRAIN', color='blue')
plt.plot(test_data, label='TEST', color='green')
plt.plot(test_predictions_add, label='PREDICTION', color='red')
plt.title('Train, Test, and Additive Holt-Winters Predictions')
plt.legend(loc='best')
plt.show()
```

#### Forecast future values

```py
final_model = ExponentialSmoothing(data, trend='mul', seasonal='mul', seasonal_periods=12).fit()

forecast_predictions = final_model.forecast(steps=12)
```

```py
data.plot(figsize=(12, 8), legend=True, label='Current Tank Losses')
forecast_predictions.plot(legend=True, label='Forecasted Tank Losses')
plt.title('Tank Losses Forecast')
plt.show()
```



### OUTPUT:

#### Evaluation 

![image](https://github.com/user-attachments/assets/d8aab5d2-ecc6-45db-90a4-9e61e8c45768)



#### TEST PREDICTION

![image](https://github.com/user-attachments/assets/3a344fe1-9507-486e-a2d1-529ac1518926)


#### FINAL PREDICTION

![image](https://github.com/user-attachments/assets/b5465812-50bf-4ab6-a316-466f6d66ad21)


### RESULT:

#### Thus the program run successfully based on the Holt Winters Method model.
