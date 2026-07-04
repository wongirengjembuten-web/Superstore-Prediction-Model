import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import root_mean_squared_error
import holidays
from pandas.tseries.holiday import USFederalHolidayCalendar




def cleaning():
    df = pd.read_csv("train.csv")
    df = df.dropna()
    df = df.drop_duplicates()
    df["Ship Date"] = pd.to_datetime(df["Ship Date"],format="%d/%m/%Y")
    df["Order Date"] = pd.to_datetime(df["Order Date"],format="%d/%m/%Y")

    return df

df = cleaning()

# Preprocess
Sales_per_day = df.groupby("Order Date")["Sales"].sum()
print(Sales_per_day)

rolling7 = Sales_per_day.rolling(window=7).mean()
rolling30 = Sales_per_day.rolling(window=30).mean()
seasonality = Sales_per_day - rolling30
print(seasonality)

plt.plot(Sales_per_day, alpha=0.3, label="Daily Sales (raw)")
plt.plot(rolling7, label="7-Day Rolling Average")
plt.plot(rolling30, label="30-Day Rolling Average", linewidth=2)

plt.title("Sales Over Time — Raw vs Smoothed")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.show()

cal = USFederalHolidayCalendar()
holiday = cal.holidays(start=df["Order Date"].min(),end=df["Order Date"].max())
black_f = ["2017-11-24", "2018-11-23", "2019-11-29"]

df["holiday"] = df["Order Date"].isin(holiday)
df["holiday_flag"] = df["holiday"].astype(int)

df["black_f"] = df["Order Date"].isin(pd.to_datetime(black_f))
df["black_friday"] = df["black_f"].astype(int)






df["Month label"] = df["Order Date"].dt.month
df["date label"] = df["Order Date"].dt.day_of_week
df["Year label"] = df["Order Date"].dt.year
print(df["Month label"])
print(df["date label"])
print(df["Year label"])

df = pd.merge(df,Sales_per_day,on="Order Date",how="inner")
print(df.info())


#training model
training_data = df[df["Year label"] < 2018]
test_data = df[df["Year label"] == 2018]
print(training_data)
print(len(test_data))


X_train = training_data.filter(items=["Month label","date label","Year label","holiday_flag","black_friday"])
y_train = training_data.filter(items=["Sales_y"])

X_test = test_data.filter(items=["Month label","date label","Year label","holiday_flag","black_friday"])
y_test = test_data.filter(items=["Sales_y"])

model = RandomForestRegressor(n_estimators=200,random_state=42)

model.fit(X_train,y_train)



#Evaluate
prediction = model.predict(X_test)

print(prediction[:365])
mae = mean_absolute_error(y_test, prediction)
rmse = root_mean_squared_error(y_test, prediction)

print("MAE :", mae)
print("RMSE:", rmse)



predicted_date = np.arange('2019-01-01', '2019-12-30', dtype='datetime64[D]')

prediction_data = {"Prediction Date" : predicted_date,
                 "Sales prediction" : prediction[:363]
                 }

prediction_df = pd.DataFrame(prediction_data)


print(len(predicted_date))
print(len(prediction[:363]))
print(prediction_df.info())

prediction_plot = prediction_df.groupby("Prediction Date")["Sales prediction"].sum()

rolling7_p = prediction_plot.rolling(window=7).mean()
rolling30_p= prediction_plot.rolling(window=30).mean()


\
plt.plot(prediction_plot,alpha=0.3,label="raw prediction")
plt.plot(rolling7_p, label="7-Day Rolling Average(prediction)")
plt.plot(rolling30_p, label="30-Day Rolling Average(prediction)", linewidth=2)
plt.title("Sales Prediction")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend(loc="upper right")
plt.show()









#plt.plot(Sales_per_day, alpha=0.3, label="Daily Sales (raw)")
#plt.plot(rolling7, label="7-Day Rolling Average")
#plt.plot(rolling30, label="30-Day Rolling Average", linewidth=2)

#plt.title("Sales Over Time — Raw vs Smoothed")
#plt.xlabel("Date")
#plt.ylabel("Sales")
#plt.legend()





