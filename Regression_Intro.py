import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style    # Style the graphs
import pickle

style.use('ggplot')     # Define the style of our plot

# Retrieve the data set
df = quandl.get('WIKI/GOOGL')

# Print the head of the data-frame
print(df.head())

# Re-create the data frame to include the following columns
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# Define the percent change in prices through-out the day
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100

df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

# print(df.head())

# Rename a column
forecast_col = 'Adj. Close'

# Ensure that there isn't any missing data
df.fillna(-99999, inplace=True)

# Attempt to print out 10% percent of the data in the future
forecast_out = int(math.ceil(0.01*len(df)))

# Shift the columns negatively (up), so that each row represent data 10-days into the future
df['label'] = df[forecast_col].shift(-forecast_out)

# print(df.head())

# Create our input (i.e. features)
X = np.array(df.drop(['label'], 1))
# Scale our data (we must scale the new values)
X_scaled = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]


df.dropna(inplace=True)     # Drop the values for which there is no corresponding y
# Create our label
y = np.array(df['label'])


# Perform the train-test split
X_train, X_test, y_train, y_test = model_selection.train_test_split(X_lately, y, test_size=0.2)

# Create the classifier
classifier = LinearRegression()

# Fit the data in the classifier
'''
    !!!!!!!The fit function is synonymous with 'train' and the score function is synonymous with 'test'!!!!!!!!!!!!!
'''
classifier.fit(X_train, y_train)

# Here, we save the classifier for future reference
#with open('linear_Regression.pickle', 'wb') as f:
#    pickle.dump(classifier, f)

# Load the saved pickle file
# pickle_in = open('linear_Regression.pickle', 'rb')
# classifier = pickle.load(pickle_in)

accuracy = classifier.score(X_test, y_test)

# print(accuracy)

# Make predictions
forecast_set = classifier.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

# Get data from the last available day and plot the future
last_date = df.iloc[-1].name    # Get the last day
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day


# Populate the data-frame
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

