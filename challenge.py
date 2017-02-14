# import dependencies
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV

# load the dataset
data = pd.read_csv('database.csv')
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Magnitude']]
data = np.array(data)
np.random.shuffle(data)
X = data[:,:4]
Y = data[:,-1]

X_train = X[:int(len(X)*0.7),:]
Y_train = Y[:int(len(Y)*0.7)]
X_test = X[int(len(X)*0.7):,:]
Y_test = Y[int(len(Y)*0.7):]

mlp = MLPRegressor()
mlp.fit(X_train[:,2:], Y_train)
predictions = mlp.predict(X_test[:,2:])
#print predictions
print 'Mean squared error:', mean_squared_error(Y_test, predictions)

# Optimizing hyperparameter
param_dist = {'learning_rate_init': [0.001, 0.003, 0.006, 0.009]}
n_iter_search = 4
random_search = RandomizedSearchCV(mlp, param_distributions=param_dist, n_iter=n_iter_search)
random_search.fit(X_train[:,2:], Y_train)
predictions = random_search.predict(X_test[:,2:])
#print predictions
print 'Mean squared error optimizing initial learning rate:', mean_squared_error(Y_test, predictions)