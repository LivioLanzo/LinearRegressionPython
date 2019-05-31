from LinearRegression import LinearRegression
import pandas as pd 
import numpy as np 

# load the data
data = pd.read_csv(r'datasets\MoviesSales.csv')

# get ys
ys = data[['Total Sales']].to_numpy()

# get predictors and add 1 column with 1's
xs = data[['First Year Gains', 'Total Production Cost', 'Total Promotional Cost']].to_numpy()
xs = np.hstack((np.ones((xs.shape[0],1)), xs))

# calculate the coefficients
L = LinearRegression()
L.fit(xs, ys)

print('Intercept is {0}'.format(L.intercept))
print('Coefficients are: {0}'.format(L.coefficients))
print('R^2 is: {0:4f}'.format(L.R_squared(ys, xs)))

