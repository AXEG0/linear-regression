import numpy as np
import pandas as pd

X = np.array([[4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
              [2, -4, 2, 6, 1, 3, 5, 4],
              [12, 16, 11, 9, 19, 12, 15, 13]])
y = np.array([34, 44, 45, 53, 53, 60, 65, 70])


class LinearRegression:

    def __init__(self):
        self.coefficients = 0
        self.intercept = 0

    def fit(self, X, y):
        X = np.vstack((np.ones(len(X[1])), X)).T
        Xt = X.T
        Xt_mul_X = Xt.dot(X)
        Xt_mul_X_inv = np.linalg.inv(Xt_mul_X)
        Xt_mul_y = Xt.dot(y)
        res = Xt_mul_X_inv.dot(Xt_mul_y)
        self.intercept = res[0]
        self.coefficients = np.array(res[1:])
        pass

    def predict(self, X):
        return X.T.dot(self.coefficients) + self.intercept


model = LinearRegression()
model.fit(X, y)
prediction = model.predict(X)
print(pd.DataFrame({'Real': y, 'Predicted': prediction}).round(1).to_string(index=False))

#  Real  Predicted
#    34       35.5
#    44       42.1
#    45       46.5
#    53       50.5
#    53       53.6
#    60       61.2
#    65       63.9
#    70       70.7
