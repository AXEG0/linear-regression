import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

X = np.array([[4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8],
              [2, -4, 2, 6, 1, 3, 5, 4],
              [12, 16, 11, 9, 19, 12, 15, 13]])
y = np.array([34, 44, 45, 53, 53, 60, 65, 70])

model = LinearRegression().fit(X.T, y)
prediction = model.predict(X.T)
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
