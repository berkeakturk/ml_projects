import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from linear_regression import LinearRegression

X1 = np.random.normal(50,10,100)
X2 = np.random.uniform(0,50,100)
X3 = np.random.uniform(0, 100, 100)
y = 2*X1 + 3*X2 + X1 * X2 + np.random.normal(10,5,100)
df = pd.DataFrame({'X1':X1, 'X2': X2, 'X3': X3, 'y':y})
X = df.drop('y',axis=1)
y = df.y

lr = LinearRegression()
lr.fit(X=X, y=y)
lr.summary(significant_threshold=0.05)
lr.plot_regression_summary()


