from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('laundry.csv')
train_array = df.weight

model = ExponentialSmoothing(train_array, seasonal='add', seasonal_periods=7)

model_fit = model.fit()
test_predict = np.nan_to_num(model_fit.predict(start=train_array.index[-1], end=train_array.index[-1]+100))

plt.plot(test_predict)
plt.show()

