# Sample for using the model
import pickle
import numpy as np

with open('model.pkl', 'rb') as op:
    model = pickle.load(op)

open_f = 1.00427
high_f = 1.00833
low_f = 1.00003
marketCap_f = 24323316.23
volume_f = 108803
year_f = 2018

x = np.array([[year_f, high_f, low_f, open_f, volume_f, marketCap_f]])

y_pred = model.predict(x)
print(y_pred)
