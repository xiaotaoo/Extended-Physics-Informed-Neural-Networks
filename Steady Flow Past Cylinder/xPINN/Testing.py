import numpy as np
from sklearn.model_selection import train_test_split
from pyDOE import lhs 
import matplotlib.pyplot as plt

field_ref = np.loadtxt('pvt.csv',delimiter=',',skiprows=6)
x_ref = field_ref[:,0:1]
y_ref = field_ref[:,1:2]

x_ref = (x_ref/180)*np.pi
plt.plot(x_ref, y_ref)

plt.show()