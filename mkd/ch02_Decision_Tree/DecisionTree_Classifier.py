import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

def creat_data(n):
    np.random.seed(0)
    X = 5 * np.random.rand(n,1)
    y = np.sin(X).ravel()
    noise_num = (int)(n/5)
    y[::5] += 3 * (0.5 - np.random.rand(noise_num))
    return train_test_split(X,y,test_size=0.25,random_state=1)



if __name__ == "__main__":
    X_train,X_test,y_train,y_test = creat_data(100)
