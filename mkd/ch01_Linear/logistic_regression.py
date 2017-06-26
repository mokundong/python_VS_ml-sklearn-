import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

def load_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return train_test_split(X_train,y_train,test_size=0.25,random_state=0,stratify=y_train)

def test_LogisticRegression(*data):
    X_train, X_test, y_train, y_test = data
    regr

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()