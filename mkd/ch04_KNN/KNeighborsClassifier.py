import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors,datasets
from sklearn.model_selection import train_test_split

def load_classification_data():
    digits = datasets.load_digits()
    X_train = digits.data
    y_train = digits.target
    return train_test_split(X_train,y_train,test_size=0.25,random_state=0,stratify=y_train)

def create_regression_data(n):
    X = 5 * np.random.rand(n,1)
    y = np.sin(X).ravel()
    y[::5] = 1 * (0.5 - np.random.rand(int(n/5)))
    return train_test_split(X,y,test_size=0.25,random_state=0)

def test_KNeighborClassifier(*data):
    X_train,X_test,y_train,y_test=data
    clf=neighbors.KNeighborsClassifier()
    clf.fit(X_train,y_train)
    print("Training Score:%f"%clf.score(X_train,y_train))
    print("Testing Score:%f"%clf.score(X_test,y_test))

def test_KNeighborClassifier_k_w(*data):
    X_train,X_test,y_train,y_test=data
    ks = np.linspace(1,y_train.size,num=100,endpoint=False,dtype='int')



if __name__ == "__main__":
    X_train,y_train,X_test,y_test = load_classification_data()
    test_KNeighborClassifier(X_train,y_train,X_test,y_test)