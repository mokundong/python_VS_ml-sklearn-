# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,ensemble
from sklearn.model_selection import train_test_split

def load_data_regression():
    diabetes = datasets.load_diabetes()
    return train_test_split(diabetes.data,diabetes.target,test_size=0.25,random_state=0)

def load_data_classification():
    digits = datasets.load_digits()  # 使用 scikit-learn 自带的 digits 数据集
    return train_test_split(digits.data, digits.target,test_size=0.25, random_state=0, stratify=digits.target)

def test_AdaBoostClassifier(*data):
    X_train,X_test,y_train,y_test=data
    clf=ensemble.AdaBoostClassifier(learning_rate=0.1)
    clf.fit(X_train,y_train)
    ## 绘图
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    estimators_num=len(clf.estimators_)
    X=range(1,estimators_num+1)
    ax.plot(list(X),list(clf.staged_score(X_train,y_train)),label="Traing score")
    ax.plot(list(X),list(clf.staged_score(X_test,y_test)),label="Testing score")
    ax.set_xlabel("estimator num")
    ax.set_ylabel("score")
    ax.legend(loc="best")
    ax.set_title("AdaBoostClassifier")
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data_classification()
    test_AdaBoostClassifier(X_train, X_test, y_train, y_test)
