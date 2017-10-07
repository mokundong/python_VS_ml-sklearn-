#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,linear_model,svm
from sklearn.model_selection import train_test_split

def load_data_classfication():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return train_test_split(X_train,y_train,test_size=0.25,random_state=0,stratify=y_train)

def test_LinearSVM(*data):
    X_train,X_test,y_train,y_test = data
    cls = svm.LinearSVC()
    cls.fit(X_train,y_train)
    print('Coefficient:%s,intercept:%s'%(cls.coef_,cls.intercept_))
    print('Score:%.2f'%cls.score(X_test,y_test))

def test_LinearSVC_loss(*data):#考察损失函数的影响
    X_train, X_test, y_train, y_test = data
    losses = ['hinge','squared_hinge']
    for loss in losses:
        cls = svm.LinearSVC(loss=loss)
        cls.fit(X_train,y_train)
        print('Loss:%s'%loss)
        print('Cofficients:%s,intercept:%s'%(cls.coef_,cls.intercept_))
        print('Score:%.2f'%cls.score(X_test,y_test))

def test_LinearSVC_L12(*data):#罚项形式 L1、L2
    X_train, X_test, y_train, y_test = data
    L12 = ['l1','l2']
    for p in L12:
        cls = svm.LinearSVC(penalty=p,dual=False)
        cls.fit(X_train,y_train)
        print('penalty:%s'%p)
        print('Cofficients:%s,intercept:%s' % (cls.coef_, cls.intercept_))
        print('Score:%.2f' % cls.score(X_test, y_test))

def tset_LinearSVC_C(*data):#考察罚项系数的影响 C
    X_train, X_test, y_train, y_test = data
    Cs = np.logspace(-2,1)
    train_score = []
    test_score = []
    for c in Cs:
        cls = svm.LinearSVC(C=c)
        cls.fit(X_train,y_train)
        train_score.append(cls.score(X_train,y_train))
        test_score.append(cls.score(X_test,y_test))

    ##绘图
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(Cs,train_score,label='Train Score')
    ax.plot(Cs,test_score,label='Test Score')
    ax.set_xlabel(r"C")
    ax.set_ylabel(r"score")
    ax.set_xscale("log")
    ax.set_title("LinearSVC")
    ax.legend(loc='best')
    plt.show()




if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_data_classfication()
    # test_LinearSVM(X_train, X_test, y_train, y_test)
    # test_LinearSVC_loss(X_train, X_test, y_train, y_test)
    # test_LinearSVC_L12(X_train, X_test, y_train, y_test)
    tset_LinearSVC_C(X_train, X_test, y_train, y_test)