import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt

def load_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return train_test_split(X_train, y_train, test_size=0.25, random_state=0, stratify=y_train)

def creat_data(n):
    np.random.seed(0)
    X = 5 * np.random.rand(n,1)
    y = np.sin(X).ravel()
    noise_num = (int)(n/5)
    y[::5] += 3 * (0.5 - np.random.rand(noise_num))
    return train_test_split(X,y,test_size=0.25,random_state=1)

def test_DecisionTreeClassifier_criterion(*data):
    X_train,X_test,y_train,y_test = data
    criterions=['gini','entropy']
    for criterion in criterions:
        clf = DecisionTreeClassifier(criterion=criterion)
        clf.fit(X_train,y_train)
        print("criterion:%s"%criterion)
        print("Training score:%f"%(clf.score(X_train,y_train)))
        print("Testing score:%f"%(clf.score(X_test,y_test)))

def test_DecisionTreeClassifier_spliter(*data):
    X_train,X_test,y_train,y_test = data
    splitters = ['best','random']
    for splitter in splitters:
        clf = DecisionTreeClassifier(splitter=splitter)
        clf.fit(X_train,y_train)
        print("criterion:%s" % splitter)
        print("Training score:%f" % (clf.score(X_train, y_train)))
        print("Testing score:%f" % (clf.score(X_test, y_test)))

def test_DecisionTreeClassifier_depth(*data,maxdepth):
    X_train, X_test, y_train, y_test = data
    depths = np.arange(1,maxdepth)
    training_score = []
    testing_score = []
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth)
        clf.fit(X_train,y_train)
        training_score.append(clf.score(X_train,y_train))
        testing_score.append(clf.score(X_test,y_test))
    ##plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(depths,training_score,label="training score",marker='o')
    ax.plot(depths,testing_score,label="testing score",marker='*')
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Classification")
    ax.legend(framealpha=0.5,loc='best')
    plt.show()

if __name__ == "__main__":
    X_train,X_test,y_train,y_test = load_data()
    # test_DecisionTreeClassifier_criterion(X_train,X_test,y_train,y_test)
    #test_DecisionTreeClassifier_spliter(X_train,X_test,y_train,y_test)
    #X_train, X_test, y_train, y_test = creat_data(10)
    test_DecisionTreeClassifier_depth(X_train,X_test,y_train,y_test,maxdepth=100)