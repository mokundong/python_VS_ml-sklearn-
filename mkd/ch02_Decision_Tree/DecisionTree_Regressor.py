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

def test_DecisionTreeRegressor(*data):
    X_train,X_test,y_train,y_test = data
    regr = DecisionTreeRegressor()
    regr.fit(X_train,y_train)
    print('Train Score:%f' % (regr.score(X_train,y_train)))
    print('Test Score:%f' % (regr.score(X_test,y_test)))
    #plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    X = np.arange(0.0,5.0,0.01)[:,np.newaxis]
    Y = regr.predict(X)
    ax.scatter(X_train,y_train,label="train sample",c='g')
    ax.scatter(X_test,y_test,label="train sample",c='r')
    ax.plot(X,Y,label="predict_value",lineWidth=2,alpha=0.5)
    ax.set_xlabel("data")
    ax.set_ylabel("target")
    ax.set_title("Decision Tree Regerssion")
    ax.legend(framealpha = 0.5)
    plt.show()

def test_DecisionTreeRegressor_splitter(*data):
    X_train,X_test,y_train,y_test = data
    splitters=['best','random']
    for splitter in splitters:
        regr = DecisionTreeRegressor(splitter=splitter)
        regr.fit(X_train,y_train)
        print("Splitter %s" %splitter)
        print("Training Score:%f" %(regr.score(X_train,y_train)))
        print("Testing Score:%f" %(regr.score(X_test,y_test)))

def test_DecisionTreeRegressor_depth(*data,maxdepth):
    X_train,X_test,y_train,y_test = data
    depths = np.arange(1,maxdepth)
    training_scores = []
    testing_scores = []
    for depth in depths:
        regr = DecisionTreeRegressor(max_depth=depth)
        regr.fit(X_train,y_train)
        #print("Training Score:%f" %(regr.score(X_train,y_train)))
        #print("Testing Score:%f" %(regr.score(X_test,y_test)))
    #plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(depths,training_scores,label = "training score")
    ax.plot(depths,testing_scores,label = "testing score")
    ax.set_xlabel("maxdepth")
    ax.set_ylabel("score")
    ax.set_title("Decision Tree Regression")
    ax.legend(framealpha = 0.5)
    plt.show()

if __name__ == "__main__":
    X_train,X_test,y_train,y_test = creat_data(100)
    #test_DecisionTreeRegressor(X_train,X_test,y_train,y_test)
    #test_DecisionTreeRegressor_splitter(X_train,X_test,y_train,y_test)
    test_DecisionTreeRegressor_depth(X_train,X_test,y_train,y_test,maxdepth=20)