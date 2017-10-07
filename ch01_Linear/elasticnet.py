import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_data():
    diabetes = datasets.load_diabetes()
    x = diabetes.data
    y = diabetes.target
    return train_test_split(x,y,test_size = 0.25,random_state = 0)

def test_ElasticNet(*data):
    X_train, X_test, y_train, y_test = data
    regr = linear_model.ElasticNet()
    regr.fit(X_train,y_train)
    print('Coefficients:%s,intercept %.2f' % (regr.coef_, regr.intercept_))
    print('Residual sum of squares:%.2f' % np.mean((regr.predict(X_test) - y_test) ** 2))
    print('Score:%.2f' % regr.score(X_test, y_test))

def test_ElasticNet_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = np.logspace(-2,2)
    rhos = np.linspace(0.01,1)
    scores = []
    for alpha in alphas:
        for rho in rhos:
            regr = linear_model.ElasticNet(alpha=alpha,l1_ratio=rho)
            regr.fit(X_train,y_train)
            scores.append(regr.score(X_test,y_test))
    ##plot
    alphas,rhos = np.meshgrid(alphas,rhos)
    scores = np.array(scores).reshape(alphas.shape)
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(alphas,rhos,scores,rstride=1,cstride=1,cmap=cm.jet,linewidth=0,antialiased=False)
    fig.colorbar(surf,shrink=0.5,aspect=5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\rho$")
    ax.set_zlabel("score")
    ax.set_title("ElasticNet")
    plt.show()

if __name__ == "__main__":
    X_train, X_test, y_train, y_test=load_data()
    #test_ElasticNet(X_train, X_test, y_train, y_test)
    test_ElasticNet_alpha(X_train, X_test, y_train, y_test)