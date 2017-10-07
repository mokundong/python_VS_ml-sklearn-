import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis
from sklearn.model_selection import train_test_split

def load_data():
    iris = datasets.load_iris()
    X_train = iris.data
    y_train = iris.target
    return train_test_split(X_train,y_train,test_size=0.25,random_state=0,stratify=y_train)

def test_linearDiscriminantAnalysis(*data):
    X_train,X_test,y_train,y_test = data
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X_train,y_train)
    print('Coefficients:%s,intercept %s' % (lda.coef_, lda.intercept_))
    print('Score:%.2f' % lda.score(X_test, y_test))

def plot_LDA(converted_X,y):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    colors = 'rgb'
    markers = 'o*s'
    for target,colors,markers in zip([0,1,2],colors,markers):
        pos = (y == target).ravel()
        X = converted_X[pos,:]
        ax.scatter(X[:,0],X[:,1],X[:,2],color=colors,marker=markers,label='Label %d'%target)
    ax.legend(loc = 'best')
    fig.suptitle('Iris After LDA')
    plt.show()

def test_linearDiscriminantAnalysis_solver(*data):
    X_train,X_test,y_train,y_test = data
    solvers = ['svd','lsqr','eigen']
    for solver in solvers:
        if(solver=='svd'):
            lda = discriminant_analysis.LinearDiscriminantAnalysis(solver = solver)
        else:
            lda = discriminant_analysis.LinearDiscriminantAnalysis(solver = solver,shrinkage=None)
        lda.fit(X_train,y_train)
        print('Score at solve=%s:%.2f'%(solver,lda.score(X_test,y_test)))

def test_linearDiscriminantAnalysis_shrinkage(*data):
    X_train,X_test,y_train,y_test = data
    shrinkages = np.linspace(0.0,1.0,num=20)
    scores = []
    for shrinkage in shrinkages:
        lda = discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr',shrinkage=shrinkage)
        lda.fit(X_train,y_train)
        scores.append(lda.score(X_test,y_test))
    ##plot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(shrinkages,scores)
    ax.set_xlabel(r'shrinkage')
    ax.set_ylabel(r'score')
    ax.set_ylim(0,1.05)
    ax.set_title("LinearDiscriminanAnalysis")
    plt.show()


if __name__ == "__main__":
    X_train,X_test,y_train,y_test = load_data()
    """
    X = np.vstack((X_train,X_test))
    Y = np.vstack((y_train.reshape(y_train.size,1),y_test.reshape(y_test.size,1)))
    lda = discriminant_analysis.LinearDiscriminantAnalysis()
    lda.fit(X,Y)
    converted_X = np.dot(X,np.transpose(lda.coef_)) + lda.intercept_
    plot_LDA(converted_X,Y)
    """
    #test_linearDiscriminantAnalysis_solver(X_train,X_test,y_train,y_test)

    test_linearDiscriminantAnalysis_shrinkage(X_train,X_test,y_train,y_test)