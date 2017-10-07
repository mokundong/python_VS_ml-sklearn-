#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets,decomposition,manifold

def load_data():
    iris = datasets.load_iris()
    return iris.data,iris.target

def test_pca(*data):
    X,y = data
    pca = decomposition.PCA(n_components=None)
    pca.fit(X)
    print('explained variance ratio: %s' %str(pca.explained_variance_ratio_))

def plot_PCA(*data):
    X,y = data
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    X_r = pca.transform(X)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    colors = ((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
              (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2))
    for label,color in zip(np.unique(y),colors):
        position=y == label
        ax.scatter(X_r[position,0],X_r[position,1],label="target = %d"%label,color=color)
    ax.set_xlabel("x[0]")
    ax.set_ylabel("y[0]")
    ax.legend(loc="best")
    ax.set_title("PCA")
    plt.show()

def test_KPCA(*data):
    X,y = data
    kernels = ['linear','poly','rbf','sigmoid']
    for kernel in kernels:
        kpca = decomposition.KernelPCA(n_components=None,kernel=kernel)
        kpca.fit(X)
        print('Kernels = %s --> lambdas: %s' %(kernel,kpca.lambdas_))

def plot_KPCA(*data):
    X,y = data
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    fig = plt.figure()
    colors = ((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),
              (0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2))
    for i,kernel in enumerate(kernels):#同时遍历kernels 序号和内容
        kpca = decomposition.KernelPCA(n_components=2,kernel=kernel)
        kpca.fit(X)
        X_r = kpca.transform(X)
        ax = fig.add_subplot(2,2,i+1)
        for label,color in zip(np.unique(y),colors):
            position=y == label
            ax.scatter(X_r[position, 0], X_r[position, 1], label="target = %d" % label, color=color)
        ax.set_xlabel("x[0]")
        ax.set_ylabel("y[0]")
        ax.legend(loc="best")
        ax.set_title("kernel = %s"%kernel)
    plt.suptitle("KPCA")
    plt.show()

if __name__ == '__main__':
    X,y = load_data()
    test_pca(X,y)
    plot_PCA(X,y)
    test_KPCA(X,y)
    plot_KPCA(X,y)