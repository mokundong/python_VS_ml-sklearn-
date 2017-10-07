#-*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn import cluster
from sklearn.metrics import adjusted_rand_score
from sklearn import mixture

def create_data(centers,num,std):
    X, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    return X,labels_true

def plot_data(*data):
    X,labels_true = data
    labels = np.unique(labels_true)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    colors = 'rgbyckm'
    for i,label in enumerate(labels):
        position = labels_true == label
        ax.scatter(X[position,0],X[position,1],label="cluster %d"%label)
        color = colors[i%len(colors)]
    ax.legend(loc="best",framealpha=0.5)
    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.set_title("data")
    plt.show()

def test_KMeans(*data):
    X, labels_true = data
    clst = cluster.KMeans()
    clst.fit(X)
    predicted_labels = clst.predict(X)
    print("ARI:%s"%adjusted_rand_score(labels_true,predicted_labels))
    print("Sum center distance %s"%clst.inertia_)

if __name__ == "__main__":
    # X,labels_true = create_data([[1,1],[2,2],[1,2],[10,20]],1000,0.5)
    # plot_data(X,labels_true)

    centers = [[1,1],[2,2],[1,2],[10,20]]
    X,labels_true = create_data(centers,1000,0.5)
    test_KMeans(X,labels_true)
