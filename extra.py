# Miscellaneous auxilliary functions

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


# A function to visualize 2D decision boundaries
def plot_decision_regions(X, y, classifier, test_idx=None, 
                          resolution=0.02):
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    
    # meshgrid returns two grids - interesting function for
    # pairing all values of x1 with all values of xy. alternative
    # to using nested for loops. useful for function evaluation 
    # on all possible combinations of two variables. 
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution),
                          )
    x1_x2_pairs = np.array([xx1.ravel(),xx2.ravel()]).T
    Z = classifier.predict(x1_x2_pairs) # gives the prediction at each grid point
    Z = Z.reshape(xx1.shape) # reshapes to the size of the grid

    # plot the decision regions
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        # plot the points in each class independently
        plt.scatter(x=X[y == cl, 0], # **remember this for conditional indexing
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black'
                   )

    # highlight the test samples
    if test_idx:
        # plot all the samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='test set')








