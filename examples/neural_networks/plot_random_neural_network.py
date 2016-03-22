"""
===========================================
Effect of parameters in RandomBasisFunction
===========================================

This example generates plots that illustrate the impact of varying the RandomBasisFunction parameters on the decision
function of the random neural network model.

This generates three plots, each corresponding to varying one single parameter. The plots correspond to varying the
parameter alpha, weight_scale, and n_output, respectively.

If there is high bias in the model, which can lead to a high training error, then decreasing alpha,
increasing weight_scale, and/or increasing n_output decreases bias and therefore reduces underfitting.
Similarly, if there is high variance in the model, which is when the training error poorly approximates the testing
error, then increasing alpha, decreasing weight_scale, and/or decreasing n_output would decrease variance and therefore
reduces overfitting.

One way to find a balance between bias and variance when tuning these parameters is by
testing a range of values using cross-validation as seen in this example.

"""
print(__doc__)


# Author: Issam H. Laradji
# License: BSD 3 clause

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import RandomBasisFunction
from sklearn.learning_curve import validation_curve
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.utils.fixes import expit as logistic_sigmoid


# To be removed (no predict_proba in Ridge)
def predict_proba(clf, x):
    return logistic_sigmoid(clf.predict(x))

h = .02  # step size in the mesh
rng = np.random.RandomState(1)

alpha_list = np.logspace(-4, 4, 5)
weight_scale_list = np.logspace(-2, 2, 5)
n_outputs_list = [2, 10, 100, 200, 500]




def plot(names, classifiers, title):
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                               random_state=rng, n_clusters_per_class=1)

    linearly_separable = (X, y)

    datasets = [make_moons(noise=1., random_state=rng),
                make_circles(noise=0.2, factor=0.5, random_state=rng),
                linearly_separable]

    figure = plt.figure(figsize=(17, 9))
    figure.suptitle(title)
    i = 1
    # iterate over datasets
    for X, y in datasets:
        # initialize standard scaler
        scaler = StandardScaler()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4,
                                                            random_state=1)
        # Compute the mean and standard deviation of each feature of the
        # training set and scale the training set
        X_train = scaler.fit_transform(X_train)

        # Using the same mean and standard deviation, scale the testing set
        X_test = scaler.transform(X_test)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            # Plot the decision boundary.
            Z = predict_proba(clf, np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)

            ax.contourf(xx, yy, Z, cmap=plt.cm.RdBu, alpha=.8)

            # Plot also the training points
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)
            # and testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                       alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')
            i += 1

classifiers = []
names = []
for alpha in alpha_list:
    clf = make_pipeline(RandomBasisFunction(weight_scale=1.), Ridge(alpha=alpha))
    classifiers.append(clf)
    names.append("alpha = " + str(alpha))

title = "Effect of varying alpha for fixed weight_scale=1"
plot(names, classifiers, title)

classifiers = []
names = []
for weight_scale in weight_scale_list:
    clf = make_pipeline(RandomBasisFunction(weight_scale=weight_scale), Ridge(alpha=1.))
    classifiers.append(clf)
    names.append("weight_scale = " + str(weight_scale))

title = "Effect of varying weight_scale for fixed alpha=1"
plot(names, classifiers, title)

classifiers = []
names = []
for n_outputs in n_outputs_list:
    clf = make_pipeline(RandomBasisFunction(n_outputs=n_outputs), Ridge(alpha=1.))

    classifiers.append(clf)
    names.append("n_output = " + str(n_outputs))

title = "Effect of varying n_output in RandomBasisFunction"
plot(names, classifiers, title)

plt.show()
