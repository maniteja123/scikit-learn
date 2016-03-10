# coding=utf8
"""
Self-taught learning uses unlabeled data in supervised classification tasks and assumes that the unlabeled data follows the 
same class labels  or generative distribution as labeled data. It uses a large number of unlabeled training data to improve
performance on a given classification task. Such unlabeled data is significantly easier to obtain than in typical
semisupervised or transfer learning settings, making selftaught learning widely applicable to many practical learning problems
We describe an approach to self-taught learning that uses sparse coding to construct higher-level features using the unlabeled
data. These features form a succinct input representation and significantly improve classification performance.

Examples
--------
>>> from sklearn.semi_supervised import SelfTaughtLearner
>>> from sklearn.svm import LinearSVC
>>> from sklearn import datasets
>>> import numpy as np
>>> from sklearn import metrics
>>> from sklearn.cluster import KMeans
>>> from sklearn.preprocessing import scale
>>> digits = datasets.load_digits()
>>> data = scale(digits.data)
>>> n_samples, n_features = data.shape
>>> n_digits = len(np.unique(digits.target))
>>> d = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
>>> s = LinearSVC()
>>> m = SelfTaughtLearner(d, s)
>>> X, y = digits.data / 255., digits.target
>>> X_train, X_test = X[:1500], X[1500:]
>>> y_train, y_test = y[:1500], y[1500:]
>>> labels = np.copy(digits.target)
>>> random_unlabeled_points = np.where(np.random.random_integers(0, 1, size=len(digits.target)))
>>> labels[random_unlabeled_points] = -1
>>> m.fit(X, labels)
SelfTaughtLearner(estimator=LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0),
         transformer=KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=10, n_init=10,
    n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
    verbose=0))
>>> y_pred = m.predict(X_test)
>>> print metrics.accuracy_score(y_pred, y_test)
0.804713804714

Notes
-----
References:
[1] Raina, Rohit et. al. Self-taught Learning: Transfer Learning from Unlabeled Data
ICML 2007
"""

# Licence: BSD

from abc import ABCMeta

import numpy as np
from scipy import sparse

from ..base import BaseEstimator, ClassifierMixin
from ..externals import six
from ..utils.multiclass import check_classification_targets
from ..utils.validation import check_X_y, check_is_fitted, check_array

class SelfTaughtLearner(six.with_metaclass(ABCMeta, BaseEstimator,
                                              ClassifierMixin)):
    def __init__(self, transformer, estimator):
        """
        Attributes
        ----------
        transformer : estimator object
            An estimator object implementing `transform`.

        estimator : estimator object
            An estimator object implementing `fit`,`score` and `predict_proba`.

        """

        self.transformer = transformer;
        self.estimator = estimator;

    def fit(self, X, y):
        """Fit a semi-supervised self taught model based

        All the input data is provided matrix X (labeled and unlabeled)
        and corresponding label matrix y with a dedicated marker value for
        unlabeled samples.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            A {n_samples by n_samples} size matrix will be created from this

        y : array_like, shape = [n_samples]
            n_labeled_samples (unlabeled points are marked as -1)
            All unlabeled samples will be transductively assigned labels

        Returns
        -------
        self : returns an instance of self.
        """
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        mask = y == -1
        self.transformer.fit(X[mask])
        Xt = self.transformer.transform(X[~mask])
        self.estimator.fit(Xt, y[~mask])
        return self

    def predict(self, X):
        """Predicts the output using the fitted estimator model.

        Parameters
        ----------
        X : array_like, shape = [n_samples, n_features]

        Returns
        -------
        y : array_like, shape = [n_samples]
            Predictions for input data
        """

        check_is_fitted(self, 'estimator', 'transformer')
        X = check_array(X, accept_sparse=['csc', 'csr', 'coo', 'dok',
                                             'bsr', 'lil', 'dia'])

        Xt = self.transformer.transform(X)
        return self.estimator.predict(Xt)

