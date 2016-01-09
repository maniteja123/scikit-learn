"""
MultiOneVsRestClassifier
===========================
This module includes several classes that extend base estimators to
multi-target estimators.

Most sklearn estimators use a response matrix to train a target function
with a single output variable.  I.e. typical estimators use the
training set X to estimate a target function f(X) that
predicts a single Y.  The purpose of this class is to extend estimators
to be able to estimate a series of target functions (f1,f2,f3...,fn)
that are trained on a single X predictor matrix to predict a series
of reponses (y1,y2,y3...,yn).
"""

#Author: Hugo Bowne-Anderson <hugobowne@gmail.com>
#Author: Chris Rivera <chris.richard.rivera@gmail.com>
#Author: Michael Williamson

#License: BSD 3 clause

import array
import numpy as np
import warnings
import scipy.sparse as sp

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.base import clone, is_classifier
from sklearn.base import MetaEstimatorMixin, is_regressor
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import check_consistent_length
from sklearn.utils.validation import check_is_fitted
from sklearn.externals.joblib import Parallel
from sklearn.externals.joblib import delayed


from sklearn.multiclass import OneVsRestClassifier


class MultiOneVsRestClassifier():
    """
    Converts any classifer estimator into
    a multi-target classifier estimator.

    This class fits and predicts a series of one-versus-all models
    to response matrix Y, which has n_samples and p_target variables,
    on the predictor Matrix X with n_samples and m_feature variables.
    This allows for multiple target variable classifications. For each
    target variable (column in Y), a separate OneVsRestClassifier is fit.
    See the base OneVsRestClassifier Class in sklearn.multiclass
    for more details.

    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` &
         `predict_proba`.
    n_jobs : int, optional, default: 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used.
        If 1 is given, no parallel computing code is used at all,
        which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are
        used. Thus for n_jobs = -2, all CPUs but one are used.

        Note that parallel processing only occurs if there is
        multiple classes within each target variable.
        It does each target variable in y in series.

    Attributes
    __________
    estimator: Sklearn estimator: The base estimator used to constructe
    the model.

    """

    def __init__(self, estimator=None, n_jobs=1):

        self.estimator = estimator
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """ Fit the model to data.
        Creates a seperate model for each Response column.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
            Data.

        y : (sparse) array-like, shape = [n_samples, p_targets]
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        Returns
        -------
        self
        """

        # check to see that the data is numeric

        # check to see that the X and y have the same number of rows.
        check_consistent_length(X, y)

        # Calculate the number of classifiers
        self._num_y = y.shape[1]

        ## create a dictionary to hold the estimators.
        self.estimators_ ={}

        for i in range(self._num_y):
            # init a new classifer for each and fit it.
            estimator = clone(self.estimator)   #make a fresh clone
            ovr = OneVsRestClassifier(estimator,self.n_jobs)
            self.estimators_[i] = ovr.fit(X,y[:, i])

        return self

    def predict(self, X):
        """Predict multi-class multiple target variable using a model
         trained for each target variable.

        Parameters
        ----------
        X : (sparse) array-like, shape = [n_samples, n_features]
        Data.

        Returns
        -------
        y : dict of [sparse array-like], shape = {predictors: n_samples}
          or {predictors: [n_samples, n_classes], n_predictors}.
            Predicted multi-class targets across multiple predictors.
            Note:  entirely separate models are generated
            for each predictor.
        """
        # check to see if the fit has been performed
        check_is_fitted(self, 'estimators_')

        results = {}
        for label, model_ in self.estimators_.iteritems():
            results[label] = model_.predict( X)
        return(results)

    def predict_proba(self, X):
        """Probability estimates. This returns prediction probabilites
        for each class for each label in the form of a dictionary.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Returns
        -------
        prob_dict (dict) A dictionary containing n_label sparse arrays
        with shape = [n_samples, n_classes].
        Each row in the array contains the the probability of the
        sample for each class in the model,
        where classes are ordered as they are in `self.classes_`.
        """
        # check to see whether the fit has occured.
        check_is_fitted(self, 'estimators_')

        results ={}
        for label, model_ in self.estimators_.iteritems():
            results[label] = model_.predict_proba(X)
        return(results)

    def score(self, X, Y):
        """"Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        Y : (sparse) array-like, shape = [n_samples, p_targets]

        Returns
        -------
        scores (np.array) Array of p_target floats of the mean accuracy
        of each estimator_.predict wrt. y.
        """
        check_is_fitted(self, 'estimators_')
        # Score the results for each function
        results =[]
        for i in range(self._num_y):
            estimator = self.estimators_[i]
            results.append(estimator.score(X,Y[:,i]))
        return results

    def get_params(self):
        '''returns the parameters of the estimator.'''
        return self.estimator.get_params()

    def set_params(self, params):
        """sets the params for the estimator."""
        self.estimator.set_params(params)

    def __repr__(self):
        return 'MultiOneVsRestClassifier( %s )' %self.estimator.__repr__()

    @property
    def multilabel_(self):
        """returns boolean vector iniwhether each classifer is a multilabel classifier"""
        return [(label, model_.multilabel_) for label, model_ in self.estimators_.iteritems()]

    @property
    def classes_(self):
        return [(label, model_.classes_) for label, model_ in self.estimators_.iteritems()]
