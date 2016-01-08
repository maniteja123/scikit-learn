import numpy as np

from sklearn.multioutput import MultiOneVsRestClassifier 
from sklearn.datasets import load_digits
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import libraries for vaidation
import sklearn.utils.estimator_checks as ec
import sklearn.utils.validation  as val
from sklearn import datasets

# import the shuffle 
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer

# these are function for testing
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_almost_equal


def _create_data_set():
    # Creates a multi-target data set using the iris data set
    # X : The iris predictor data.
    # Y : A multi-target (150x3) array generated from the original response data
    
    # Import the data 
    iris = datasets.load_iris()
    X = iris.data

    # create a multiple targets by randomizing 
    #the shuffling and concatenating y. 
    y1 = iris.target
    y2 = shuffle(y1, random_state = 1) 
    y3 = shuffle(y1, random_state = 2)

    # concatenate the array and transpose
    Y = np.vstack((y1,y2,y3)).T
    
    return(X,Y)


def _set_up_multi_target_random_forest():
    # Set up the forest and multi-target forest'''
    
    forest = RandomForestClassifier(n_estimators =100, random_state=1)
    multi_target_forest = MultiOneVsRestClassifier(forest, n_jobs = -1)
    
    return forest, multi_target_forest


def test_multi_target_fit_with_random_forest():
    # test if multi_target initilizes correctly and fit procedure with random forest
    # Assert that predictions work as expected for predict, prodict_proba and score
    
    X,Y = _create_data_set()
    forest, multi_target_forest = _set_up_multi_target_random_forest()
    
    # check to see that the estimator type is correct
    assert_equal(forest, multi_target_forest.estimator)
    #check to that the number of jobs is correct
    assert_equal(-1,multi_target_forest.n_jobs)
    # train the multi_target_forest and also get the predictions. 
    multi_target_forest.fit(X,Y)
    predictions = multi_target_forest.predict(X)
    assert_equal(3,len(predictions))
    predict_proba = multi_target_forest.predict_proba(X)
    multi_score = multi_target_forest.score(X,Y)
    
    # train the forest with each column and then assert that the predictions are equal
    for i in range(3):   
        forest_ = clone(forest)  #create a clone with the same state
        forest_.fit(X,Y[:,i])
        assert_equal(list(forest_.predict(X)), list(predictions[i]))
        assert_almost_equal(list(forest_.predict_proba(X)), list(predict_proba[i]), decimal = 1)
        score = forest_.score(X,Y[:,i])
        assert_almost_equal(score, multi_score[i])
