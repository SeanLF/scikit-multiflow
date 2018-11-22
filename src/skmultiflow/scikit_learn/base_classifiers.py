# No-change classifier

# Sean Floyd 2018
#
# Implementation of a no-change classifier
# Author: Sean Floyd <sfloyd@uottawa.ca>

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.stats import itemfreq
import operator

from skmultiflow.options import Classifier, Window, Voting, DriftReset


class NoChangeClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    """No-Change classifier predicts that the next class will be the same as the last seen class.

    Attributes
    ----------
    last_seen : integer
        The last seen class label
    """
    def __init__(self):
        self.last_seen = None

    def partial_fit(self, X, y, classes=None):
        return self.fit(X, y, classes=classes)
    
    def fit(self, X, y, classes=None):
        self.last_seen = y[-1]
    
    def predict(self, X):
        return np.asarray([self.last_seen] * len(X))

    def reset(self, reset_strategy):
        self.last_seen = None

    def refit(self, X, y, reset_strategy):
        self.last_seen = y[-1]


class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):

    """Majority class classifier: always predicts the class that has been observed most frequently the in the training data.

    Attributes
    ----------
    class_distribution : hash_map, shape: [int]=int
        hash map, keys are class labels, values are instances seen
    """
    def __init__(self):
        self._observed_class_distribution = {} # Dictionary (class_value, weight)
    
    def partial_fit(self, X, y, classes=None):
        return self.fit(X, y, classes=classes)
    
    def fit(self, X, y, classes=None):
        freq = itemfreq(y)
        for label, count in freq:
            if label in self._observed_class_distribution.keys():
              self._observed_class_distribution[label] += count
            else:
              self._observed_class_distribution[label] = count
    
    def predict(self, X):
        return np.asarray([max(self._observed_class_distribution.items(), key=operator.itemgetter(1))[0]] * len(X))

    def reset(self, reset_strategy):
        self._observed_class_distribution = {}

    def refit(self, X, y, reset_strategy):
        self.fit(X, y)