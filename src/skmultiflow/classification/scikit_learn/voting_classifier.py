# Soft Voting/Majority Rule classifier

# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# Implementation of an meta-classification algorithm for majority voting.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from mlxtend.externals.name_estimators import _name_estimators
from mlxtend.externals import six
import numpy as np
from skmultiflow.core.utils.data_structures import FastInstanceWindow
from skmultiflow.drift_detection.fhddm import FHDDM
from sklearn.model_selection import ParameterGrid


class EnsembleVoteClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    """Soft Voting/Majority Rule classifier for scikit-learn estimators.

    Parameters
    ----------
    clfs : array-like, shape = [n_classifiers]
        A list of classifiers.
    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probalities, which is recommended for
        an ensemble of well-calibrated classifiers.
    weights : array-like, shape = [n_classifiers], optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurances of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the clf being fitted
        - `verbose=2`: Prints info about the parameters of the clf being fitted
        - `verbose>2`: Changes `verbose` param of the underlying clf to
           self.verbose - 2

    Attributes
    ----------
    classes_ : array-like, shape = [n_predictions]
    clf : array-like, shape = [n_predictions]
        The unmodified input classifiers

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from mlxtend.sklearn import EnsembleVoteClassifier
    >>> clf1 = LogisticRegression(random_seed=1)
    >>> clf2 = RandomForestClassifier(random_seed=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
    ... voting='hard', verbose=1)
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
    ...                          voting='soft', weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>
    """
    def __init__(self, clfs, voting='hard',
                 weights=None, verbose=0, classes=None, window_slide=1, reset_all_clfs=True):

        self.clfs = clfs
        self.named_clfs = {key: value for key, value in _name_estimators(clfs)}
        self.voting = voting
        self.weights = weights
        self.verbose = verbose
        self.first_fit = True
        self.window_slide = window_slide
        self.window = FastInstanceWindow(max_size=len(clfs)*self.window_slide)
        self.mod = [0, len(clfs)]
        self.drift_detectors = [FHDDM(delta=0.000001) for _ in clfs]
        self.classes_ = classes

        self.le_ = LabelEncoder()
        self.le_.fit(classes)
        self.classes_ = self.le_.classes_

        self.proba_clfs = []
        self.non_proba_clfs = []

        self.reset_clfs = [False]*len(clfs)
        self.reset_all_clfs = reset_all_clfs

        for idx, clf in enumerate(self.clfs):
            if callable(getattr(clf, "predict_proba", None)):
                self.proba_clfs.append(idx)
            else:
                self.non_proba_clfs.append(idx)

    def partial_fit(self, X, y, classes=None):
        return self.fit(X,y, classes=self.classes_)

    def fit(self, X, y, classes=None):
        """Learn weight coefficients from training data for each classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard', 'sum_prob'):
            raise ValueError("Voting must be 'soft' or 'hard' or 'sum_prob'; got (voting=%r)"
                             % self.voting)

        if self.weights and len(self.weights) != len(self.clfs):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d clfs'
                             % (len(self.weights), len(self.clfs)))

        if self.first_fit:
            if self.verbose > 0:
                print("Fitting %d classifiers..." % (len(self.clfs)))

            for clf in self.clfs:
                self._first_fit_one_clf(clf, X, y)

            self.first_fit=False
        else:
            # First fit is only true on the first function call
            # This should only be called when streaming
            self.window.add_elements(np.asarray(X), np.asarray(self.le_.transform(y)))
            # self.clfs = [clf.partial_fit(self.window) for clf in self.clfs]
            to_fit = self.clfs[self.mod[0]]
            to_fit.partial_fit(self.window.get_attributes_matrix(), self.window.get_targets_matrix().ravel())
            self.mod[0] = (self.mod[0] + 1) % self.mod[1]
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.

        """
        if not hasattr(self, 'clfs'):
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        if self.voting == 'soft':

            maj = np.argmax(self.predict_proba(X), axis=1)

        elif self.voting == 'hard':
            predictions = self._predict(X)

            maj = np.apply_along_axis(lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)

        elif self.voting == 'sum_prob':
            predictions = self._predict_probas(X)
            sum = np.sum(predictions, axis=0)
            maj = np.argmax(sum, axis=1)

            for index, detector in enumerate(self.drift_detectors):
                pr = [maj[i] == [np.argmax(predictions[index][i])] for i in range(len(predictions[index]))]
                if detector.run(np.hstack(pr)):
                    print("drift detected by ", self.clfs[index].__class__, end='')
                    self.first_fit = True
                    break
        
        maj = self.le_.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """
        if not hasattr(self, 'clfs'):
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        avg = np.average(self._predict_probas(X), axis=0, weights=self.weights)
        return avg

    def transform(self, X):
        """ Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'` : array-like = [n_classifiers, n_samples, n_classes]
            Class probabilties calculated by each classifier.
        If `voting='hard'` : array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.

        """
        if self.voting == 'soft':
            return self._predict_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support."""
        if not deep:
            return super(EnsembleVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_clfs.copy()
            for name, step in six.iteritems(self.named_clfs):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            for key, value in six.iteritems(super(EnsembleVoteClassifier,
                                            self).get_params(deep=False)):
                out['%s' % key] = value
            return out

    def reset(self):
        if self.reset_all_clfs:
            self.clfs = [clf.__class__(**ParameterGrid(clf.get_params()).param_grid[0]) for clf in self.clfs] # hack
            for d in self.drift_detectors:
                d.reset()
        else:
            # each classifier has a 70% chance of being reset
            for idx, clf in enumerate(self.clfs):
                will_reset = np.random.choice([False, True], p=[0.3, 0.7])
                self.reset_clfs[idx] = will_reset
                if will_reset:
                    self.clfs[idx] = clf.__class__(**ParameterGrid(clf.get_params()).param_grid[0])
                    self.drift_detectors[idx].reset()

    def refit(self):
        X, y = self.window.get_attributes_matrix(), self.window.get_targets_matrix().ravel()
        if self.reset_all_clfs:
            self.partial_fit(X, y, classes=self.classes_)
        else:
            for idx, clf in enumerate(self.clfs):
                if self.reset_clfs[idx]:
                    self._first_fit_one_clf(clf, X, y)
                    self.reset_clfs[idx] = False

    def _predict(self, X):
        """Collect results from clf.predict calls."""

        if self.first_fit:
            return np.asarray([clf.predict(X) for clf in self.clfs]).T
        else:
            return np.asarray([self.le_.transform(clf.predict(X))
                               for clf in self.clfs]).T

    def _predict_probas(self, X):
        """Collect results from clf.predict_proba calls."""
        return np.asarray([clf.predict_proba(X) for clf in self.clfs])

    def _first_fit_one_clf(self, clf, X, y):
        if self.verbose > 0:
            i = self.clfs.index(clf) + 1
            print("Fitting clf%d: %s (%d/%d)" %
                    (i, _name_estimators((clf,))[0][0], i,
                    len(self.clfs)))

        if self.verbose > 2:
            if hasattr(clf, 'verbose'):
                clf.set_params(verbose=self.verbose - 2)

        if self.verbose > 1:
            print(_name_estimators((clf,))[0][1])

        clf.partial_fit(X, self.le_.transform(y), classes=self.classes_)