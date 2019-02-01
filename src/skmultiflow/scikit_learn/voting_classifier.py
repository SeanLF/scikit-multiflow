# Probability Voting/Majority Rule classifier

# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# Implementation of an meta-classification algorithm for majority voting.
# Author: Sebastian Raschka <sebastianraschka.com>
# Author: Sean Floyd <sfloyd@uottawa.ca>
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
from skmultiflow.utils.data_structures import FastInstanceWindow
from skmultiflow.drift_detection.fhddm import FHDDM, FHDDMS, PFHDDM, PFHDDMS
from sklearn.model_selection import ParameterGrid
from math import e, sin, pi, tanh

from skmultiflow.options import Classifier, Window, Voting, DriftReset, DriftContent, DriftDetectorCount


class EnsembleVoteClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    """Probability Voting/Majority Rule classifier for scikit-learn estimators.

    Parameters
    ----------
    clfs : array-like, shape = [n_classifiers]
        A list of classifiers.
    voting : Voting enum
        If 'BOOLEAN', uses predicted class labels for majority rule voting.
        Else if 'PROBABILITY', predicts the class label based on the argmax of
        the sums of the predicted probalities, which is recommended for
        an ensemble of well-calibrated classifiers.
    weights : array-like, shape = [n_classifiers], optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurances of
        predicted class labels (`BOOLEAN` voting) or class probabilities
        before averaging (`PROBABILITY` voting). Uses uniform weights if `None`.
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
    ... voting=Voting.BOOLEAN, verbose=1)
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting=Voting.BOOLEAN)
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
    ...                          voting=Voting.BOOLEAN, weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>
    """
    def __init__(self,
                clfs,
                drift,
                voting=Voting.BOOLEAN,
                window_type=Window.HYBRID,
                weights=None,
                verbose=0,
                classes=None):

        self.clfs = clfs

        self.drift_detection_enabled = bool(drift)
        if self.drift_detection_enabled:
            self.drift_reset = drift['reset']
            self.drift_detector_count = drift['detector_count']
            self.drift_content = drift['content']
            self.partial_drift_reset_probabilities = drift['partial_drift_reset_p']
            count = {DriftDetectorCount.ONE_PER_CLASSIFIER: len(clfs), DriftDetectorCount.ONE_FOR_ENSEMBLE: 1}[self.drift_detector_count]
            self.drift_detectors = [PFHDDMS() for _ in range(count)]

        self.named_clfs = {key: value for key, value in _name_estimators(clfs)}
        self.weights = weights
        self.verbose = verbose
        self.first_fit = True

        self.voting = voting

        self.window_type = window_type
        self.mod = [0, len(clfs)] # for SlidingTumbling windows

        self.classes_ = classes

        self.sigm = [14, 0.5]
        self.tanh = [3.5, 7]
        self.weight_fn = 'tanh'

        self.le_ = LabelEncoder()
        self.le_.fit(classes)
        self.classes_ = self.le_.classes_

        # self.proba_clfs = []
        # self.non_proba_clfs = []

        self.reset_clfs = [False]*len(clfs)

        # for idx, clf in enumerate(self.clfs):
        #     if callable(getattr(clf, "predict_proba", None)):
        #         self.proba_clfs.append(idx)
        #     else:
        #         self.non_proba_clfs.append(idx)

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

        if self.voting not in Voting:
            raise ValueError("Voting type incorrect; got (voting=%r). See enums.py"
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
            y = self.le_.transform(y)

            if self.window_type == Window.HYBRID:
                to_fit = self.clfs[self.mod[0]]
                to_fit.partial_fit(X, y)
                self.mod[0] = (self.mod[0] + 1) % self.mod[1]
            else:
                for clf in self.clfs:
                    self._first_fit_one_clf(clf, X, y)

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

        if self.voting == Voting.BOOLEAN: # majority vote
            predictions = self._predict(X)
        
        else: # PROBABILITY, AVG_W OR W_AVG
            # Common to all probability voting options
            probabilities = self._predict_probas(X)
            averaged_probabilities = np.average(probabilities, axis=0, weights=self.weights)
            weighted_probabilities, averaged_weighted_probabilities, weighted_averaged_probabilities = None, None, None
		
            # avg probabilities then apply weights
            if self.voting == Voting.AVG_W_PROBABILITY:
                weighted_probabilities = [np.apply_along_axis(self._prediction_weighting, arr=x, axis=1) for x in probabilities]
                averaged_weighted_probabilities = np.average(weighted_probabilities, axis=0)
            
            # apply weights to each classifiers probability then average
            if self.voting == Voting.W_AVG_PROBABILITY:
                weighted_averaged_probabilities = np.apply_along_axis(self._prediction_weighting, arr=averaged_probabilities, axis=0)

        # Determine winning class
        if self.voting == Voting.BOOLEAN:
            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x, weights=self.weights)), axis=1, arr=predictions)
        else:
            to_vote = { Voting.PROBABILITIY: averaged_probabilities, Voting.AVG_W_PROBABILITY: averaged_weighted_probabilities, Voting.W_AVG_PROBABILITY: weighted_averaged_probabilities }
            maj = np.argmax(to_vote[self.voting], axis=1)

        if self.drift_detection_enabled:
			# one drift detector per ensemble
            if self.drift_detector_count == DriftDetectorCount.ONE_FOR_ENSEMBLE:
                detector = self.drift_detectors[0]
                pr = []
                if self.drift_content == DriftContent.BOOLEAN:
					# count classifiers who voted winning class
                    pr = [np.bincount(x)[maj[i]] for i,x in enumerate(predictions)]
                else:
                    weighted = {Voting.AVG_W_PROBABILITY: averaged_weighted_probabilities, Voting.W_AVG_PROBABLITY: weighted_averaged_probabilities }[self.voting]
                    content = {DriftContent.PROBABILITY: averaged_probabilities, DriftContent.WEIGHTED_PROBABILITY: weighted}[self.drift_content]
                    # takes the average probability of the winning class (for unweighted)
                    # takes the average weighted probability of the winning class (for weighted)
                    pr = [v[maj[i]] for i, v in enumerate(content)]
                # check for drift over pr
                detector.run(pr)

            elif self.drift_detector_count == DriftDetectorCount.ONE_PER_CLASSIFIER:
                # one drift detector per classifier in the ensemble
                for index, detector in enumerate(self.drift_detectors):
                    pr = []
                    if self.drift_content == DriftContent.BOOLEAN:
                        # did classifier vote winning vote ?
                        pr = [maj[i] == predictions[i][index] for i in range(len(predictions))]
                    else: # note that since AVG_G doesn't calculate weights for each probability, it cannot be used with ONE PER CLASSIFIER
                        content = {DriftContent.PROBABILITY: probabilities, DriftContent.WEIGHTED_PROBABILITY: weighted_probabilities}[self.drift_content]
                        # probability of winning vote
                        pr = [content[index][i][maj[i]] for i in range(len(content[index]))]

                    detector.run(pr)

        return self.le_.inverse_transform(maj)

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
        If `voting='PROBABILITY'` : array-like = [n_classifiers, n_samples, n_classes]
            Class probabilties calculated by each classifier.
        If `voting='BOOLEAN'` : array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.

        """
        if self.voting == Voting.PROBABILITIY:
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

    # def run_drift_detection(self):
        

    def reset(self, reset_strategy):
        if reset_strategy == DriftReset.ALL:
            self.clfs = [self.reset_one_clf(clf) for clf in self.clfs] # hack
            for d in self.drift_detectors:
                d.reset()
        elif reset_strategy == DriftReset.PARTIAL:
            # each classifier has a 70% chance of being reset
            for idx, _ in enumerate(self.clfs):
                will_reset = np.random.choice([False, True], p=self.partial_drift_reset_probabilities)
                self.reset_clfs[idx] = will_reset
                if will_reset:
                    self.clfs[idx] = self.reset_one_clf(self.clfs[idx])
                    self.drift_detectors[idx].reset()

    def reset_one_clf(self, clf):
        if hasattr(clf, 'reset'):
            clf.reset()
            return clf
        else:
            return clf.__class__(**clf.get_params())
        

    def refit(self, X, y, reset_strategy):
        if reset_strategy == DriftReset.ALL:
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

    def _prediction_weighting(self, *args):
        if self.weight_fn == 'sigm':
            return [self._sigmoid_weight(x) for x in args[0]]
        elif self.weight_fn == 'tanh':
            w_p=[]
            for x in args[0]:
                w_p.append(self._tanh_weight(x))
            return w_p
        elif self.weight_fn == 'none':
            return args[0]
        
    def _tanh_weight(self, x):
        return (1-tanh(self.tanh[0]-self.tanh[1]*x))/2
    
    def _sigmoid_weight(self, x):
        return 1 / (1 + (np.exp((-self.sigm[0]) * (x - self.sigm[1]))))
