__author__ = 'Sean Floyd'

from skmultiflow.data.waveform_generator import WaveformGenerator
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.file_stream import FileStream

from skmultiflow.core.pipeline import Pipeline
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from skmultiflow.meta.oza_bagging import OzaBagging
from skmultiflow.scikit_learn.voting_classifier import EnsembleVoteClassifier
from sklearn import linear_model

import numpy as np

from enum import Enum, auto

class Window(Enum):
    SLIDING=auto()
    TUMBLING=auto()
    SLIDING_TUMBLING=auto()

class DriftReset(Enum):
    NONE=auto()
    BLIND=auto()
    PARTIAL=auto()
    ALL=auto()

class Voting(Enum):
    SUM_PROB='sum_prob'
    HARD='hard'
    SOFT='soft'

class Classifier(Enum):
    VOTING_ENSEMBLE=auto()
    OZA_BAGGING=auto()
    # OZA_BOOSTING=auto() TODO: uncomment when implemented
    GAUSSIAN_NB=auto()
    SGD=auto()
    Multinomial_NB=auto()

TORNADO_FILES = [
    "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_101.csv",
    "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_103.csv",
    "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_102.csv",
    "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_105.csv",
    "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_104.csv",

    "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_105.csv",
    "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_104.csv",
    "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_103.csv",
    "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_102.csv",
    "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_101.csv",

    "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_104.csv",
    "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_105.csv",
    "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_101.csv",
    "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_102.csv",
    "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_103.csv",
    
    "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_105.csv",
    "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_104.csv",
    "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_101.csv",
    "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_103.csv",
    "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_102.csv"
]

def demo(stream, classes, classifier=Classifier.VOTING_EMSEMBLE, batch_size=1, show_plot=False):
    """ thesis
    This demo demonstrates the use of an ensemble learner.
    """

    # Setup the classifier
    clf = setup_clf(classifier, classes, batch_size=batch_size)

    # pipe = Pipeline([('vc', eclf1)])

    evaluator = EvaluatePrequential(show_plot=show_plot, pretrain_size=1000, max_samples=100000, batch_size=batch_size)
    evaluator.evaluate(stream=stream, model=clf)

def setup_clf(classifier, classes, batch_size=0):
    clf = None
    if classifier == Classifier.VOTING_ENSEMBLE:
        clf = EnsembleVoteClassifier(voting='sum_prob', classes=classes, window_slide=batch_size, clfs=[
            GaussianNB()
            ,linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
            ,MultinomialNB()
        ])
    elif classifier == Classifier.OZA_BAGGING:
        clf = OzaBagging()
    elif classifier == Classifier.OZA_BOOSTING:
        return None
    elif classifier == Classifier.GAUSSIAN_NB:
        clf = GaussianNB()
    elif classifier == Classifier.SGD:
        clf = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
    elif classifier == Classifier.Multinomial_NB:
        clf = MultinomialNB()
    return clf 

# Experiments to perform

# window size depends on the number of classifiers in the ensemble
def thesis_experiment(window_type=Window.SLIDING_TUMBLING, window_size=0, drift_reset=DriftReset.PARTIAL_RESET, drift_g_t_percentage=0.5, voting=Voting.SUM_PROB_VOTING, classifier=Classifier.VOTING_ENSEMBLE):
    # for Tornado files
    for filepath in TORNADO_FILES:
        stream = FileStream(filepath=filepath)
        demo(stream, prepare_for_use(stream, True), batch_size=window_size)
    
    # for SEA generator
    for noise_percentage in range(0.0, 0.7, 0.1):
        stream = SEAGenerator(noise_percentage=noise_percentage)
        demo(stream, prepare_for_use(stream, False), batch_size=window_size)
    
    # for Waveform generator (with and without noise)
    stream = WaveformGenerator(has_noise=True)
    demo(stream, prepare_for_use(stream, False), batch_size=window_size)

    stream = WaveformGenerator(has_noise=False)
    demo(stream, prepare_for_use(stream, False), batch_size=window_size)

def prepare_for_use(stream, file_stream):
    stream.prepare_for_use()
    return np.asarray(
        stream.get_target_values() if file_stream else stream.target_values)

## Examine how sliding windows perform against tumbling windows and against sliding tumbling windows
## See how the modified concept drift detector performs depending on window type
thesis_experiment(window_type=Window.SLIDING)
thesis_experiment(window_type=Window.TUMBLING)
thesis_experiment(window_type=Window.SLIDING_TUMBLING)

## The size of the window or batch (w) will have an impact on the results; we probably need some experiments about that.
for window_size in range(0, 100, 10):
    thesis_experiment(window_size=window_size)

## Compare different voting ensemble strategies against one another and against single classifiers 
# and against other ensemble methods. Compare outcomes
thesis_experiment(voting=Voting.SUM_PROB_VOTING)
thesis_experiment(voting=Voting.HARD_VOTING)
thesis_experiment(voting=Voting.SOFT_VOTING)

## Find the right balance of ground truth that can be omitted versus using predicted values as the ground truth
for percentage in range(0.0, 1.0, 0.1):
    thesis_experiment(drift_g_t_percentage=percentage)

## See how the ensemble classifier reset logic affects the results
# It is good to compare to blind adaptation, i.e. a simple model reset at every x instances.
thesis_experiment(drift_reset=DriftReset.NONE)
thesis_experiment(drift_reset=DriftReset.BLIND_RESET)
thesis_experiment(drift_reset=DriftReset.RESET_ALL)
thesis_experiment(drift_reset=DriftReset.PARTIAL_RESET)

## Evaluate the performance, stream velocity, accuracy against other methods
for clf in Classifier:
    thesis_experiment(classifier=clf)

# TODO: implement
## Determine if the summarising classifiers improve performance
## Determine threshold when best to use summarizer over the normal voting classifiers.