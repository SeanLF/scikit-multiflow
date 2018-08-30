__author__ = 'Sean Floyd'

from skmultiflow.data.waveform_generator import WaveformGenerator
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.file_stream import FileStream
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from skmultiflow.meta.oza_bagging import OzaBagging
from skmultiflow.meta.leverage_bagging import LeverageBagging
from skmultiflow.scikit_learn.voting_classifier import EnsembleVoteClassifier
from sklearn import linear_model
from skmultiflow.options import Classifier, Window, Voting, DriftReset

import numpy as np
import re

TORNADO_FILES = [
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/mixed_w_50_n_0.1/mixed_w_50_n_0.1_101.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/mixed_w_50_n_0.1/mixed_w_50_n_0.1_103.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/mixed_w_50_n_0.1/mixed_w_50_n_0.1_102.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/mixed_w_50_n_0.1/mixed_w_50_n_0.1_105.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/mixed_w_50_n_0.1/mixed_w_50_n_0.1_104.csv",

    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/circles_w_500_n_0.1/circles_w_500_n_0.1_105.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/circles_w_500_n_0.1/circles_w_500_n_0.1_104.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/circles_w_500_n_0.1/circles_w_500_n_0.1_103.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/circles_w_500_n_0.1/circles_w_500_n_0.1_102.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/circles_w_500_n_0.1/circles_w_500_n_0.1_101.csv",

    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_104.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_105.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_101.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_102.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_103.csv",
    
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_105.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_104.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_101.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_103.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_102.csv"
]

def demo(stream, streamName, classifier=clf, batch_size=1, show_plot=False):
    """ thesis
    This demo demonstrates the use of an ensemble learner.
    """

    # pipe = Pipeline([('vc', eclf1)])
    evaluator = EvaluatePrequential(
        show_plot=show_plot,
        pretrain_size=1000, 
        max_samples=100000,
        batch_size=batch_size,
        output_file=str(classifier.name) + '_' + streamName + '.txt')
    evaluator.evaluate(stream=stream, model=clf)

def setup_clf(classifier, classes, batch_size=1, window_size=1, voting=Voting('sum_prob'), window_type=Window(3)):
    clf = None
    if classifier == Classifier.VOTING_ENSEMBLE:
        clf = EnsembleVoteClassifier(voting=voting, classes=classes, window_slide=window_size, window_type=window_type, clfs=[
            GaussianNB()
            ,linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
            ,MultinomialNB()
        ])
    elif classifier == Classifier.OZA_BAGGING:
        clf = OzaBagging()
    elif classifier == Classifier.LEVERAGE_BAGGING:
        clf = LeverageBagging()
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
def thesis_experiment(
                    window_type=Window(3),
                    window_size=10,
                    batch_size=1,
                    voting=Voting('sum_prob'),
                    classifier=Classifier(1),
                    drift_reset=DriftReset(3),
                    drift_g_t_percentage=0.5):
    
    if batch_size > window_size:
        raise ValueError('Batch size must be smaller than window size')
    elif window_size <= batch_size and window_type == Window.SLIDING:
        raise ValueError('Window must be smaller than batch size [SLIDING]')
    elif window_size != batch_size:
        raise ValueError('Window and batch size must be identical [HYBRID/TUMBLING]')

    file = classifier.name

    # for Tornado files
    for filepath in TORNADO_FILES:
        # get file name for output file
        stream = FileStream(filepath=filepath)
        file += re.findall(r'\/(\w+)_w_\d+_n_0.1_(\d+).csv', filepath)[0]; file += '[' + file[0] + '_' + file[1] + ']'
        
        demo(stream, file, batch_size=batch_size, classifier=setup_clf(classifier, prepare_for_use(stream, True), batch_size=batch_size, voting=voting, window_type=window_type))
    
    # for SEA generator
    for noise_percentage in np.linspace(0, 0.7, num=8):
        stream = SEAGenerator(noise_percentage=noise_percentage)
        file += '[SEA_noise_' + str(noise_percentage) + ']'
        demo(stream, file, batch_size=window_size, classifier=setup_clf(classifier, prepare_for_use(stream), batch_size=window_size, voting=voting, window_type=window_type))
    
    # for Waveform generator (with and without noise)
    # stream = WaveformGenerator(has_noise=True)
    # file += '[Waveform_noise]'
    # demo(stream, file, batch_size=window_size, classifier=setup_clf(classifier, prepare_for_use(stream), batch_size=window_size, voting=voting, window_type=window_type=))

    # stream = WaveformGenerator(has_noise=False)
    # file += '[Waveform_no_noise]'
    # demo(stream, file, batch_size=window_size, classifier=setup_clf(classifier, prepare_for_use(stream), batch_size=window_size, voting=voting, window_type=window_type=))

def prepare_for_use(stream, file_stream=False):
    stream.prepare_for_use()
    return np.asarray(
        stream.get_target_values() if file_stream else stream.target_values)

# ------------------------------------------------------------- #
#                                                               #
#                      THESIS EXPERIMENTS                       #
#                                                               #
# ------------------------------------------------------------- #

## Evaluate the performance, stream velocity, accuracy against other methods
for clf in Classifier:
    thesis_experiment(classifier=Classifier(clf))

## Examine how sliding windows perform against tumbling windows and against sliding tumbling windows
## See how the modified concept drift detector performs depending on window type
thesis_experiment(classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.SLIDING, window_size=30, batch_size=10)
thesis_experiment(classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.TUMBLING)
thesis_experiment(classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.SLIDING_TUMBLING)

## The size of the window or batch (w) will have an impact on the results; we probably need some experiments about that.
for window_size in range(0, 100, 10):
    thesis_experiment(window_size=window_size, batch_size=window_size)

## Compare different voting ensemble strategies against one another and against single classifiers 
# and against other ensemble methods. Compare outcomes
# thesis_experiment(voting=Voting.SUM_PROB_VOTING) # default strategy already tested above
thesis_experiment(classifier=Classifier.VOTING_ENSEMBLE, voting=Voting.HARD_VOTING)
thesis_experiment(classifier=Classifier.VOTING_ENSEMBLE, voting=Voting.SOFT_VOTING)

## Find the right balance of ground truth that can be omitted versus using predicted values as the ground truth
for percentage in range(0.0, 1.0, 0.1):
    thesis_experiment(drift_g_t_percentage=percentage)

## See how the ensemble classifier reset logic affects the results
# It is good to compare to blind adaptation, i.e. a simple model reset at every x instances.
thesis_experiment(drift_reset=DriftReset.NONE)
thesis_experiment(drift_reset=DriftReset.BLIND_RESET)
thesis_experiment(drift_reset=DriftReset.RESET_ALL)
thesis_experiment(drift_reset=DriftReset.PARTIAL_RESET)

# TODO: implement
## Determine if the summarising classifiers improve performance
## Determine threshold when best to use summarizer over the normal voting classifiers.