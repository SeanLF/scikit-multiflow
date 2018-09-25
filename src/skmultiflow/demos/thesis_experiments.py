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
import os
import inspect

TORNADO_FILES = [
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/mixed_w_50_n_0.1/mixed_w_50_n_0.1_101.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/mixed_w_50_n_0.1/mixed_w_50_n_0.1_102.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/mixed_w_50_n_0.1/mixed_w_50_n_0.1_103.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/mixed_w_50_n_0.1/mixed_w_50_n_0.1_104.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/mixed_w_50_n_0.1/mixed_w_50_n_0.1_105.csv",

    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/circles_w_500_n_0.1/circles_w_500_n_0.1_101.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/circles_w_500_n_0.1/circles_w_500_n_0.1_102.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/circles_w_500_n_0.1/circles_w_500_n_0.1_103.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/circles_w_500_n_0.1/circles_w_500_n_0.1_104.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/circles_w_500_n_0.1/circles_w_500_n_0.1_105.csv",

    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_101.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_102.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_103.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_104.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_105.csv",
    
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_101.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_102.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_103.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_104.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_105.csv",
]

def demo(stream, output_file, classifier, batch_size, window_size, window_type, drift_reset, drift_g_t_percentage, show_plot=False):
    """ thesis
    This demo demonstrates the use of an ensemble learner.
    """

    __args=inspect.getargvalues(inspect.currentframe()).locals; __args.pop('stream', None); __args.pop('show_plot', None)
    print(__args)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w+') as f:
        f.write(str(__args)+"\n\n")
                

    # pipe = Pipeline([('vc', eclf1)])
    evaluator = EvaluatePrequential(
        show_plot=show_plot,
        pretrain_size=1000, 
        max_samples=100000,
        window_size=window_size,
        window_type=window_type,
        batch_size=batch_size,
        output_file=output_file)
    evaluator.evaluate(stream=stream, model=classifier)

def setup_clf(classifier, classes, window_type, voting=Voting('sum_prob')):
    clf = None
    if classifier == Classifier.VOTING_ENSEMBLE:
        clf = EnsembleVoteClassifier(voting=voting, classes=classes, window_type=window_type, clfs=[
            GaussianNB()
            ,linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
            ,MultinomialNB()
        ])
    elif classifier == Classifier.OZA_BAGGING:
        clf = OzaBagging()
    elif classifier == Classifier.LEVERAGE_BAGGING:
        clf = LeverageBagging()
    elif classifier == Classifier.GAUSSIAN_NB:
        clf = GaussianNB()
    elif classifier == Classifier.SGD:
        clf = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
    elif classifier == Classifier.MULTINOMIAL_NB:
        clf = MultinomialNB()
    return clf 

# Experiments to perform

# window size = [for SlidingTumbling] ensemble_number_classifiers*batch size []
def thesis_experiment(
                    experiment_name,
                    window_type=Window(3),
                    window_size=99,
                    batch_size=33,
                    voting=Voting('sum_prob'),
                    classifier=Classifier(1),
                    drift_reset=DriftReset(3),
                    drift_g_t_percentage=0.5):
    
    # Verify validity of batch and window sizes
    if batch_size > window_size:
        raise ValueError('Batch size must be smaller than window size')
    if window_size <= batch_size and window_type == Window.SLIDING:
        raise ValueError('Window must be smaller than batch size [SLIDING]')
    if window_size != batch_size and (window_type == Window.TUMBLING):
        raise ValueError('Window and batch size must be identical [TUMBLING]')
    if classifier == Classifier.VOTING_ENSEMBLE and window_type == Window.SLIDING_TUMBLING and window_size != 3*batch_size:
        window_size = 3*batch_size # TODO: undo hardcode ensemble size

    streams__output_files = list()

    # for Tornado files
    for filepath in TORNADO_FILES:
        _=re.findall(r'\/(\w+)_w_\d+_n_0.1_(\d+).csv',filepath)[0]
        streams__output_files.append([FileStream(filepath=filepath), True, os.path.join(str('./experiment_results/'+experiment_name),classifier.name+'['+_[0]+'_'+_[1]+'].txt')])
    # for SEA generator
    for noise_percentage in np.linspace(0.0, 0.7, num=8, dtype=np.dtype('f')):
        streams__output_files.append([SEAGenerator(noise_percentage=noise_percentage), False, os.path.join(str('./experiment_results/' + experiment_name), classifier.name + '[SEA_noise_' + str(noise_percentage) + '].txt')])
    # # for Waveform generator (with and without noise)
    # streams__output_files.append([WaveformGenerator(has_noise=True), False, os.path.join(str('./experiment_results/' + experiment_name), classifier.name + '[Waveform_no_noise].txt')])
    # streams__output_files.append([WaveformGenerator(has_noise=False), False, os.path.join(str('./experiment_results/' + experiment_name), classifier.name + '[Waveform_noise].txt')])

    # run all
    for stream, is_file, output_file in streams__output_files:
        clf=setup_clf(classifier, prepare_for_use(stream, is_file), window_type, voting=voting)
        demo(stream, output_file, batch_size=batch_size, window_type=window_type, window_size=window_size, classifier=clf, drift_reset=drift_reset, drift_g_t_percentage=drift_g_t_percentage)
    
def prepare_for_use(stream, file_stream):
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
    thesis_experiment('compare_all', classifier=Classifier(clf))

## Examine how sliding windows perform against tumbling windows and against sliding tumbling windows
thesis_experiment('window_type', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.SLIDING)
thesis_experiment('window_type', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.TUMBLING, window_size=33)
thesis_experiment('window_type', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.SLIDING_TUMBLING)

## The size of the window or batch (w) will have an impact on the results; we probably need some experiments about that.
for window_size in range(0, 100, 10):
    thesis_experiment('window_size', window_size=window_size, batch_size=window_size)

## Compare different voting ensemble strategies against one another and against single classifiers 
# and against other ensemble methods. Compare outcomes
thesis_experiment('voting_type', classifier=Classifier.VOTING_ENSEMBLE, voting=Voting.SUM_PROB_VOTING) # default strategy
thesis_experiment('voting_type', classifier=Classifier.VOTING_ENSEMBLE, voting=Voting.HARD_VOTING)
thesis_experiment('voting_type', classifier=Classifier.VOTING_ENSEMBLE, voting=Voting.SOFT_VOTING)

## Find the right balance of ground truth that can be omitted versus using predicted values as the ground truth
for percentage in range(0.0, 1.0, 0.1):
    thesis_experiment('drift_ground_truth_reliance', drift_g_t_percentage=percentage)

## See how the ensemble classifier reset logic affects the results
# It is good to compare to blind adaptation, i.e. a simple model reset at every x instances.
thesis_experiment('drift_reset', drift_reset=DriftReset.NONE)
thesis_experiment('drift_reset', drift_reset=DriftReset.BLIND_RESET)
thesis_experiment('drift_reset', drift_reset=DriftReset.RESET_ALL)
thesis_experiment('drift_reset', drift_reset=DriftReset.PARTIAL_RESET)

## See how the modified concept drift detector performs depending on window type

# TODO: implement
## Determine if the summarising classifiers improve performance
## Determine threshold when best to use summarizer over the normal voting classifiers.