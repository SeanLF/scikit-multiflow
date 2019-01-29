__author__ = 'Sean Floyd'

from skmultiflow.data.waveform_generator import WaveformGenerator
from skmultiflow.data.sea_generator import SEAGenerator
from skmultiflow.data.file_stream import FileStream
from skmultiflow.core.pipeline import Pipeline
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from skmultiflow.meta.oza_bagging_adwin import OzaBaggingAdwin
from skmultiflow.meta.leverage_bagging import LeverageBagging
from skmultiflow.scikit_learn.voting_classifier import EnsembleVoteClassifier
from skmultiflow.trees import HoeffdingTree
from skmultiflow.scikit_learn.base_classifiers import NoChangeClassifier, MajorityVoteClassifier
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

    # "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_101.csv",
    # "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_102.csv",
    # "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_103.csv",
    # "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_104.csv",
    # "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/led_w_500_n_0.1/led_w_500_n_0.1_105.csv",

    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_101.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_102.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_103.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_104.csv",
    "/Users/sean/Developer/scikit-multiflow/src/skmultiflow/datasets/tornado/sine1_w_50_n_0.1/sine1_w_50_n_0.1_105.csv",
]

def demo(stream, output_file, classifier, batch_size, window_size, window_type, drift, g_t_percentage, show_plot=False):
    """ thesis
    This demo demonstrates the use of an ensemble learner.
    """

    __args=inspect.getargvalues(inspect.currentframe()).locals; __args.pop('stream', None); __args.pop('show_plot', None)
    print(__args)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w+') as f:
        f.write(str(__args)+"\n\n")

    evaluator = EvaluatePrequential(
        show_plot=show_plot,
        pretrain_size=1000, 
        max_samples=100000,
        window_size=window_size,
        window_type=window_type,
        batch_size=batch_size,
        drift=drift,
        g_t_percentage=g_t_percentage,
        output_file=output_file,
        metrics=['kappa_t'])
    evaluator.evaluate(stream=stream, model=classifier)

def setup_clf(classifier, classes, window_type, drift, voting):
    clf = None
    if classifier == Classifier.VOTING_ENSEMBLE:
        clf = EnsembleVoteClassifier(drift=drift, voting=voting, classes=classes, window_type=window_type, clfs=[
            GaussianNB()
            ,linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3, n_jobs=-1)
            ,MultinomialNB()
        ])
    elif classifier == Classifier.LEVERAGE_BAGGING:
        clf = LeverageBagging(base_estimator=HoeffdingTree(), n_estimators=10)
    elif classifier == Classifier.GAUSSIAN_NB:
        clf = GaussianNB()
    elif classifier == Classifier.SGD:
        clf = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3, n_jobs=-1)
    elif classifier == Classifier.MULTINOMIAL_NB:
        clf = MultinomialNB()
    elif classifier == Classifier.MAJORITY_VOTING:
        clf = MajorityVoteClassifier()
    elif classifier == Classifier.NO_CHANGE:
        clf = NoChangeClassifier()
    elif classifier == Classifier.HOEFFDING_TREE:
        clf = HoeffdingTree()
    return clf 


# window size = [for SlidingTumbling] ensemble_number_classifiers*batch size []
def thesis_experiment(
                    experiment_name,
                    folder='./_analysis_of_results/experiment_results',
                    window_type=Window(3),
                    window_size=75,
                    batch_size=25,
                    voting=Voting('before_weight'),
                    classifier=Classifier(1),
                    drift={},
                    g_t_percentage=100):
    
    # Verify validity of batch and window sizes
    if batch_size > window_size:
        raise ValueError('Batch size must be smaller than window size')
    if window_size <= batch_size and window_type == Window.SLIDING:
        raise ValueError('Window must be smaller than batch size [SLIDING]')
    if window_type == Window.TUMBLING:
        window_size = batch_size

    streams__output_files = list()

    # for SEA generator
    for noise_percentage in np.linspace(0.0, 0.2, num=3, dtype=np.dtype('f')):
        path = '{}/{}/{}[SEA_noise_{}][{}].txt'.format(folder, experiment_name, classifier.name, noise_percentage, experiment_name)
        streams__output_files.append([SEAGenerator(noise_percentage=noise_percentage, random_state=1), False, os.path.join(path)])
    # for Tornado files
    for filepath in TORNADO_FILES:
        _=re.findall(r'\/(\w+)_w_\d+_n_0.1_(\d+).csv',filepath)[0]
        path = '{}/{}/{}[{}_{}][{}].txt'.format(folder, experiment_name, classifier.name, _[0], _[1], experiment_name)
        streams__output_files.append([FileStream(filepath=filepath), True, os.path.join(path)])
    
    # run all
    for stream, is_file, output_file in streams__output_files:
        clf=setup_clf(classifier, prepare_for_use(stream, is_file), window_type, drift, voting)
        if classifier == Classifier.VOTING_ENSEMBLE and window_type == Window.HYBRID:
            size = len(clf.clfs)*batch_size
            if window_size != size:
                window_size = size
        demo(stream, output_file, batch_size=batch_size, window_type=window_type, window_size=window_size, classifier=clf, drift=drift, g_t_percentage=g_t_percentage)
    
def prepare_for_use(stream, file_stream):
    stream.prepare_for_use()
    return np.asarray(
        stream.get_target_values() if file_stream else stream.target_values)

# ------------------------------------------------------------- #
#                                                               #
#                      THESIS EXPERIMENTS                       #
#                                                               #
# ------------------------------------------------------------- #

# The experiments were split up into the following files:
    # batch_size__voting_type__window_type__100_ground_truth.py