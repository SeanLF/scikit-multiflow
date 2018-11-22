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

def demo(stream, output_file, classifier, batch_size, window_size, window_type, drift, show_plot=True):
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
        drift=drift,
        output_file=output_file)
    evaluator.evaluate(stream=stream, model=classifier)

def setup_clf(classifier, classes, window_type, drift, voting=Voting('sum_prob')):
    clf = None
    if classifier == Classifier.VOTING_ENSEMBLE:
        clf = EnsembleVoteClassifier(drift=drift, voting=voting, classes=classes, window_type=window_type, clfs=[
            GaussianNB()
            ,linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3, n_jobs=-1)
            ,MultinomialNB()
        ])
    elif classifier == Classifier.LEVERAGE_BAGGING:
        clf = LeverageBagging()
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
                    drift={}):
    
    # Verify validity of batch and window sizes
    if batch_size > window_size:
        raise ValueError('Batch size must be smaller than window size')
    if window_size <= batch_size and window_type == Window.SLIDING:
        raise ValueError('Window must be smaller than batch size [SLIDING]')
    if window_size != batch_size and (window_type == Window.TUMBLING):
        raise ValueError('Window and batch size must be identical [TUMBLING]')
        # -------------------------------------------- #
        # TODO: undo hardcode ensemble size ^          #
        # -------------------------------------------- #

    streams__output_files = list()

    # for Tornado files
    for filepath in TORNADO_FILES:
        _=re.findall(r'\/(\w+)_w_\d+_n_0.1_(\d+).csv',filepath)[0]
        streams__output_files.append([FileStream(filepath=filepath), True, os.path.join(str('./experiment_results/'+experiment_name),classifier.name+'['+_[0]+'_'+_[1]+'].txt')])
    # for SEA generator
    for noise_percentage in np.linspace(0.0, 0.2, num=5, dtype=np.dtype('f')):
        streams__output_files.append([SEAGenerator(noise_percentage=noise_percentage), False, os.path.join(str('./experiment_results/' + experiment_name), classifier.name + '[SEA_noise_' + str(noise_percentage) + '].txt')])
    
    # run all
    for stream, is_file, output_file in streams__output_files:
        clf=setup_clf(classifier, prepare_for_use(stream, is_file), window_type, drift, voting=voting)
        if classifier == Classifier.VOTING_ENSEMBLE and window_type == Window.SLIDING_TUMBLING and window_size != len(clf.clfs)*batch_size:
            window_size = len(clf.clfs)*batch_size

        demo(stream, output_file, batch_size=batch_size, window_type=window_type, window_size=window_size, classifier=clf, drift=drift)
    
def prepare_for_use(stream, file_stream):
    stream.prepare_for_use()
    return np.asarray(
        stream.get_target_values() if file_stream else stream.target_values)

# ------------------------------------------------------------- #
#                                                               #
#                      THESIS EXPERIMENTS                       #
#                                                               #
# ------------------------------------------------------------- #

# Evaluate the performance, stream velocity, accuracy against other methods
# for clf in Classifier:
#     thesis_experiment('compare_all', classifier=Classifier(clf))

# # Examine how sliding windows perform against tumbling windows and against sliding tumbling windows
# thesis_experiment('window_type/sliding', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.SLIDING)
# thesis_experiment('window_type/tumbling', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.TUMBLING, window_size=33)
# thesis_experiment('window_type/hybrid', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.SLIDING_TUMBLING)

# # The size of the window or batch (w) will have an impact on the results; we probably need some experiments about that.
# for window_size in range(0, 110, 10):
#     thesis_experiment('window_size', window_size=window_size, batch_size=window_size)

# # Compare different voting ensemble strategies against one another and against single classifiers 
# # and against other ensemble methods. Compare outcomes
# thesis_experiment('voting_type/sum', classifier=Classifier.VOTING_ENSEMBLE, voting=Voting.SUM_PROB) # default strategy
# thesis_experiment('voting_type/hard', classifier=Classifier.VOTING_ENSEMBLE, voting=Voting.HARD)
# thesis_experiment('voting_type/soft', classifier=Classifier.VOTING_ENSEMBLE, voting=Voting.SOFT)

# # Find the right balance of ground truth that can be omitted versus using predicted values as the ground truth
# for percentage in [100, 90, 80, 70, 60, 50]:
#     for reset_type in DriftReset:
#         thesis_experiment(str(reset_type) + '_ground_truth_'+str(percentage), drift={'drift_reset': reset_type, 'g_t_%': percentage})

# See how the ensemble classifier reset logic affects the results
# It is good to compare to blind adaptation, i.e. a simple model reset at every x instances.

# See how the modified concept drift detector performs depending on window type
drift={'drift_reset': DriftReset.PARTIAL, 'g_t_%': 75}
# thesis_experiment('concept_drift/sliding', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.SLIDING, drift=drift)
# thesis_experiment('concept_drift/tumbling', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.TUMBLING, window_size=33, drift=drift)
thesis_experiment('concept_drift/sliding_tumbling', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.SLIDING_TUMBLING, drift=drift)

# majority class and no change classifier
thesis_experiment('baseline', classifier=Classifier.NO_CHANGE, window_size=33, batch_size=33, window_type=Window.TUMBLING)
thesis_experiment('baseline', classifier=Classifier.MAJORITY_VOTING, window_size=33, batch_size=33, window_type=Window.TUMBLING)