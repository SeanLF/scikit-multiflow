__author__ = 'Sean Floyd'
from skmultiflow.options import Classifier, Window, Voting, DriftReset, DriftDetectorCount, DriftContent
from skmultiflow.demos.thesis_experiments import thesis_experiment
import sys

# CLASSIFIERS = []#Classifier.HOEFFDING_TREE, Classifier.MAJORITY_VOTING, Classifier.NO_CHANGE, Classifier.SGD]
WINDOW_TYPE = Window.SLIDING
BATCH_SIZES = [100,75,25]
VOTING_TYPE = Voting.PROBABILITY

args = sys.argv
classifier = Classifier.LEVERAGE_BAGGING
batch_size = BATCH_SIZES[int(args[1])]

# NB: DRIFT, VOTING_TYPE have no impact on these classifiers

experiment_name = '{}|{}|{}ws'.format(classifier.name, WINDOW_TYPE.name, batch_size)
thesis_experiment(
    experiment_name,
    classifier=classifier,
    voting_type=VOTING_TYPE,
    window_type=WINDOW_TYPE,
    folder='./_analysis_of_results/experiment_results_step4_sota/',
    window_size=batch_size*3,
    batch_size=batch_size,
)