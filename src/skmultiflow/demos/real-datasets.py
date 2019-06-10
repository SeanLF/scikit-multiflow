__author__ = 'Sean Floyd'
from skmultiflow.options import Classifier, Window, Voting, DriftReset, DriftDetectorCount, DriftContent
from skmultiflow.demos.thesis_experiments import thesis_experiment
import sys

classifier = Classifier.VOTING_ENSEMBLE
args = { 
  's10':[Window.SLIDING, Voting.PROBABILITY, 100, 100, DriftReset.PARTIAL, DriftDetectorCount.ONE_FOR_ENSEMBLE, DriftContent.PROBABILITY],
  's90': [Window.SLIDING, Voting.PROBABILITY, 90, 100, DriftReset.PARTIAL, DriftDetectorCount.ONE_FOR_ENSEMBLE, DriftContent.PROBABILITY],
  'h100_75': [Window.HYBRID, Voting.PROBABILITY, 100, 75, DriftReset.ALL, DriftDetectorCount.ONE_FOR_ENSEMBLE, DriftContent.PROBABILITY],
}
partial_drift_reset_probability = 0.7
pdrp = [1.-partial_drift_reset_probability, partial_drift_reset_probability]

# NB: DRIFT, VOTING_TYPE have no impact on these classifiers

# for key in args.keys():
#   experiment_name = '{}|{}'.format(classifier.name, key)
#   params = args[key]
#   thesis_experiment(
#       experiment_name,
#       classifier=classifier,
#       window_type=params[0],
#       voting_type=params[1],
#       g_t_percentage=params[2],
#       window_size=params[3]*3,
#       batch_size=params[3],
#       drift={'reset': params[4], 'detector_count': params[5], 'content': params[6], 'partial_drift_reset_p': pdrp},
#       folder='./_analysis_of_results/adfa-ld/',
#   )

BATCH_SIZE = 100
classifier = Classifier.LEVERAGE_BAGGING
experiment_name = '{}'.format(classifier.name)
thesis_experiment(
    experiment_name,
    classifier=classifier,
    voting_type=Voting.PROBABILITY,
    window_type=Window.SLIDING,
    folder='./_analysis_of_results/adfa-ld/',
    window_size=BATCH_SIZE*3,
    batch_size=BATCH_SIZE,
)

# BATCH_SIZE = 1
# classifier = Classifier.MAJORITY_VOTING
# experiment_name = '{}'.format(classifier.name)
# thesis_experiment(
#     experiment_name,
#     classifier=classifier,
#     voting_type=Voting.PROBABILITY,
#     window_type=Window.SLIDING,
#     folder='./_analysis_of_results/adfa-ld/',
#     window_size=BATCH_SIZE*3,
#     batch_size=BATCH_SIZE,
# )