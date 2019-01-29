__author__ = 'Sean Floyd'
from skmultiflow.options import Classifier, Window, Voting, DriftReset
from skmultiflow.demos.thesis_experiments import thesis_experiment

CLASSIFIER = Classifier.VOTING_ENSEMBLE
VOTING_TYPE = Voting.BEFORE_WEIGHT
WINDOW_TYPES = [Window.SLIDING, Window.HYBRID]

# drift params


for batch_size in [100,75,50,25]:
    for window_type in WINDOW_TYPES:
        for ground_truth_percentage in [100,90,80,70,60]:
            experiment_name = '{}|{}|{}gt|{}ws'.format(window_type.name, VOTING_TYPE.name, ground_truth_percentage, batch_size)
            thesis_experiment(
                experiment_name,
                folder='./_analysis_of_results/experiment_results_step3_drift',
                window_type=window_type,
                voting=VOTING_TYPE,
                window_size=batch_size*3,
                batch_size=batch_size,
                classifier=CLASSIFIER,
                g_t_percentage=ground_truth_percentage
            )