__author__ = 'Sean Floyd'
from skmultiflow.options import Classifier, Window, Voting
from skmultiflow.demos.thesis_experiments import thesis_experiment

CLASSIFIER = Classifier.VOTING_ENSEMBLE
GROUND_TRUTH_PERCENTAGE = 100

for batch_size in [100,75,50,25,10,5,1]:
    for voting_type in Voting: # hard, soft, w_before, w_after
        for window_type in Window: # sliding, tumbling, hybrid
            experiment_name = '{}|{}|{}gt|{}ws'.format(window_type.name, voting_type.name, GROUND_TRUTH_PERCENTAGE, batch_size)
            thesis_experiment(
                experiment_name,
                folder='./_analysis_of_results/experiment_results',
                window_type=window_type,
                voting=voting_type,
                window_size=batch_size*3,
                batch_size=batch_size,
                classifier=CLASSIFIER,
                g_t_percentage=GROUND_TRUTH_PERCENTAGE
            )