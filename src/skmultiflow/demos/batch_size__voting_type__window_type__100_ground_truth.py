__author__ = 'Sean Floyd'
from skmultiflow.options import Classifier, Window, Voting
from skmultiflow.demos.thesis_experiments import thesis_experiment

CLASSIFIER = Classifier.VOTING_ENSEMBLE
GROUND_TRUTH_PERCENTAGE = 100

for batch_size in [100,75,50,25,10,5]:
    for voting_type in Voting: # boolean, probability, avg_w, w_avg
        for window_type in Window: # sliding, tumbling, hybrid
            experiment_name = '{}|{}|{}gt|{}ws'.format(window_type.name, voting_type.name, GROUND_TRUTH_PERCENTAGE, batch_size)
            thesis_experiment(
                experiment_name,
                classifier=CLASSIFIER,
                voting_type=voting_type,
                window_type=window_type,
                folder='./_analysis_of_results/experiment_results_step1',
                window_size=batch_size*3,
                batch_size=batch_size,
                g_t_percentage=GROUND_TRUTH_PERCENTAGE
            )