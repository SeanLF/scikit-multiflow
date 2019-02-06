__author__ = 'Sean Floyd'
from skmultiflow.options import Classifier, Window, Voting
from skmultiflow.demos.thesis_experiments import thesis_experiment

CLASSIFIER = Classifier.VOTING_ENSEMBLE
VOTING_TYPES = [Voting.W_AVG_PROBABILITY, Voting.AVG_W_PROBABILITY, Voting.PROBABILITY]
WINDOW_TYPES = [Window.TUMBLING, Window.SLIDING, Window.HYBRID]

for batch_size in [100,75,50,25]:
    for window_type in WINDOW_TYPES:
        for voting_type in VOTING_TYPES:
            for ground_truth_percentage in [100,90,80,70,60]:
                experiment_name = '{}|{}|{}gt|{}ws'.format(window_type.name, voting_type.name, ground_truth_percentage, batch_size)
                thesis_experiment(
                    experiment_name,
                    classifier=CLASSIFIER,
                    voting_type=voting_type,
                    window_type=window_type,
                    folder='./_analysis_of_results/experiment_results_step2_gt',
                    window_size=batch_size*3,
                    batch_size=batch_size,
                    g_t_percentage=ground_truth_percentage
                )