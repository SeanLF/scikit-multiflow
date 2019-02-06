__author__ = 'Sean Floyd'
from skmultiflow.options import Classifier, Window, Voting 
from skmultiflow.demos.thesis_experiments import thesis_experiment

CLASSIFIER = Classifier.VOTING_ENSEMBLE
VOTING_TYPE = Voting.W_AVG_PROBABILITY
WINDOW_TYPE = Window.HYBRID
GROUND_TRUTH_PERCENTAGE = 100

# drift params

# for ground_truth_percentage in [100,90,80,70,60]:
for detector_count in DriftDetectorCount:
    for drift_reset in DriftReset:
        for drift_content in DriftContent:
            for batch_size in [100,75,50,25]:
                drift = {'reset_type': drift_reset, 'count': detector_count, 'content': drift_content}

                experiment_name = '{}|{}|{}gt|{}ws'.format(WINDOW_TYPE.name, VOTING_TYPE.name, GROUND_TRUTH_PERCENTAGE, batch_size)
                thesis_experiment(
                    experiment_name,
                    folder='./_analysis_of_results/experiment_results_step3_drift',
                    window_type=WINDOW_TYPE,
                    voting=VOTING_TYPE,
                    window_size=batch_size*3,
                    batch_size=batch_size,
                    classifier=CLASSIFIER,
                    drift=drift,
                    g_t_percentage=GROUND_TRUTH_PERCENTAGE,
                )