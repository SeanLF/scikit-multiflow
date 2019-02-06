__author__ = 'Sean Floyd'
from skmultiflow.options import Classifier, Window, Voting, DriftReset, DriftDetectorCount, DriftContent
from skmultiflow.demos.thesis_experiments import thesis_experiment
import sys

CLASSIFIER = Classifier.VOTING_ENSEMBLE
GROUND_TRUTH_PERCENTAGES = [100,90,80,70,60]

args = sys.argv
ground_truth_percentage = GROUND_TRUTH_PERCENTAGES[int(args[1])]

VOTING_TYPES = [Voting.AVG_W_PROBABILITY, Voting.PROBABILITY, Voting.W_AVG_PROBABILITY]
WINDOW_TYPES = [Window.HYBRID, Window.SLIDING]
DRIFT_RESET_TYPES = [DriftReset.ALL, DriftReset.PARTIAL]
DRIFT_CONTENT = [DriftContent.PROBABILITY, DriftContent.WEIGHTED_PROBABILITY]

for batch_size in [100,75,25]:
  for voting_type in VOTING_TYPES:
      for window_type in WINDOW_TYPES:
        for drift_reset_type in DRIFT_RESET_TYPES:
          for drift_content in DRIFT_CONTENT:
            for detector_count in DriftDetectorCount:
              one_per__avg_w__weighted = voting_type == Voting.AVG_W_PROBABILITY and detector_count == DriftDetectorCount.ONE_PER_CLASSIFIER and drift_content == DriftContent.WEIGHTED_PROBABILITY
              proba__weighted = voting_type == Voting.PROBABILITY and drift_content == DriftContent.WEIGHTED_PROBABILITY
              bool__proba = voting_type == (Voting.BOOLEAN and (drift_content in [DriftContent.PROBABILITY, DriftContent.WEIGHTED_PROBABILITY])) or (voting_type != Voting.BOOLEAN and ((drift_content not in [DriftContent.PROBABILITY, DriftContent.WEIGHTED_PROBABILITY])))
              if one_per__avg_w__weighted or proba__weighted or bool__proba:
                continue # skip experiment if impossible combination of params
              
              partial_drift_reset_probability = 0.7
              pdrp = [1.-partial_drift_reset_probability, partial_drift_reset_probability]
              drift={'reset': drift_reset_type, 'detector_count': detector_count, 'content': drift_content, 'partial_drift_reset_p': pdrp}

              experiment_name = '{}|{}|{}gt|{}ws'.format(window_type.name, voting_type.name, ground_truth_percentage, batch_size)
              thesis_experiment(
                  experiment_name,
                  classifier=CLASSIFIER,
                  voting_type=voting_type,
                  window_type=window_type,
                  folder='./_analysis_of_results/experiment_results_step3_drift/{}gt'.format(ground_truth_percentage),
                  window_size=batch_size*3,
                  batch_size=batch_size,
                  drift=drift,
                  g_t_percentage=ground_truth_percentage,
              )