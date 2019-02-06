# __author__ = 'Sean Floyd'
# from skmultiflow.options import Classifier, Window, Voting, DriftReset
# from skmultiflow.demos.thesis_experiments import thesis_experiment

# to_test = {'tanh': {'drift_use_w': [True, False], 'w': [[3.5, 7],], 'drift_detection_method': ['one_proba']}}
# # test_experimentally('test_experimentally_ddm', to_test, drift={'drift_reset': DriftReset.ALL, 'g_t_%': 100})

# partial_drift_reset_probability = 0.7
# pdrp = [1.-partial_drift_reset_probability, partial_drift_reset_probability]
# drift={'drift_reset': DriftReset.ALL, 'drift_detection_method': 'one_proba', 'drift_use_weighted_probabilities': True}#, 'partial_drift_reset_p': pdrp}
# # Examine how sliding windows perform against tumbling windows and against sliding tumbling windows
# thesis_experiment('window_type/sliding', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.SLIDING, drift=drift)
# thesis_experiment('window_type/tumbling', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.TUMBLING, window_size=33, drift=drift)
# thesis_experiment('window_type/hybrid', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.HYBRID, drift=drift)

# # See how the modified concept drift detector performs depending on window type
# drift={'drift_reset': DriftReset.ALL, 'g_t_%': 80}
# thesis_experiment('concept_drift/sliding', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.SLIDING, drift=drift)
# thesis_experiment('concept_drift/tumbling', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.TUMBLING, window_size=5, drift=drift)
# thesis_experiment('concept_drift/hybrid', classifier=Classifier.VOTING_ENSEMBLE, window_type=Window.HYBRID, drift=drift)

# for percentage in [100,90,80,70,60]:
#     experiment_name='ground_truth_percentage/{}'.format(percentage)
#     thesis_experiment(experiment_name, classifier=Classifier.VOTING_ENSEMBLE, voting=Voting.W_AVG_PROBABILITY, g_t_percentage=percentage)

# # The size of the window or batch (w) will have an impact on the results; we probably need some experiments about that.
# for window_size in [1,5,10,25,50,75,100]:
#     thesis_experiment('window_size_drift/' + str(window_size), window_size=window_size, batch_size=window_size, drift=drift)

# # Compare different voting ensemble strategies against one another and against single classifiers 
# # and against other ensemble methods. Compare outcomes
# thesis_experiment('voting_type_drift/', classifier=Classifier.VOTING_ENSEMBLE, voting=Voting.PROBABILITY)#, drift=drift) # default strategy
# thesis_experiment('voting_type_drift/W_AVG', classifier=Classifier.VOTING_ENSEMBLE, voting=Voting.W_AVG_PROBABILITY)#, drift={'drift_reset': DriftReset.ALL, 'g_t_%': 80, 'drift_detection_method': 'one_proba', 'drift_use_weighted_probabilities': False})
# thesis_experiment('voting_type_drift/AVG_W', classifier=Classifier.VOTING_ENSEMBLE, voting=Voting.AVG_W_PROBABILITY)#, drift={'drift_reset': DriftReset.ALL, 'g_t_%': 80, 'drift_detection_method': 'one_proba', 'drift_use_weighted_probabilities': False})
# # TODO: add drift detection, and drift reset type, and ground truth percentage when best option found



# # Evaluate the performance, stream velocity, accuracy against other methods
# for clf in Classifier:
#     if clf != Classifier.LEVERAGE_BAGGING:
#         thesis_experiment('compare_all_drift/' + clf.name, classifier=Classifier(clf), drift=drift)
# # TODO: add drift detection, and drift reset type, and ground truth percentage when best option found

# # See how the ensemble classifier reset logic affects the results
# # Find the right balance of ground truth that can be omitted versus using predicted values as the ground truth
# for percentage in [100, 90, 80, 70, 60, 50]:
#     for reset_type in DriftReset:
#         thesis_experiment('reset/{}_{}'.format(reset_type, percentage), drift={'drift_reset': reset_type, 'g_t_%': percentage})