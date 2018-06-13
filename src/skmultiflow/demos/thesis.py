__author__ = 'Sean Floyd'

from skmultiflow.core.pipeline import Pipeline
from skmultiflow.evaluation.evaluate_prequential import EvaluatePrequential
from skmultiflow.data.generators.waveform_generator import WaveformGenerator
from skmultiflow.data.generators.sea_generator import SEAGenerator
from skmultiflow.data.file_stream import FileStream

import numpy as np
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from skmultiflow.classification.scikit_learn.voting_classifier import EnsembleVoteClassifier
from skmultiflow.classification.trees.hoeffding_tree import HoeffdingTree
from skmultiflow.classification.meta.oza_bagging_adwin import OzaBaggingAdwin
from sklearn import linear_model
import logging
import random


def demo(filepath=None, noise_percentage=0.10, batch_size=1, show_plot=False):
    """ thesis
    
    This demo demonstrates the use of an ensemble learner.
     
    """

    stream = None
    file_stream = False
    if filepath is not None:
        stream = FileStream(filepath=filepath)
        file_stream = True
    else:
        # Test with WaveformGenerator with random seed
        stream = SEAGenerator(noise_percentage=noise_percentage)
        # stream = WaveformGenerator(seed=random_seed)

    stream.prepare_for_use()

    # Setup the classifier
    classes = np.asarray(stream.get_target_values() if file_stream else stream.target_values)
    ensemble = EnsembleVoteClassifier(voting='sum_prob', classes=classes, window_slide=batch_size, clfs=[
        GaussianNB()
        # ,BernoulliNB()
        ,linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
        ,MultinomialNB()
        # ,Perceptron()
    ])

    # Setup the pipeline
    # pipe = Pipeline([('vc', eclf1)])

    # Setup the evaluator
    evaluator = EvaluatePrequential(show_plot=show_plot,pretrain_size=1000, max_samples=100000, batch_size=batch_size)

    # Evaluate
    evaluator.evaluate(stream=stream, model=ensemble)

if __name__ == '__main__':
    files = [
        "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_101.csv",
        "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_103.csv",
        "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_102.csv",
        "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_105.csv",
        "/Users/sean/dev/tornado/data_streams/mixed_w_50_n_0.1/mixed_w_50_n_0.1_104.csv",

        "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_105.csv",
        "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_104.csv",
        "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_103.csv",
        "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_102.csv",
        "/Users/sean/dev/tornado/data_streams/circles_w_500_n_0.1/circles_w_500_n_0.1_101.csv",

        "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_104.csv",
        "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_105.csv",
        "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_101.csv",
        "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_102.csv",
        "/Users/sean/dev/tornado/data_streams/led_w_500_n_0.1/led_w_500_n_0.1_103.csv",
        
        "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_105.csv",
        "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_104.csv",
        "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_101.csv",
        "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_103.csv",
        "/Users/sean/dev/tornado/data_streams/sine1_w_50_n_0.1/sine1_w_50_n_0.1_102.csv"
    ]
    # for file in files:
    #     print(file)
    #     demo(file)

    demo(batch_size=3, show_plot=False)