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


def demo(stream, classes, batch_size=1, show_plot=False):
    """ thesis
    
    This demo demonstrates the use of an ensemble learner.
     
    """

    # Setup the classifier
    ensemble = EnsembleVoteClassifier(voting='sum_prob', classes=classes, window_slide=batch_size, clfs=[
        GaussianNB()
        ,linear_model.SGDClassifier(loss='log', max_iter=1000, tol=1e-3)
        ,MultinomialNB()
    ])

    # Setup the pipeline
    # pipe = Pipeline([('vc', eclf1)])

    # Setup the evaluator
    evaluator = EvaluatePrequential(show_plot=show_plot,pretrain_size=1000, max_samples=100000, batch_size=batch_size)

    # Evaluate
    evaluator.evaluate(stream=stream, model=ensemble)