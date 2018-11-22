"""
The :mod:`skmultiflow.scikit_learn` module includes the voting classifier
"""

from .voting_classifier import EnsembleVoteClassifier
from .base_classifiers import NoChangeClassifier
from .base_classifiers import MajorityVoteClassifier

__all__ = ["EnsembleVoteClassifier"]
