import numpy as np
from math import log
from sklearn.svm import SVC
from sklearn.utils.extmath import safe_sparse_dot


class FeatSVC(SVC):
    """A SVC classifier that can be trained using labeled features.
    """
    alpha = 1

    def fit(self, X, Y, sample_weight=None, features=None):
        """Fit Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).

        features : array-like, shape = [n_classes, n_features], optional
            Boost for the prior probability of a feature given a class. For no
            boost use the value alpha given on the initialization.

        Returns
        -------
        self : object
            Returns self.
        """
        return super(FeatSVC, self).fit(X, Y, sample_weight)

    def _information_gain(self):
        """Calculates the information gain for each feature.

        Stores the value in self.feat_information_gain
        """
        pass

    def instance_proba(self, X):
        """Calculates the probability of each instance in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        array-like, shape = [n_samples]
        """
        pass
