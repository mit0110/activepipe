# -*- coding: utf-8 -*-

"""
The :mod:`sklearn.naive_bayes` module implements Naive Bayes algorithms. These
are supervised learning methods based on applying Bayes' theorem with strong
(naive) feature independence assumptions.
"""

# Author: Vincent Michel <vincent.michel@inria.fr>
#         Minor fixes by Fabian Pedregosa
#         Amit Aides <amitibo@tx.technion.ac.il>
#         Yehuda Finkelstein <yehudaf@tx.technion.ac.il>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
#         (parts based on earlier work by Mathieu Blondel)
#
# License: BSD Style.

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import norm
from scipy.sparse import issparse

from .base import BaseEstimator, ClassifierMixin
from .preprocessing import binarize, LabelBinarizer
from .utils import array2d, atleast2d_or_csr
from .utils.extmath import safe_sparse_dot, logsumexp


class BaseNB(BaseEstimator, ClassifierMixin):
    """Abstract base class for naive Bayes estimators"""

    __metaclass__ = ABCMeta

    @abstractmethod
    def _joint_log_likelihood(self, X):
        """Compute the unnormalized posterior log probability of X

        I.e. log P(c) + log P(x|c) for all rows x of X, as an array-like of
        shape [n_classes, n_samples].

        Input is passed to _joint_log_likelihood as-is by predict,
        predict_proba and predict_log_proba.
        """

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted target values for X
        """
        jll = self._joint_log_likelihood(X)
        return self._classes[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class
            in the model, where classes are ordered arithmetically.
        """
        jll = self._joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        C : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in
            the model, where classes are ordered arithmetically.
        """
        return np.exp(self.predict_log_proba(X))


class GaussianNB(BaseNB):
    """
    Gaussian Naive Bayes (GaussianNB)

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.

    y : array, shape = [n_samples]
        Target vector relative to X

    Attributes
    ----------
    class_prior : array, shape = [n_classes]
        probability of each class.

    theta : array, shape = [n_classes, n_features]
        mean of each feature per class

    sigma : array, shape = [n_classes, n_features]
        variance of each feature per class

    Methods
    -------
    fit(X, y) : self
        Fit the model

    predict(X) : array
        Predict using the model.

    predict_proba(X) : array
        Predict the probability of each class using the model.

    predict_log_proba(X) : array
        Predict the log-probability of each class using the model.


    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> Y = np.array([1, 1, 1, 2, 2, 2])
    >>> from sklearn.naive_bayes import GaussianNB
    >>> clf = GaussianNB()
    >>> clf.fit(X, Y)
    GaussianNB()
    >>> print clf.predict([[-0.8, -1]])
    [1]
    """

    def fit(self, X, y):
        """Fit Gaussian Naive Bayes according to X, y

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """

        X = np.asarray(X)
        y = np.asarray(y)

        self._classes = unique_y = np.unique(y)
        n_classes = unique_y.shape[0]
        _, n_features = X.shape

        self.theta = np.empty((n_classes, n_features))
        self.sigma = np.empty((n_classes, n_features))
        self.class_prior = np.empty(n_classes)
        for i, y_i in enumerate(unique_y):
            self.theta[i, :] = np.mean(X[y == y_i, :], axis=0)
            self.sigma[i, :] = np.var(X[y == y_i, :], axis=0)
            self.class_prior[i] = np.float(np.sum(y == y_i)) / n_classes
        return self

    def _joint_log_likelihood(self, X):
        X = array2d(X)
        joint_log_likelihood = []
        for i in xrange(np.size(self._classes)):
            jointi = np.log(self.class_prior[i])
            n_ij = - 0.5 * np.sum(np.log(np.pi * self.sigma[i, :]))
            n_ij -= 0.5 * np.sum(((X - self.theta[i, :]) ** 2) / \
                                    (self.sigma[i, :]), 1)
            joint_log_likelihood.append(jointi + n_ij)
        joint_log_likelihood = np.array(joint_log_likelihood).T
        return joint_log_likelihood


class BaseDiscreteNB(BaseNB):
    """Abstract base class for naive Bayes on discrete/categorical data

    Any estimator based on this class should provide:

    __init__
    _joint_log_likelihood(X) as per BaseNB
    """

    def fit(self, X, y, sample_weight=None, class_prior=None):
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

        class_prior : array, shape [n_classes]
            Custom prior probability per class.
            Overrides the fit_prior parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        X = atleast2d_or_csr(X)
        Y = self._label_1ofK(y)

        if X.shape[0] != Y.shape[0]:
            msg = "X and y have incompatible shapes."
            if issparse(X):
                msg += "\nNote: Sparse matrices cannot be indexed w/ boolean \
                masks (use `indices=True` in CV)."
            raise ValueError(msg)

        self._fit1ofK(X, Y, sample_weight, class_prior)
        return self

    def _fit1ofK(self, X, Y, sample_weight, class_prior):
        """Guts of the fit method; takes labels in 1-of-K encoding Y"""

        n_classes = Y.shape[1]

        if sample_weight is not None:
            Y *= array2d(sample_weight).T

        if class_prior:
            assert len(class_prior) == n_classes, \
                   'Number of priors must match number of classes'
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            # empirical prior, with sample_weight taken into account
            y_freq = Y.sum(axis=0)
            self.class_log_prior_ = np.log(y_freq) - np.log(y_freq.sum())
        else:
            self.class_log_prior_ = np.zeros(n_classes) - np.log(n_classes)

        N_c, N_c_i = self._count(X, Y)

        self.feature_log_prob_ = (np.log(N_c_i + self.alpha)
                                - np.log(N_c.reshape(-1, 1)
                                       + self.alpha * X.shape[1]))

    @staticmethod
    def _count(X, Y):
        """Count feature occurrences.

        Returns (N_c, N_c_i), where
            N_c is the count of all features in all samples of class c;
            N_c_i is the count of feature i in all samples of class c.
        """
        N_c_i = safe_sparse_dot(Y.T, X)
        N_c = np.sum(N_c_i, axis=1)

        return N_c, N_c_i

    def _label_1ofK(self, y):
        """Convert label vector to 1-of-K and set self._classes"""

        y = np.asarray(y)
        if y.ndim == 1:
            labelbin = LabelBinarizer()
            Y = labelbin.fit_transform(y)
            self._classes = labelbin.classes_
            if Y.shape[1] == 1:
                Y = np.concatenate((1 - Y, Y), axis=1)
        else:
            Y = np.copy(y)

        return Y

    intercept_ = property(lambda self: self.class_log_prior_)
    coef_ = property(lambda self: self.feature_log_prob_)


class MultinomialNB(BaseDiscreteNB):
    """
    Naive Bayes classifier for multinomial models

    The multinomial Naive Bayes classifier is suitable for classification with
    discrete features (e.g., word counts for text classification). The
    multinomial distribution normally requires integer feature counts. However,
    in practice, fractional counts such as tf-idf may also work.

    Parameters
    ----------
    alpha: float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    fit_prior: boolean
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    Methods
    -------
    fit(X, y) : self
        Fit the model

    predict(X) : array
        Predict using the model.

    predict_proba(X) : array
        Predict the probability of each class using the model.

    predict_log_proba(X) : array
        Predict the log probability of each class using the model.

    Attributes
    ----------
    `intercept_`, `class_log_prior_` : array, shape = [n_classes]
        Log probability of each class (smoothed).

    `feature_log_prob_`, `coef_` : array, shape = [n_classes, n_features]
        Empirical log probability of features given a class, P(x_i|y).

        (`intercept_` and `coef_` are properties referring to
        `class_log_prior_` and `feature_log_prob_`, respectively.)

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(5, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 5, 6])
    >>> from sklearn.naive_bayes import MultinomialNB
    >>> clf = MultinomialNB()
    >>> clf.fit(X, Y)
    MultinomialNB(alpha=1.0, fit_prior=True)
    >>> print clf.predict(X[2])
    [3]

    References
    ----------
    For the rationale behind the names `coef_` and `intercept_`, i.e.
    naive Bayes as a linear classifier, see J. Rennie et al. (2003),
    Tackling the poor assumptions of naive Bayes text classifiers, ICML.
    """

    def __init__(self, alpha=1.0, fit_prior=True):
        self.alpha = alpha
        self.fit_prior = fit_prior

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        X = atleast2d_or_csr(X)
        return safe_sparse_dot(X, self.coef_.T) + self.intercept_


class BernoulliNB(BaseDiscreteNB):
    """Naive Bayes classifier for multivariate Bernoulli models.

    Like MultinomialNB, this classifier is suitable for discrete data. The
    difference is that while MultinomialNB works with occurrence counts,
    BernoulliNB is designed for binary/boolean features.

    Note: this class does not check whether features are actually boolean.

    Parameters
    ----------
    alpha: float, optional (default=1.0)
        Additive (Laplace/Lidstone) smoothing parameter
        (0 for no smoothing).
    binarize: float or None, optional
        Threshold for binarizing (mapping to booleans) of sample features.
        If None, input is presumed to already consist of binary vectors.
    fit_prior: boolean
        Whether to learn class prior probabilities or not.
        If false, a uniform prior will be used.

    Methods
    -------
    fit(X, y) : self
        Fit the model

    predict(X) : array
        Predict using the model.

    predict_proba(X) : array
        Predict the probability of each class using the model.

    predict_log_proba(X) : array
        Predict the log probability of each class using the model.

    Attributes
    ----------
    `class_log_prior_` : array, shape = [n_classes]
        Log probability of each class (smoothed).

    `feature_log_prob_` : array, shape = [n_classes, n_features]
        Empirical log probability of features given a class, P(x_i|y).

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randint(2, size=(6, 100))
    >>> Y = np.array([1, 2, 3, 4, 4, 5])
    >>> from sklearn.naive_bayes import BernoulliNB
    >>> clf = BernoulliNB()
    >>> clf.fit(X, Y)
    BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True)
    >>> print clf.predict(X[2])
    [3]

    References
    ----------
    C.D. Manning, P. Raghavan and H. Schütze (2008). Introduction to
    Information Retrieval. Cambridge University Press, pp. 234–265.

    A. McCallum and K. Nigam (1998). A comparison of event models for naive
    Bayes text classification. Proc. AAAI/ICML-98 Workshop on Learning for
    Text Categorization, pp. 41–48.

    V. Metsis, I. Androutsopoulos and G. Paliouras (2006). Spam filtering with
    naive Bayes -- Which naive Bayes? 3rd Conf. on Email and Anti-Spam (CEAS).
    """

    def __init__(self, alpha=1.0, binarize=.0, fit_prior=True):
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior

    def _count(self, X, Y):
        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)
        return super(BernoulliNB, self)._count(X, Y)

    def _joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""

        X = atleast2d_or_csr(X)

        if self.binarize is not None:
            X = binarize(X, threshold=self.binarize)

        n_classes, n_features = self.feature_log_prob_.shape
        n_samples, n_features_X = X.shape

        if n_features_X != n_features:
            raise ValueError("Expected input with %d features, got %d instead"
                             % (n_features, n_features_X))

        neg_prob = np.log(1 - np.exp(self.feature_log_prob_))
        # Compute  neg_prob · (1 - X).T  as  ∑neg_prob - X · neg_prob
        X_neg_prob = (neg_prob.sum(axis=1)
                    - safe_sparse_dot(X, neg_prob.T))
        jll = safe_sparse_dot(X, self.coef_.T) + X_neg_prob

        return jll + self.intercept_


class SemisupervisedNB(BaseNB):
    """Semisupervised Naive Bayes using expectation-maximization (EM)

    This meta-estimator can be used to train a Naive Bayes model in
    semisupervised mode, i.e. with a mix of labeled and unlabeled samples.

    Parameters
    ----------
    estimator : {BernoulliNB, MultinomialNB}
        Underlying Naive Bayes estimator. `GaussianNB` is not supported at
        this moment.
    n_iter : int, optional
        Maximum number of iterations.
    relabel_all : bool, optional
        Whether to re-estimate class memberships for labeled samples as well.
        Disabling this may result in bad performance, but follows Nigam et al.
        closely.
    tol : float, optional
        Tolerance, per coefficient, for the convergence criterion.
        Convergence is determined based on the coefficients (log probabilities)
        instead of the model log likelihood.
    verbose : boolean, optional
        Whether to print progress information.
    """

    def __init__(self, estimator, n_iter=10, relabel_all=True, tol=1e-5,
                 verbose=False):
        if not isinstance(estimator, BaseDiscreteNB):
            raise TypeError("%r is not a supported Naive Bayes classifier"
                            % (estimator,))
        self.estimator = estimator
        self.n_iter = n_iter
        self.relabel_all = relabel_all
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, class_prior=None):
        """Fit Naive Bayes estimator using EM

        This fits the underlying estimator at most n_iter times until its
        parameter vector converges. After every iteration, the posterior label
        probabilities (as returned by predict_proba) are used to fit in the
        next iteration.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values. Unlabeled samples should have a target value of -1.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).

        class_prior : array, shape [n_classes]
            Custom prior probability per class.
            Overrides the fit_prior parameter.

        Returns
        -------
        self : object
            Returns self.
        """

        clf = self.estimator
        X = atleast2d_or_csr(X)
        Y = clf._label_1ofK(y)

        labeled = np.where(y != -1)[0]
        if self.relabel_all:
            unlabeled = np.where(y == -1)[0]
            X_unlabeled = X[unlabeled, :]
            Y_unlabeled = Y[unlabeled, :]

        n_features = X.shape[1]
        tol = self.tol * n_features

        clf._fit1ofK(X[labeled, :], Y[labeled, :],
                     sample_weight[labeled, :] if sample_weight else None,
                     class_prior)
        old_coef = clf.coef_.copy()
        old_intercept = clf.intercept_.copy()

        for i in xrange(self.n_iter):
            if self.verbose:
                print "Naive Bayes EM, iteration %d," % i,

            # E
            if self.relabel_all:
                Y = clf.predict_proba(X)
            else:
                Y_unlabeled[:] = clf.predict_proba(X_unlabeled)

            # M
            clf._fit1ofK(X, Y, sample_weight, class_prior)

            d = (norm(old_coef - clf.coef_, 1)
               + norm(old_intercept - clf.intercept_, 1))
            if self.verbose:
                print "diff = %.3g" % d
            if d < tol:
                if self.verbose:
                    print "Naive Bayes EM converged"
                break

            old_coef[:] = clf.coef_
            old_intercept[:] = clf.intercept_

        return self

    # we "inherit" the crucial parts of an NB estimator from the underlying
    # one, so we can inherit the predict* methods from BaseNB with docstrings
    coef_ = property(lambda self: self.estimator.coef_)
    intercept_ = property(lambda self: self.estimator.intercept_)
    _classes = property(lambda self: self.estimator._classes)
    _joint_log_likelihood = \
        property(lambda self: self.estimator._joint_log_likelihood)