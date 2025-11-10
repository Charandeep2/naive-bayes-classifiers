"""
Naive Bayes Implementation
Supports both Numerical (Gaussian) and Categorical (Multinomial) features
Extracted from back-1.py for web application
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from math import sqrt, pi, exp
import warnings
warnings.filterwarnings('ignore')

class NaiveBayes:
    """
    Naive Bayes classifier supporting both numerical and categorical features
    """

    def __init__(self):
        self.classes = None
        self.class_probs = {}
        self.feature_params = defaultdict(dict)  # {class: {feature_idx: params}}
        self.feature_types = []  # 'numerical' or 'categorical'
        self.n_features = 0
        self.feature_names = []

    def _is_numerical(self, feature_values):
        """Check if a feature is numerical"""
        try:
            pd.to_numeric(feature_values, errors='raise')
            return True
        except:
            return False

    def _determine_feature_types(self, X):
        """Automatically determine if each feature is numerical or categorical"""
        self.feature_types = []
        for i in range(X.shape[1]):
            feature_values = X[:, i]
            is_num = self._is_numerical(feature_values)
            self.feature_types.append('numerical' if is_num else 'categorical')
        return self.feature_types

    def fit(self, X, y, feature_names=None):
        """
        Train the Naive Bayes classifier

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels
        feature_names : list, optional
            Names of features for visualization
        """
        X = np.array(X)
        y = np.array(y)
        self.n_features = X.shape[1]

        # Set feature names
        if feature_names is None:
            self.feature_names = [f'Feature_{i+1}' for i in range(self.n_features)]
        else:
            self.feature_names = feature_names

        # Determine feature types
        self._determine_feature_types(X)

        # Get unique classes
        self.classes = np.unique(y)
        n_samples = len(y)

        # Calculate prior probabilities P(class)
        for cls in self.classes:
            self.class_probs[cls] = np.sum(y == cls) / n_samples

        # Calculate feature parameters for each class
        for cls in self.classes:
            cls_mask = y == cls
            cls_X = X[cls_mask]
            n_cls_samples = len(cls_X)

            for feature_idx in range(self.n_features):
                feature_values = cls_X[:, feature_idx]

                if self.feature_types[feature_idx] == 'numerical':
                    # Gaussian: store mean and std
                    mean = np.mean(pd.to_numeric(feature_values, errors='coerce'))
                    std = np.std(pd.to_numeric(feature_values, errors='coerce'))
                    # Add small epsilon to avoid zero std
                    if std < 1e-6:
                        std = 1e-6
                    self.feature_params[cls][feature_idx] = {'type': 'numerical', 'mean': mean, 'std': std}
                else:
                    # Categorical: store probability distribution
                    unique_vals, counts = np.unique(feature_values, return_counts=True)
                    prob_dist = {}
                    for val, count in zip(unique_vals, counts):
                        # Laplace smoothing: add 1 to avoid zero probability
                        prob_dist[str(val)] = (count + 1) / (n_cls_samples + len(unique_vals))
                    self.feature_params[cls][feature_idx] = {'type': 'categorical', 'prob_dist': prob_dist}

    def _gaussian_pdf(self, x, mean, std):
        """Calculate Gaussian probability density function"""
        try:
            x = float(x)
            coefficient = 1 / (std * sqrt(2 * pi))
            exponent = -0.5 * ((x - mean) / std) ** 2
            return coefficient * exp(exponent)
        except:
            return 1e-10  # Return small probability if conversion fails

    def _predict_proba_single(self, x):
        """Predict probability for a single sample"""
        probabilities = {}

        for cls in self.classes:
            # Start with prior probability P(class)
            prob = np.log(self.class_probs[cls])

            # Multiply by likelihood P(feature|class) for each feature
            for feature_idx in range(self.n_features):
                feature_value = x[feature_idx]

                if self.feature_types[feature_idx] == 'numerical':
                    params = self.feature_params[cls][feature_idx]
                    feature_prob = self._gaussian_pdf(feature_value, params['mean'], params['std'])
                else:
                    params = self.feature_params[cls][feature_idx]
                    prob_dist = params['prob_dist']
                    feature_prob = prob_dist.get(str(feature_value), 1 / (len(prob_dist) + 1))

                # Use log probabilities to avoid underflow
                prob += np.log(max(feature_prob, 1e-10))

            probabilities[cls] = prob

        # Convert log probabilities back to regular probabilities
        # Subtract max for numerical stability
        max_log_prob = max(probabilities.values())
        probabilities = {k: exp(v - max_log_prob) for k, v in probabilities.items()}

        # Normalize
        total = sum(probabilities.values())
        probabilities = {k: v / total for k, v in probabilities.items()}

        return probabilities

    def predict_proba(self, X):
        """Predict class probabilities for samples"""
        X = np.array(X)
        results = []
        for x in X:
            results.append(self._predict_proba_single(x))
        return results

    def predict(self, X):
        """Predict class labels for samples"""
        probas = self.predict_proba(X)
        predictions = []
        for prob_dict in probas:
            predictions.append(max(prob_dict, key=prob_dict.get))
        return np.array(predictions)

    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)

