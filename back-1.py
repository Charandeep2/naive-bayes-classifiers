"""
Naive Bayes Implementation with Visualization
Supports both Numerical (Gaussian) and Categorical (Multinomial) features
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    def get_params(self, deep=True):
        """Get parameters for sklearn compatibility"""
        return {}
    
    def set_params(self, **params):
        """Set parameters for sklearn compatibility"""
        return self

    def predict_with_calculations(self, x, verbose=True):
        """
        Predict probability with detailed step-by-step calculations

        Parameters:
        -----------
        x : array-like
            Single sample to predict
        verbose : bool
            If True, print detailed calculations

        Returns:
        --------
        dict : Class probabilities
        """
        # Convert to numpy array and ensure it's 1D
        x = np.array(x)
        if x.ndim > 1:
            x = x.flatten()
        elif x.ndim == 0:
            x = np.array([x])

        # Ensure we have the right length
        if len(x) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {len(x)}")

        log_probs = {}

        if verbose:
            print("\n" + "="*70)
            print("STEP-BY-STEP NAIVE BAYES PROBABILITY CALCULATION")
            print("="*70)
            print("\nNaive Bayes Formula:")
            print("  P(Class|Features) proportional to P(Class) * P(Feature1|Class) * P(Feature2|Class) * ...")
            print("\n  Or using log probabilities (to avoid underflow):")
            print("  log(P(Class|Features)) = log(P(Class)) + sum of log(P(Feature_i|Class))")
            print("\n  Then normalize to get probabilities that sum to 1")
            print(f"\nSample Features:")
            for i, name in enumerate(self.feature_names):
                print(f"  {name}: {x[i]}")

        # Calculate for each class
        for cls in self.classes:
            if verbose:
                print(f"\n{'='*70}")
                print(f"CALCULATING PROBABILITY FOR CLASS: {cls}")
                print('='*70)

            # Step 1: Prior Probability
            prior_prob = self.class_probs[cls]
            log_prior = np.log(prior_prob)

            if verbose:
                print(f"\nStep 1: Prior Probability P(Class={cls})")
                print(f"  P(Class={cls}) = {prior_prob:.6f}")
                print(f"  log(P(Class={cls})) = {log_prior:.6f}")

            # Step 2: Feature Likelihoods
            feature_likelihoods = {}
            log_likelihood_sum = 0

            if verbose:
                print(f"\nStep 2: Calculating Likelihoods P(Feature|Class={cls})")

            for feature_idx in range(self.n_features):
                feature_value = x[feature_idx]
                feature_name = self.feature_names[feature_idx]

                if self.feature_types[feature_idx] == 'numerical':
                    # Convert to float if it's a string
                    try:
                        feature_value = float(feature_value)
                    except (ValueError, TypeError):
                        # If conversion fails, try to convert from string
                        if isinstance(feature_value, str):
                            try:
                                feature_value = float(feature_value)
                            except:
                                raise ValueError(f"Feature {feature_name} (numerical) has non-numeric value: {feature_value}")
                        else:
                            feature_value = float(feature_value)

                    # Gaussian PDF
                    params = self.feature_params[cls][feature_idx]
                    mean = params['mean']
                    std = params['std']

                    # Calculate Gaussian PDF
                    feature_prob = self._gaussian_pdf(feature_value, mean, std)
                    log_feature_prob = np.log(max(feature_prob, 1e-10))

                    if verbose:
                        # Calculate distance from mean (now feature_value is definitely a float)
                        distance_from_mean = abs(feature_value - mean)
                        z_score = (feature_value - mean) / std if std > 0 else 0

                        print(f"\n  Feature {feature_idx+1}: {feature_name} (Numerical - Gaussian Distribution)")
                        print(f"    Input Value: {feature_value}")
                        print(f"\n    [Gaussian] GAUSSIAN PARAMETERS:")
                        print(f"      Mean (mu): {mean:.6f}")
                        print(f"      Standard Deviation (sigma): {std:.6f}")
                        print(f"\n    [Distance] DISTANCE ANALYSIS:")
                        print(f"      Distance from Mean: |{feature_value:.6f} - {mean:.6f}| = {distance_from_mean:.6f}")
                        print(f"      Z-Score: ({feature_value:.6f} - {mean:.6f}) / {std:.6f} = {z_score:.3f}")

                        # Explain significance
                        if abs(z_score) < 1:
                            sig_text = "Very Close (within 1sigma)"
                            prob_explanation = "HIGH probability - value is typical for this class"
                        elif abs(z_score) < 2:
                            sig_text = "Close (within 2sigma)"
                            prob_explanation = "MODERATE probability - value is somewhat typical"
                        else:
                            sig_text = "Far (beyond 2sigma)"
                            prob_explanation = "LOW probability - value is unusual for this class"

                        print(f"      Significance: {sig_text}")
                        print(f"\n    [Info] IMPORTANCE OF MEAN AND STANDARD DEVIATION:")
                        print(f"      • MEAN (mu={mean:.2f}): Center of distribution - most likely value for this class")
                        print(f"        - Values closer to mean = higher probability")
                        print(f"        - Our value ({feature_value:.2f}) is {distance_from_mean:.2f} away from mean")
                        print(f"      • STANDARD DEVIATION (sigma={std:.2f}): Measure of spread/variability")
                        print(f"        - Smaller sigma = tighter distribution, values cluster around mean")
                        print(f"        - Larger sigma = wider distribution, more variation allowed")
                        print(f"        - 68% of values fall within mu±1sigma, 95% within mu±2sigma")
                        print(f"      • IMPACT: sigma={std:.2f} means typical range is [{mean-2*std:.2f}, {mean+2*std:.2f}]")

                        print(f"\n    [CALC] GAUSSIAN PROBABILITY CALCULATION:")
                        variance = std * std
                        exponent = -0.5 * ((feature_value - mean) / std) ** 2
                        coefficient = 1 / (std * sqrt(2 * pi))
                        print(f"      Formula: P(x|mu,sigma) = (1/sqrt(2pisigma^2)) * exp(-(x-mu)^2/(2sigma^2))")
                        print(f"      Step 1: Variance (sigma^2) = {std:.6f}^2 = {variance:.6f}")
                        print(f"      Step 2: Coefficient = 1/({std:.6f}*sqrt(2pi)) = {coefficient:.6f}")
                        print(f"      Step 3: Exponent = -0.5*(({feature_value:.6f}-{mean:.6f})/{std:.6f})^2 = {exponent:.6f}")
                        print(f"      Step 4: exp({exponent:.6f}) = {exp(exponent):.6f}")
                        print(f"      Step 5: PDF = {coefficient:.6f} * {exp(exponent):.6f} = {feature_prob:.6f}")
                        print(f"\n    [Result] PROBABILITY RESULT:")
                        print(f"      P({feature_value}|Class={cls}, mu={mean:.2f}, sigma={std:.2f}) = {feature_prob:.6f}")
                        print(f"      {prob_explanation}")
                        print(f"      log(P({feature_value}|Class={cls})) = {log_feature_prob:.6f}")

                    feature_likelihoods[feature_name] = {
                        'type': 'numerical',
                        'value': feature_value,
                        'mean': mean,
                        'std': std,
                        'probability': feature_prob,
                        'log_probability': log_feature_prob
                    }
                else:
                    # Categorical - ensure feature_value is a string
                    feature_value_str = str(feature_value)

                    # Categorical
                    params = self.feature_params[cls][feature_idx]
                    prob_dist = params['prob_dist']
                    feature_prob = prob_dist.get(feature_value_str, 1 / (len(prob_dist) + 1))
                    log_feature_prob = np.log(max(feature_prob, 1e-10))

                    if verbose:
                        print(f"\n  Feature {feature_idx+1}: {feature_name} (Categorical)")
                        print(f"    Input Value: {feature_value_str}")
                        print(f"\n    [Categorical] CATEGORICAL PROBABILITY CALCULATION:")
                        print(f"      Value in dataset: '{feature_value_str}'")

                        if feature_value_str in prob_dist:
                            print(f"      [OK] Found in training data for Class {cls}")
                            print(f"      P({feature_value_str}|Class={cls}) = {prob_dist[feature_value_str]:.6f} (from training)")
                        else:
                            print(f"      [WARN] Not seen in training data for Class {cls}")
                            print(f"      Using Laplace smoothing (add-1 smoothing)")
                            print(f"      P({feature_value_str}|Class={cls}) = {feature_prob:.6f} (smoothed)")

                        print(f"\n    [Values] AVAILABLE VALUES IN TRAINING (Class {cls}):")
                        sorted_probs = sorted(prob_dist.items(), key=lambda x: -x[1])
                        for val, prob in sorted_probs[:5]:  # Show top 5
                            marker = "[X]" if val == feature_value_str else "   "
                            print(f"      {marker} '{val}': {prob:.4f} ({prob*100:.2f}%)")

                        if len(prob_dist) > 5:
                            print(f"      ... and {len(prob_dist) - 5} more values")

                        print(f"\n    [Result] PROBABILITY RESULT:")
                        print(f"      P({feature_value_str}|Class={cls}) = {feature_prob:.6f}")
                        print(f"      log(P({feature_value_str}|Class={cls})) = {log_feature_prob:.6f}")

                    feature_likelihoods[feature_name] = {
                        'type': 'categorical',
                        'value': feature_value_str,
                        'probability': feature_prob,
                        'log_probability': log_feature_prob,
                        'prob_distribution': prob_dist
                    }

                log_likelihood_sum += log_feature_prob

            # Step 3: Combine (using log probabilities)
            log_posterior = log_prior + log_likelihood_sum

            if verbose:
                print(f"\nStep 3: Combine Prior and Likelihoods (using log space)")
                print(f"  log(P(Class={cls}|Features)) = log(P(Class={cls})) + sum of log(P(Feature_i|Class={cls}))")
                print(f"  log(P(Class={cls}|Features)) = {log_prior:.6f} + {log_likelihood_sum:.6f}")
                print(f"  log(P(Class={cls}|Features)) = {log_posterior:.6f}")

            log_probs[cls] = {
                'prior': prior_prob,
                'log_prior': log_prior,
                'feature_likelihoods': feature_likelihoods,
                'log_likelihood_sum': log_likelihood_sum,
                'log_posterior': log_posterior
            }

        # Step 4: Convert back from log space and normalize
        max_log_prob = max([calc['log_posterior'] for calc in log_probs.values()])

        if verbose:
            print(f"\n{'='*70}")
            print("STEP 4: NORMALIZATION")
            print('='*70)
            print(f"\nUnnormalized log probabilities:")
            for cls in self.classes:
                print(f"  log(P(Class={cls}|Features)) = {log_probs[cls]['log_posterior']:.6f}")
            print(f"\nMaximum log probability: {max_log_prob:.6f}")
            print(f"\nSubtracting maximum for numerical stability and converting from log:")

        final_probs = {}
        unnormalized_probs = {}

        for cls in self.classes:
            # Convert from log, subtracting max for stability
            unnormalized_prob = exp(log_probs[cls]['log_posterior'] - max_log_prob)
            unnormalized_probs[cls] = unnormalized_prob

            if verbose:
                print(f"  P(Class={cls}|Features) (unnormalized) = exp({log_probs[cls]['log_posterior']:.6f} - {max_log_prob:.6f}) = {unnormalized_prob:.6f}")

        # Normalize
        total = sum(unnormalized_probs.values())
        for cls in self.classes:
            final_probs[cls] = unnormalized_probs[cls] / total
            if verbose:
                print(f"  P(Class={cls}|Features) (normalized) = {unnormalized_probs[cls]:.6f} / {total:.6f} = {final_probs[cls]:.6f} ({final_probs[cls]*100:.2f}%)")

        if verbose:
            print(f"\n{'='*70}")
            print("FINAL PROBABILITIES")
            print('='*70)
            for cls in sorted(final_probs.keys(), key=lambda x: -final_probs[x]):
                print(f"  P(Class={cls}|Features) = {final_probs[cls]:.6f} ({final_probs[cls]*100:.2f}%)")

            predicted_class = max(final_probs, key=final_probs.get)
            print(f"\n{'='*70}")
            print(f"PREDICTED CLASS: {predicted_class}")
            print(f"Confidence: {final_probs[predicted_class]*100:.2f}%")
            print('='*70)

        return final_probs


def visualize_naive_bayes(nb_model, X, y, save_path=None):
    """
    Create comprehensive visualizations for Naive Bayes implementation
    Enhanced with multiple visualization types for better interpretation

    Parameters:
    -----------
    nb_model : NaiveBayes
        Trained Naive Bayes model
    X : array-like
        Training data
    y : array-like
        Target labels
    save_path : str, optional
        Path to save the figure
    """
    X = np.array(X)
    y = np.array(y)
    classes = list(nb_model.class_probs.keys())
    
    # Create multiple figure panels for comprehensive visualization
    print("\n[Viz] Generating comprehensive visualizations...")
    print("   This may take a moment...\n")
    
    # ===== FIGURE 1: Overview and Distribution Analysis =====
    fig1 = plt.figure(figsize=(20, 14))
    gs1 = fig1.add_gridspec(3, 4, hspace=0.4, wspace=0.4)
    
    # 1. Class Prior Probabilities (Bar Chart)
    ax1 = fig1.add_subplot(gs1[0, 0])
    probs = [nb_model.class_probs[cls] for cls in classes]
    bars = ax1.bar(range(len(classes)), probs, color=sns.color_palette("husl", len(classes)))
    ax1.set_xlabel('Classes', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Prior Probability P(Class)', fontsize=11, fontweight='bold')
    ax1.set_title('1. Prior Class Probabilities', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels([str(c) for c in classes])
    ax1.grid(axis='y', alpha=0.3)
    for i, (bar, prob) in enumerate(zip(bars, probs)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 2. Class Prior Probabilities (Pie Chart)
    ax2 = fig1.add_subplot(gs1[0, 1])
    colors = sns.color_palette("husl", len(classes))
    wedges, texts, autotexts = ax2.pie(probs, labels=[str(c) for c in classes], autopct='%1.2f%%',
                                      colors=colors, startangle=90)
    ax2.set_title('2. Class Distribution (Pie Chart)', fontsize=12, fontweight='bold')
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # 3. Feature Distribution Comparison (Box Plot for Numerical Features)
    numerical_features = [i for i, ft in enumerate(nb_model.feature_types) if ft == 'numerical']
    if numerical_features:
        ax3 = fig1.add_subplot(gs1[0, 2])
        feat_idx = numerical_features[0]
        data_for_box = []
        labels_for_box = []
        for cls in classes:
            cls_mask = y == cls
            cls_feature_values = pd.to_numeric(X[cls_mask, feat_idx], errors='coerce')
            data_for_box.append(cls_feature_values)
            labels_for_box.append(f'Class {cls}')
        
        bp = ax3.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(classes)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax3.set_xlabel('Classes', fontsize=11, fontweight='bold')
        ax3.set_ylabel(f'{nb_model.feature_names[feat_idx]}', fontsize=11, fontweight='bold')
        ax3.set_title('3. Feature Distribution (Box Plot)', fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)
    
    # 4. Violin Plot for Numerical Feature
    if numerical_features:
        ax4 = fig1.add_subplot(gs1[0, 3])
        feat_idx = numerical_features[0]
        data_dict = {str(cls): pd.to_numeric(X[y == cls, feat_idx], errors='coerce') for cls in classes}
        data_list = []
        class_list = []
        for cls, values in data_dict.items():
            data_list.extend(values[~np.isnan(values)])
            class_list.extend([cls] * len(values[~np.isnan(values)]))
        df_plot = pd.DataFrame({nb_model.feature_names[feat_idx]: data_list, 'Class': class_list})
        
        parts = ax4.violinplot([df_plot[df_plot['Class'] == str(cls)][nb_model.feature_names[feat_idx]].values 
                                for cls in classes], positions=range(len(classes)), showmeans=True)
        ax4.set_xticks(range(len(classes)))
        ax4.set_xticklabels([str(c) for c in classes])
        ax4.set_ylabel(f'{nb_model.feature_names[feat_idx]}', fontsize=11, fontweight='bold')
        ax4.set_title('4. Distribution Density (Violin Plot)', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)
    
    # 5. Scatter Plot for 2D Numerical Features (if available)
    if len(numerical_features) >= 2:
        ax5 = fig1.add_subplot(gs1[1, 0])
        feat1_idx = numerical_features[0]
        feat2_idx = numerical_features[1]
        for i, cls in enumerate(classes):
            cls_mask = y == cls
            x_vals = pd.to_numeric(X[cls_mask, feat1_idx], errors='coerce')
            y_vals = pd.to_numeric(X[cls_mask, feat2_idx], errors='coerce')
            ax5.scatter(x_vals, y_vals, label=f'Class {cls}', alpha=0.6, 
                       color=colors[i], s=50, edgecolors='black', linewidth=0.5)
            
            # Plot class mean
            mean_x = np.nanmean(x_vals)
            mean_y = np.nanmean(y_vals)
            ax5.scatter(mean_x, mean_y, marker='X', s=200, color=colors[i], 
                       edgecolors='black', linewidth=2, label=f'{cls} Mean', zorder=5)
        ax5.set_xlabel(nb_model.feature_names[feat1_idx], fontsize=11, fontweight='bold')
        ax5.set_ylabel(nb_model.feature_names[feat2_idx], fontsize=11, fontweight='bold')
        ax5.set_title('5. Class Separation (Scatter Plot)', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(alpha=0.3)
    else:
        ax5 = fig1.add_subplot(gs1[1, 0])
        ax5.text(0.5, 0.5, 'Need 2+ numerical\nfeatures for scatter plot', 
                ha='center', va='center', fontsize=12, transform=ax5.transAxes)
        ax5.set_title('5. Class Separation (Not Available)', fontsize=12, fontweight='bold')
        ax5.axis('off')
    
    # 6. Feature Statistics Comparison (Mean and Std)
    if numerical_features:
        ax6 = fig1.add_subplot(gs1[1, 1])
        feat_idx = numerical_features[0] if numerical_features else 0
        means = []
        stds = []
        class_names = []
        for cls in classes:
            params = nb_model.feature_params[cls][feat_idx]
            means.append(params['mean'])
            stds.append(params['std'])
            class_names.append(str(cls))
        
        x_pos = np.arange(len(class_names))
        width = 0.35
        bars1 = ax6.bar(x_pos - width/2, means, width, label='Mean (mu)', alpha=0.8, color='skyblue')
        bars2 = ax6.bar(x_pos + width/2, stds, width, label='Std Dev (sigma)', alpha=0.8, color='lightcoral')
        ax6.set_xlabel('Classes', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax6.set_title(f'6. Mean & Std Dev: {nb_model.feature_names[feat_idx]}', fontsize=12, fontweight='bold')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(class_names)
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)
    
    # 7. Class Probability Heatmap across Features
    ax7 = fig1.add_subplot(gs1[1, 2])
    heatmap_data = []
    for cls in classes:
        row = []
        for feat_idx in range(len(nb_model.feature_names)):
            if nb_model.feature_types[feat_idx] == 'numerical':
                # Use coefficient of variation as importance proxy
                params = nb_model.feature_params[cls][feat_idx]
                if abs(params['mean']) > 1e-6:
                    cv = params['std'] / abs(params['mean'])
                    row.append(cv)
                else:
                    row.append(params['std'])
            else:
                # For categorical, use entropy of distribution
                prob_dist = nb_model.feature_params[cls][feat_idx]['prob_dist']
                entropy = -sum(p * np.log(max(p, 1e-10)) for p in prob_dist.values())
                row.append(entropy)
        heatmap_data.append(row)
    
    sns.heatmap(heatmap_data, xticklabels=nb_model.feature_names, 
                yticklabels=[str(c) for c in classes], annot=True, fmt='.3f',
                cmap='YlOrRd', ax=ax7, cbar_kws={'label': 'Variability Measure'})
    ax7.set_title('7. Feature Variability by Class', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Features', fontsize=10, fontweight='bold')
    ax7.set_ylabel('Classes', fontsize=10, fontweight='bold')
    
    # 8. Feature Importance/Contribution Analysis
    ax8 = fig1.add_subplot(gs1[1, 3])
    # Calculate feature contribution based on class separation
    feature_importance = []
    for feat_idx in range(len(nb_model.feature_names)):
        if nb_model.feature_types[feat_idx] == 'numerical':
            # Measure separation between classes
            means_per_class = []
            for cls in classes:
                params = nb_model.feature_params[cls][feat_idx]
                means_per_class.append(params['mean'])
            separation = np.std(means_per_class) if len(means_per_class) > 1 else 0
            feature_importance.append(separation)
        else:
            # For categorical, measure distribution difference
            all_probs = []
            for cls in classes:
                prob_dist = nb_model.feature_params[cls][feat_idx]['prob_dist']
                all_probs.extend(list(prob_dist.values()))
            feature_importance.append(np.std(all_probs) if all_probs else 0)
    
    bars = ax8.barh(range(len(nb_model.feature_names)), feature_importance, 
                    color=sns.color_palette("viridis", len(nb_model.feature_names)))
    ax8.set_yticks(range(len(nb_model.feature_names)))
    ax8.set_yticklabels(nb_model.feature_names)
    ax8.set_xlabel('Importance Score', fontsize=11, fontweight='bold')
    ax8.set_title('8. Feature Importance Ranking', fontsize=12, fontweight='bold')
    ax8.grid(axis='x', alpha=0.3)
    for i, (bar, imp) in enumerate(zip(bars, feature_importance)):
        ax8.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{imp:.3f}', va='center', fontsize=9)
    
    # 9-12. Individual Feature Detailed Visualizations
    n_features = len(nb_model.feature_names)
    for plot_idx, feat_idx in enumerate(range(min(4, n_features))):
        row = 2
        col = plot_idx
        
        ax = fig1.add_subplot(gs1[row, col])
        
        if nb_model.feature_types[feat_idx] == 'numerical':
            # Enhanced Gaussian distribution visualization
            for i, cls in enumerate(classes):
                params = nb_model.feature_params[cls][feat_idx]
                mean = params['mean']
                std = params['std']
                
                cls_mask = y == cls
                cls_feature_values = pd.to_numeric(X[cls_mask, feat_idx], errors='coerce')
                x_min = np.nanmin(cls_feature_values) - 3*std
                x_max = np.nanmax(cls_feature_values) + 3*std
                x_curve = np.linspace(x_min, x_max, 200)
                
                y_curve = [nb_model._gaussian_pdf(x, mean, std) for x in x_curve]
                
                # Plot histogram
                ax.hist(cls_feature_values, bins=20, alpha=0.3, density=True,
                       label=f'Class {cls} (data)', color=colors[i])
                # Plot Gaussian curve
                ax.plot(x_curve, y_curve, linewidth=2.5, label=f'Class {cls} (Gaussian)', color=colors[i])
                # Mark mean
                ax.axvline(mean, color=colors[i], linestyle='--', alpha=0.7, linewidth=1.5, label=f'{cls} Mean')
                # Mark ±1sigma, ±2sigma
                ax.axvline(mean - std, color=colors[i], linestyle=':', alpha=0.5, linewidth=1)
                ax.axvline(mean + std, color=colors[i], linestyle=':', alpha=0.5, linewidth=1)
            
            ax.set_xlabel(nb_model.feature_names[feat_idx], fontsize=10, fontweight='bold')
            ax.set_ylabel('Probability Density', fontsize=10, fontweight='bold')
            ax.set_title(f'F{feat_idx+1}: {nb_model.feature_names[feat_idx]}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=8, loc='best')
            ax.grid(alpha=0.3)
        else:
            # Enhanced categorical visualization
            all_values = np.unique(X[:, feat_idx])
            x_pos = np.arange(len(all_values))
            width = 0.8 / len(classes)
            
            for i, cls in enumerate(classes):
                prob_dist = nb_model.feature_params[cls][feat_idx]['prob_dist']
                probs = [prob_dist.get(str(val), 0) for val in all_values]
                ax.bar(x_pos + i*width, probs, width, label=f'Class {cls}',
                      color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xlabel(nb_model.feature_names[feat_idx], fontsize=10, fontweight='bold')
            ax.set_ylabel('P(Value|Class)', fontsize=10, fontweight='bold')
            ax.set_title(f'F{feat_idx+1}: {nb_model.feature_names[feat_idx]}', fontsize=11, fontweight='bold')
            ax.set_xticks(x_pos + width * (len(classes) - 1) / 2)
            ax.set_xticklabels([str(v) for v in all_values], rotation=45, ha='right', fontsize=8)
            ax.legend(fontsize=8)
            ax.grid(axis='y', alpha=0.3)
    
    fig1.suptitle('Naive Bayes Comprehensive Visualization - Panel 1', fontsize=16, fontweight='bold', y=0.995)
    
    # ===== FIGURE 2: Additional Analysis =====
    fig2 = plt.figure(figsize=(18, 12))
    gs2 = fig2.add_gridspec(2, 3, hspace=0.4, wspace=0.4)
    
    # Probability Distribution Comparison
    ax21 = fig2.add_subplot(gs2[0, 0])
    # Show how probability varies across feature values
    if numerical_features:
        feat_idx = numerical_features[0]
        params_per_class = {cls: nb_model.feature_params[cls][feat_idx] for cls in classes}
        x_range = []
        for cls in classes:
            cls_mask = y == cls
            cls_feature_values = pd.to_numeric(X[cls_mask, feat_idx], errors='coerce')
            x_range.extend([np.nanmin(cls_feature_values), np.nanmax(cls_feature_values)])
        x_plot = np.linspace(min(x_range), max(x_range), 300)
        
        for i, cls in enumerate(classes):
            params = params_per_class[cls]
            y_probs = [nb_model._gaussian_pdf(x, params['mean'], params['std']) for x in x_plot]
            ax21.plot(x_plot, y_probs, linewidth=2.5, label=f'Class {cls}', color=colors[i])
            ax21.fill_between(x_plot, y_probs, alpha=0.2, color=colors[i])
        ax21.set_xlabel(nb_model.feature_names[feat_idx], fontsize=11, fontweight='bold')
        ax21.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
        ax21.set_title('1. Probability Distribution Overlap', fontsize=12, fontweight='bold')
        ax21.legend()
        ax21.grid(alpha=0.3)
    
    # Confusion Matrix (if predictions available)
    try:
        from sklearn.metrics import confusion_matrix
        predictions = nb_model.predict(X)
        cm = confusion_matrix(y, predictions)
        ax22 = fig2.add_subplot(gs2[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[str(c) for c in classes],
                   yticklabels=[str(c) for c in classes], ax=ax22, cbar_kws={'label': 'Count'})
        ax22.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
        ax22.set_ylabel('Actual Class', fontsize=11, fontweight='bold')
        ax22.set_title('2. Confusion Matrix', fontsize=12, fontweight='bold')
    except:
        ax22 = fig2.add_subplot(gs2[0, 1])
        ax22.text(0.5, 0.5, 'Confusion Matrix\n(Available after prediction)', 
                 ha='center', va='center', fontsize=12, transform=ax22.transAxes)
        ax22.set_title('2. Confusion Matrix', fontsize=12, fontweight='bold')
        ax22.axis('off')
    
    # Feature Correlation (for numerical features)
    if len(numerical_features) >= 2:
        ax23 = fig2.add_subplot(gs2[0, 2])
        feat1_idx = numerical_features[0]
        feat2_idx = numerical_features[1]
        # Create correlation heatmap for numerical features
        num_feat_data = {}
        for feat_idx in numerical_features[:min(5, len(numerical_features))]:
            num_feat_data[nb_model.feature_names[feat_idx]] = pd.to_numeric(X[:, feat_idx], errors='coerce')
        df_num = pd.DataFrame(num_feat_data)
        corr_matrix = df_num.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                   square=True, ax=ax23, cbar_kws={'label': 'Correlation'})
        ax23.set_title('3. Numerical Features Correlation', fontsize=12, fontweight='bold')
    else:
        ax23 = fig2.add_subplot(gs2[0, 2])
        ax23.text(0.5, 0.5, 'Need 2+ numerical\nfeatures for correlation', 
                 ha='center', va='center', fontsize=12, transform=ax23.transAxes)
        ax23.set_title('3. Feature Correlation (Not Available)', fontsize=12, fontweight='bold')
        ax23.axis('off')
    
    # Class Separation Metrics
    ax24 = fig2.add_subplot(gs2[1, 0])
    if numerical_features:
        separation_scores = []
        feat_names_short = []
        for feat_idx in numerical_features:
            means_per_class = [nb_model.feature_params[cls][feat_idx]['mean'] for cls in classes]
            stds_per_class = [nb_model.feature_params[cls][feat_idx]['std'] for cls in classes]
            mean_diff = np.std(means_per_class) if len(means_per_class) > 1 else 0
            avg_std = np.mean(stds_per_class)
            separation_score = mean_diff / (avg_std + 1e-6) if avg_std > 0 else 0
            separation_scores.append(separation_score)
            feat_names_short.append(nb_model.feature_names[feat_idx][:15])
        
        bars = ax24.barh(range(len(feat_names_short)), separation_scores,
                        color=sns.color_palette("plasma", len(feat_names_short)))
        ax24.set_yticks(range(len(feat_names_short)))
        ax24.set_yticklabels(feat_names_short, fontsize=9)
        ax24.set_xlabel('Separation Score', fontsize=11, fontweight='bold')
        ax24.set_title('4. Class Separation by Feature', fontsize=12, fontweight='bold')
        ax24.grid(axis='x', alpha=0.3)
        for i, (bar, score) in enumerate(zip(bars, separation_scores)):
            ax24.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{score:.3f}', va='center', fontsize=9)
    
    # Categorical Feature Comparison
    categorical_features = [i for i, ft in enumerate(nb_model.feature_types) if ft == 'categorical']
    if categorical_features:
        ax25 = fig2.add_subplot(gs2[1, 1])
        feat_idx = categorical_features[0]
        all_values = np.unique(X[:, feat_idx])
        
        # Stacked bar chart showing class probabilities
        bottom = np.zeros(len(all_values))
        for i, cls in enumerate(classes):
            prob_dist = nb_model.feature_params[cls][feat_idx]['prob_dist']
            probs = [prob_dist.get(str(val), 0) for val in all_values]
            ax25.bar(range(len(all_values)), probs, bottom=bottom, label=f'Class {cls}',
                    color=colors[i], alpha=0.8, edgecolor='black', linewidth=0.5)
            bottom += probs
        
        ax25.set_xlabel('Categories', fontsize=11, fontweight='bold')
        ax25.set_ylabel('Probability', fontsize=11, fontweight='bold')
        ax25.set_title(f'5. {nb_model.feature_names[feat_idx]} (Stacked)', fontsize=12, fontweight='bold')
        ax25.set_xticks(range(len(all_values)))
        ax25.set_xticklabels([str(v) for v in all_values], rotation=45, ha='right', fontsize=9)
        ax25.legend(fontsize=9)
        ax25.grid(axis='y', alpha=0.3)
    
    # Model Performance Summary
    ax26 = fig2.add_subplot(gs2[1, 2])
    try:
        accuracy = nb_model.score(X, y)
        predictions = nb_model.predict(X)
        from sklearn.metrics import precision_score, recall_score, f1_score
        precision = precision_score(y, predictions, average='weighted', zero_division=0)
        recall = recall_score(y, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y, predictions, average='weighted', zero_division=0)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        values = [accuracy, precision, recall, f1]
        bars = ax26.bar(metrics, values, color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'], alpha=0.8)
        ax26.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax26.set_title('6. Model Performance Metrics', fontsize=12, fontweight='bold')
        ax26.set_ylim([0, 1.1])
        ax26.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, values):
            ax26.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    except:
        ax26.text(0.5, 0.5, 'Performance metrics\nwill be calculated', 
                 ha='center', va='center', fontsize=12, transform=ax26.transAxes)
        ax26.set_title('6. Model Performance Metrics', fontsize=12, fontweight='bold')
        ax26.axis('off')
    
    fig2.suptitle('Naive Bayes Comprehensive Visualization - Panel 2', fontsize=16, fontweight='bold', y=0.995)
    
    # ===== FIGURE 3: Advanced Analysis and Interpretations =====
    fig3 = plt.figure(figsize=(20, 14))
    gs3 = fig3.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    # Get predictions for analysis
    try:
        predictions = nb_model.predict(X)
        prediction_probas = nb_model.predict_proba(X)
    except:
        predictions = None
        prediction_probas = None
    
    # 1. Prediction Probability Distribution
    ax31 = fig3.add_subplot(gs3[0, 0])
    if prediction_probas:
        # Histogram of maximum probabilities (confidence scores)
        max_probs = [max(probs.values()) for probs in prediction_probas]
        ax31.hist(max_probs, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax31.axvline(np.mean(max_probs), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {np.mean(max_probs):.3f}')
        ax31.axvline(np.median(max_probs), color='green', linestyle='--', linewidth=2, 
                    label=f'Median: {np.median(max_probs):.3f}')
        ax31.set_xlabel('Maximum Prediction Probability (Confidence)', fontsize=11, fontweight='bold')
        ax31.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax31.set_title('1. Prediction Confidence Distribution', fontsize=12, fontweight='bold')
        ax31.legend()
        ax31.grid(alpha=0.3)
    else:
        ax31.text(0.5, 0.5, 'Prediction probabilities\nnot available', 
                 ha='center', va='center', fontsize=12, transform=ax31.transAxes)
        ax31.set_title('1. Prediction Confidence Distribution', fontsize=12, fontweight='bold')
        ax31.axis('off')
    
    # 2. Class Probability Distributions (for each class)
    ax32 = fig3.add_subplot(gs3[0, 1])
    if prediction_probas:
        for i, cls in enumerate(classes):
            class_probs = [probs.get(cls, 0) for probs in prediction_probas]
            ax32.hist(class_probs, bins=25, alpha=0.6, label=f'Class {cls}', 
                     color=colors[i], edgecolor='black')
        ax32.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
        ax32.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax32.set_title('2. Class Probability Distributions', fontsize=12, fontweight='bold')
        ax32.legend()
        ax32.grid(alpha=0.3)
    else:
        ax32.text(0.5, 0.5, 'Prediction probabilities\nnot available', 
                 ha='center', va='center', fontsize=12, transform=ax32.transAxes)
        ax32.set_title('2. Class Probability Distributions', fontsize=12, fontweight='bold')
        ax32.axis('off')
    
    # 3. ROC Curve (for binary classification) or Multi-class ROC
    ax33 = fig3.add_subplot(gs3[0, 2])
    try:
        if len(classes) == 2:
            # Binary classification - standard ROC
            from sklearn.metrics import roc_curve, auc, roc_auc_score
            probas = [probs.get(classes[1], 0) for probs in prediction_probas]
            y_binary = (y == classes[1]).astype(int)
            fpr, tpr, thresholds = roc_curve(y_binary, probas)
            roc_auc = auc(fpr, tpr)
            
            ax33.plot(fpr, tpr, color='darkorange', lw=2, 
                     label=f'ROC curve (AUC = {roc_auc:.3f})')
            ax33.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            ax33.set_xlim([0.0, 1.0])
            ax33.set_ylim([0.0, 1.05])
            ax33.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
            ax33.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
            ax33.set_title('3. ROC Curve', fontsize=12, fontweight='bold')
            ax33.legend(loc="lower right")
            ax33.grid(alpha=0.3)
        elif len(classes) > 2:
            # Multi-class ROC
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y, classes=list(classes))
            n_classes = len(classes)
            
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i, cls in enumerate(classes):
                probas = [probs.get(cls, 0) for probs in prediction_probas]
                fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probas)
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Plot all classes
            for i, cls in enumerate(classes):
                ax33.plot(fpr[i], tpr[i], lw=2, color=colors[i],
                         label=f'{cls} (AUC = {roc_auc[i]:.3f})')
            
            ax33.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
            ax33.set_xlim([0.0, 1.0])
            ax33.set_ylim([0.0, 1.05])
            ax33.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
            ax33.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
            ax33.set_title('3. Multi-class ROC Curves', fontsize=12, fontweight='bold')
            ax33.legend(loc="lower right", fontsize=8)
            ax33.grid(alpha=0.3)
    except Exception as e:
        ax33.text(0.5, 0.5, f'ROC curve\ncalculation\nfailed: {str(e)[:30]}', 
                 ha='center', va='center', fontsize=10, transform=ax33.transAxes)
        ax33.set_title('3. ROC Curve', fontsize=12, fontweight='bold')
        ax33.axis('off')
    
    # 4. Precision-Recall Curve (for binary) or Multi-class
    ax34 = fig3.add_subplot(gs3[1, 0])
    try:
        if len(classes) == 2:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            probas = [probs.get(classes[1], 0) for probs in prediction_probas]
            y_binary = (y == classes[1]).astype(int)
            precision, recall, thresholds = precision_recall_curve(y_binary, probas)
            avg_precision = average_precision_score(y_binary, probas)
            
            ax34.plot(recall, precision, color='darkorange', lw=2,
                     label=f'PR curve (AP = {avg_precision:.3f})')
            ax34.set_xlabel('Recall', fontsize=11, fontweight='bold')
            ax34.set_ylabel('Precision', fontsize=11, fontweight='bold')
            ax34.set_title('4. Precision-Recall Curve', fontsize=12, fontweight='bold')
            ax34.legend()
            ax34.grid(alpha=0.3)
        else:
            from sklearn.metrics import precision_recall_curve, average_precision_score
            from sklearn.preprocessing import label_binarize
            y_bin = label_binarize(y, classes=list(classes))
            
            precision = dict()
            recall = dict()
            avg_precision = dict()
            
            for i, cls in enumerate(classes):
                probas = [probs.get(cls, 0) for probs in prediction_probas]
                precision[i], recall[i], _ = precision_recall_curve(y_bin[:, i], probas)
                avg_precision[i] = average_precision_score(y_bin[:, i], probas)
            
            for i, cls in enumerate(classes):
                ax34.plot(recall[i], precision[i], lw=2, color=colors[i],
                         label=f'{cls} (AP = {avg_precision[i]:.3f})')
            
            ax34.set_xlabel('Recall', fontsize=11, fontweight='bold')
            ax34.set_ylabel('Precision', fontsize=11, fontweight='bold')
            ax34.set_title('4. Multi-class Precision-Recall', fontsize=12, fontweight='bold')
            ax34.legend(fontsize=8)
            ax34.grid(alpha=0.3)
    except Exception as e:
        ax34.text(0.5, 0.5, f'PR curve\ncalculation\nfailed', 
                 ha='center', va='center', fontsize=10, transform=ax34.transAxes)
        ax34.set_title('4. Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax34.axis('off')
    
    # 5. Error Analysis - Misclassification by Confidence
    ax35 = fig3.add_subplot(gs3[1, 1])
    if predictions is not None and prediction_probas:
        correct = predictions == y
        incorrect = ~correct
        
        correct_conf = [max(probs.values()) for i, probs in enumerate(prediction_probas) if correct[i]]
        incorrect_conf = [max(probs.values()) for i, probs in enumerate(prediction_probas) if incorrect[i]]
        
        if len(correct_conf) > 0:
            ax35.hist(correct_conf, bins=20, alpha=0.6, label='Correct', 
                     color='green', edgecolor='black')
        if len(incorrect_conf) > 0:
            ax35.hist(incorrect_conf, bins=20, alpha=0.6, label='Incorrect', 
                     color='red', edgecolor='black')
        
        ax35.set_xlabel('Prediction Confidence', fontsize=11, fontweight='bold')
        ax35.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax35.set_title('5. Error Analysis: Confidence vs Accuracy', fontsize=12, fontweight='bold')
        ax35.legend()
        ax35.grid(alpha=0.3)
    else:
        ax35.text(0.5, 0.5, 'Error analysis\nrequires predictions', 
                 ha='center', va='center', fontsize=12, transform=ax35.transAxes)
        ax35.set_title('5. Error Analysis', fontsize=12, fontweight='bold')
        ax35.axis('off')
    
    # 6. Feature Importance via Mutual Information or Correlation
    ax36 = fig3.add_subplot(gs3[1, 2])
    try:
        from sklearn.feature_selection import mutual_info_classif
        
        # Convert categorical features to numeric for MI calculation
        X_numeric = []
        for feat_idx in range(len(nb_model.feature_names)):
            if nb_model.feature_types[feat_idx] == 'numerical':
                X_numeric.append(pd.to_numeric(X[:, feat_idx], errors='coerce'))
            else:
                # Encode categorical as numeric
                unique_vals = np.unique(X[:, feat_idx])
                encoding = {val: i for i, val in enumerate(unique_vals)}
                encoded = np.array([encoding.get(str(val), 0) for val in X[:, feat_idx]])
                X_numeric.append(encoded)
        
        X_mi = np.column_stack(X_numeric)
        # Encode target
        unique_y = np.unique(y)
        y_encoding = {val: i for i, val in enumerate(unique_y)}
        y_encoded = np.array([y_encoding.get(val, 0) for val in y])
        
        mi_scores = mutual_info_classif(X_mi, y_encoded, random_state=42)
        
        bars = ax36.barh(range(len(nb_model.feature_names)), mi_scores,
                        color=sns.color_palette("muted", len(nb_model.feature_names)))
        ax36.set_yticks(range(len(nb_model.feature_names)))
        ax36.set_yticklabels([name[:20] for name in nb_model.feature_names], fontsize=9)
        ax36.set_xlabel('Mutual Information Score', fontsize=11, fontweight='bold')
        ax36.set_title('6. Feature Importance (MI)', fontsize=12, fontweight='bold')
        ax36.grid(axis='x', alpha=0.3)
        for i, (bar, score) in enumerate(zip(bars, mi_scores)):
            ax36.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                     f'{score:.3f}', va='center', fontsize=9)
    except Exception as e:
        # Fallback to simple correlation
        if numerical_features:
            correlations = []
            for feat_idx in numerical_features:
                feat_vals = pd.to_numeric(X[:, feat_idx], errors='coerce')
                y_encoded = pd.Categorical(y).codes
                corr = np.abs(np.corrcoef(feat_vals[~np.isnan(feat_vals)], 
                                         y_encoded[~np.isnan(feat_vals)])[0, 1])
                correlations.append(corr if not np.isnan(corr) else 0)
            
            feat_names_mi = [nb_model.feature_names[i] for i in numerical_features]
            bars = ax36.barh(range(len(feat_names_mi)), correlations,
                            color=sns.color_palette("muted", len(feat_names_mi)))
            ax36.set_yticks(range(len(feat_names_mi)))
            ax36.set_yticklabels([name[:20] for name in feat_names_mi], fontsize=9)
            ax36.set_xlabel('|Correlation| with Target', fontsize=11, fontweight='bold')
            ax36.set_title('6. Feature-Target Correlation', fontsize=12, fontweight='bold')
            ax36.grid(axis='x', alpha=0.3)
        else:
            ax36.text(0.5, 0.5, 'Feature importance\ncalculation failed', 
                     ha='center', va='center', fontsize=12, transform=ax36.transAxes)
            ax36.set_title('6. Feature Importance', fontsize=12, fontweight='bold')
            ax36.axis('off')
    
    # 7. Prediction Probability Heatmap (Sample × Class)
    ax37 = fig3.add_subplot(gs3[2, 0])
    if prediction_probas:
        # Show first 50 samples (or all if fewer)
        n_samples_show = min(50, len(prediction_probas))
        prob_matrix = np.array([[probs.get(cls, 0) for cls in classes] 
                               for probs in prediction_probas[:n_samples_show]])
        
        sns.heatmap(prob_matrix, xticklabels=[str(c) for c in classes],
                   yticklabels=range(1, n_samples_show+1) if n_samples_show <= 20 else False,
                   annot=False, fmt='.2f', cmap='YlGnBu', ax=ax37,
                   cbar_kws={'label': 'Probability'})
        ax37.set_xlabel('Classes', fontsize=11, fontweight='bold')
        ax37.set_ylabel(f'Samples (first {n_samples_show})', fontsize=11, fontweight='bold')
        ax37.set_title('7. Prediction Probabilities Heatmap', fontsize=12, fontweight='bold')
    else:
        ax37.text(0.5, 0.5, 'Prediction probabilities\nnot available', 
                 ha='center', va='center', fontsize=12, transform=ax37.transAxes)
        ax37.set_title('7. Prediction Probabilities Heatmap', fontsize=12, fontweight='bold')
        ax37.axis('off')
    
    # 8. Class Distribution by Confidence Levels
    ax38 = fig3.add_subplot(gs3[2, 1])
    if predictions is not None and prediction_probas:
        confidence_levels = ['Low (<0.6)', 'Medium (0.6-0.8)', 'High (>0.8)']
        class_conf_matrix = {cls: [0, 0, 0] for cls in classes}
        
        for i, probs in enumerate(prediction_probas):
            pred_cls = predictions[i]
            max_prob = max(probs.values())
            if max_prob < 0.6:
                class_conf_matrix[pred_cls][0] += 1
            elif max_prob < 0.8:
                class_conf_matrix[pred_cls][1] += 1
            else:
                class_conf_matrix[pred_cls][2] += 1
        
        x = np.arange(len(classes))
        width = 0.25
        for i, conf_level in enumerate(confidence_levels):
            values = [class_conf_matrix[cls][i] for cls in classes]
            ax38.bar(x + i*width, values, width, label=conf_level, alpha=0.8)
        
        ax38.set_xlabel('Predicted Class', fontsize=11, fontweight='bold')
        ax38.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax38.set_title('8. Predictions by Confidence Level', fontsize=12, fontweight='bold')
        ax38.set_xticks(x + width)
        ax38.set_xticklabels([str(c) for c in classes])
        ax38.legend(fontsize=9)
        ax38.grid(axis='y', alpha=0.3)
    else:
        ax38.text(0.5, 0.5, 'Predictions\nnot available', 
                 ha='center', va='center', fontsize=12, transform=ax38.transAxes)
        ax38.set_title('8. Predictions by Confidence Level', fontsize=12, fontweight='bold')
        ax38.axis('off')
    
    # 9. Cumulative Distribution of Prediction Confidence
    ax39 = fig3.add_subplot(gs3[2, 2])
    if prediction_probas:
        max_probs = np.array([max(probs.values()) for probs in prediction_probas])
        sorted_probs = np.sort(max_probs)
        cumulative = np.arange(1, len(sorted_probs) + 1) / len(sorted_probs)
        
        ax39.plot(sorted_probs, cumulative, linewidth=2.5, color='steelblue')
        ax39.axvline(np.median(max_probs), color='red', linestyle='--', 
                    label=f'Median: {np.median(max_probs):.3f}')
        ax39.fill_between(sorted_probs, cumulative, alpha=0.3, color='steelblue')
        ax39.set_xlabel('Prediction Confidence', fontsize=11, fontweight='bold')
        ax39.set_ylabel('Cumulative Proportion', fontsize=11, fontweight='bold')
        ax39.set_title('9. Cumulative Confidence Distribution', fontsize=12, fontweight='bold')
        ax39.legend()
        ax39.grid(alpha=0.3)
        ax39.set_xlim([0, 1])
        ax39.set_ylim([0, 1])
    else:
        ax39.text(0.5, 0.5, 'Prediction probabilities\nnot available', 
                 ha='center', va='center', fontsize=12, transform=ax39.transAxes)
        ax39.set_title('9. Cumulative Confidence Distribution', fontsize=12, fontweight='bold')
        ax39.axis('off')
    
    fig3.suptitle('Naive Bayes Advanced Analysis - Panel 3', fontsize=16, fontweight='bold', y=0.995)
    
    # Save if path provided
    if save_path:
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        fig1.savefig(f'{base_path}_panel1.png', dpi=300, bbox_inches='tight')
        fig2.savefig(f'{base_path}_panel2.png', dpi=300, bbox_inches='tight')
        fig3.savefig(f'{base_path}_panel3.png', dpi=300, bbox_inches='tight')
        print(f"\n[OK] Visualizations saved:")
        print(f"   - Panel 1: {base_path}_panel1.png")
        print(f"   - Panel 2: {base_path}_panel2.png")
        print(f"   - Panel 3: {base_path}_panel3.png")
    
    print("\n[OK] All visualizations generated successfully!")
    print("   Displaying graphs...\n")
    
    plt.show()


def print_model_summary(nb_model):
    """Print a detailed summary of the trained model"""
    print("\n" + "="*70)
    print("NAIVE BAYES MODEL SUMMARY")
    print("="*70)

    print(f"\nNumber of Features: {nb_model.n_features}")
    print(f"Feature Types: {dict(enumerate(nb_model.feature_types))}")
    print(f"\nClasses: {list(nb_model.classes)}")

    print("\n" + "-"*70)
    print("PRIOR PROBABILITIES P(Class):")
    print("-"*70)
    for cls in nb_model.classes:
        print(f"  P(Class={cls}) = {nb_model.class_probs[cls]:.4f}")
        print(f"    (This means {nb_model.class_probs[cls]*100:.2f}% of training samples belong to class '{cls}')")

    print("\n" + "-"*70)
    print("FEATURE PARAMETERS:")
    print("-"*70)

    for feat_idx in range(nb_model.n_features):
        feat_name = nb_model.feature_names[feat_idx]
        feat_type = nb_model.feature_types[feat_idx]
        print(f"\n{'='*70}")
        print(f"Feature {feat_idx+1}: {feat_name} ({feat_type})")
        print('='*70)

        for cls in nb_model.classes:
            params = nb_model.feature_params[cls][feat_idx]
            if feat_type == 'numerical':
                mean = params['mean']
                std = params['std']
                print(f"\n  [Class] Class: {cls}")
                print(f"    Mean (mu) = {mean:.6f}")
                print(f"    Standard Deviation (sigma) = {std:.6f}")
                print(f"    Variance (sigma^2) = {std*std:.6f}")

                # Calculate range
                range_1sigma = (mean - std, mean + std)
                range_2sigma = (mean - 2*std, mean + 2*std)
                range_3sigma = (mean - 3*std, mean + 3*std)

                print(f"\n    [Ranges] DISTRIBUTION RANGES (for Class {cls}):")
                print(f"      • 68% of values fall within: [{range_1sigma[0]:.4f}, {range_1sigma[1]:.4f}] (mu ± 1sigma)")
                print(f"      • 95% of values fall within: [{range_2sigma[0]:.4f}, {range_2sigma[1]:.4f}] (mu ± 2sigma)")
                print(f"      • 99.7% of values fall within: [{range_3sigma[0]:.4f}, {range_3sigma[1]:.4f}] (mu ± 3sigma)")

                print(f"\n    [Info] INTERPRETATION:")
                print(f"      • MEAN (mu={mean:.4f}):")
                print(f"        - Most representative value for this class")
                print(f"        - Peak of the Gaussian bell curve")
                print(f"        - New values close to {mean:.2f} will have HIGH probability")
                print(f"      • STANDARD DEVIATION (sigma={std:.4f}):")
                if std < 1.0:
                    spread_desc = "VERY TIGHT - values cluster very close to mean"
                elif std < 2.0:
                    spread_desc = "TIGHT - values stay close to mean"
                elif std < 5.0:
                    spread_desc = "MODERATE - moderate spread around mean"
                else:
                    spread_desc = "WIDE - values can vary significantly"

                print(f"        - Spread indicator: {spread_desc}")
                print(f"        - Smaller sigma ({std:.2f}) = more precise, values must be closer to mean")
                print(f"        - Larger sigma ({std:.2f}) = more flexible, allows wider range of values")
                if abs(mean) > 1e-6:
                    cv = std/abs(mean)*100
                    print(f"        - Coefficient of Variation = {cv:.2f}% (relative spread)")
                else:
                    print(f"        - Coefficient of Variation = N/A (mean is too close to zero)")

                # Compare across classes if multiple classes
                if len(nb_model.classes) > 1:
                    print(f"\n    [Compare] COMPARISON ACROSS CLASSES:")
                    all_means = []
                    all_stds = []
                    for c in nb_model.classes:
                        p = nb_model.feature_params[c][feat_idx]
                        all_means.append(p['mean'])
                        all_stds.append(p['std'])

                    print(f"      Mean values: {dict(zip(nb_model.classes, [f'{m:.4f}' for m in all_means]))}")
                    print(f"      Std Dev values: {dict(zip(nb_model.classes, [f'{s:.4f}' for s in all_stds]))}")
                    if max(all_means) - min(all_means) > 2 * max(all_stds):
                        print(f"      [OK] Classes are WELL SEPARATED (means differ significantly)")
                    else:
                        print(f"      [WARN] Classes have OVERLAPPING distributions (harder to distinguish)")
            else:
                print(f"\n  [Class] Class: {cls}")
                prob_dist = params['prob_dist']
                print(f"    Probability Distribution:")
                sorted_items = sorted(prob_dist.items(), key=lambda x: -x[1])
                for val, prob in sorted_items[:5]:  # Top 5
                    bar = "#" * int(prob * 50)
                    print(f"      P({val}|Class={cls}) = {prob:.4f} ({prob*100:.2f}%) {bar}")

    print("\n" + "="*70)
    print("[KEY INSIGHTS]")
    print("="*70)
    print("\n1. PRIOR PROBABILITIES: Show how common each class is in the dataset")
    print("2. MEAN (mu): For numerical features, indicates the typical/expected value for each class")
    print("3. STANDARD DEVIATION (sigma): Shows how much variation is allowed:")
    print("   - Small sigma = strict requirement, values must be close to mean")
    print("   - Large sigma = flexible, accepts wider range of values")
    print("4. The Gaussian formula uses both mu and sigma to calculate probability:")
    print("   P(x|mu,sigma) = (1/sqrt(2pisigma^2)) * exp(-(x-mu)^2/(2sigma^2))")
    print("   - Closer to mean = higher probability")
    print("   - Smaller sigma = steeper curve, more selective")
    print("\n" + "="*70 + "\n")


def get_user_dataset():
    """
    Interactive function to get dataset from user
    Supports multiple input methods
    """
    print("\n" + "="*70)
    print("DATASET INPUT OPTIONS")
    print("="*70)
    print("\n1. Enter data manually (paste or type row by row)")
    print("2. Load from CSV file (recommended for large datasets)")
    print("3. Use example dataset")

    choice = input("\nEnter your choice (1/2/3): ").strip()

    if choice == "1":
        return get_manual_input()
    elif choice == "2":
        return get_csv_input()
    elif choice == "3":
        return get_example_dataset()
    else:
        print("Invalid choice. Using example dataset...")
        return get_example_dataset()


def get_manual_input():
    """Get dataset through manual input"""
    print("\n--- Manual Data Input ---")
    print("The model will automatically detect numerical vs categorical features.")
    print("\nYou can enter data in two ways:")
    print("  1. Paste all data at once (rows separated by newlines)")
    print("  2. Enter data row by row (traditional method)")

    input_method = input("\nChoose input method (1=paste all, 2=row by row): ").strip()

    # Method 1: Paste all data at once
    if input_method == "1":
        print("\n" + "-"*70)
        print("PASTE DATA METHOD")
        print("-"*70)
        print("\nEnter your data:")
        print("Format: Each line should be: feature1,feature2,...,featureN,label")
        print("Example:")
        print("  10,5,High,Medium,Success")
        print("  8,3,Low,Low,Failure")
        print("  12,6,High,High,Success")
        print("\nPaste your data below (press Enter twice when finished):")

        lines = []
        empty_count = 0
        while True:
            line = input()
            if not line.strip():
                empty_count += 1
                if empty_count >= 2:  # Two empty lines = done
                    break
            else:
                empty_count = 0
                lines.append(line.strip())

        if not lines:
            print("No data entered. Using example dataset...")
            return get_example_dataset()

        # Parse the data
        data = []
        labels = []
        feature_count = None

        for line in lines:
            parts = [x.strip() for x in line.split(',')]
            if len(parts) < 2:
                print(f"[WARN]  Skipping invalid line: {line}")
                continue

            # Last part is label, rest are features
            label = parts[-1]
            features = parts[:-1]

            if feature_count is None:
                feature_count = len(features)
            elif len(features) != feature_count:
                print(f"[WARN]  Skipping line with inconsistent features: {line}")
                continue

            # Process features
            processed_row = []
            for val in features:
                try:
                    processed_row.append(float(val))
                except:
                    processed_row.append(val)

            data.append(processed_row)
            labels.append(label)

        if not data:
            print("❌ No valid data found. Using example dataset...")
            return get_example_dataset()

        # Get feature names
        n_features = feature_count
        feature_names = []
        print(f"\nDetected {len(data)} samples with {n_features} features")
        print("\nEnter feature names (or press Enter for default names):")
        for i in range(n_features):
            name = input(f"Feature {i+1} name: ").strip()
            if not name:
                name = f"Feature_{i+1}"
            feature_names.append(name)

        X = np.array(data)
        y = np.array(labels)

        print(f"\n✅ Data loaded successfully!")
        print(f"   - Samples: {len(data)}")
        print(f"   - Features: {n_features}")
        print(f"   - Classes: {np.unique(y)}")

        return X, y, feature_names

    # Method 2: Traditional row-by-row input
    else:
        print("\n" + "-"*70)
        print("ROW-BY-ROW INPUT METHOD")
        print("-"*70)
        print("\nEnter your data row by row.")
        print("First, tell us about your dataset:")

        n_features = int(input("Enter number of features: "))

        feature_names = []
        print("\nEnter feature names (or press Enter for default names):")
        for i in range(n_features):
            name = input(f"Feature {i+1} name: ").strip()
            if not name:
                name = f"Feature_{i+1}"
            feature_names.append(name)

        print("\nEnter data row by row:")
        print("Format: feature1,feature2,...,featureN")
        print("Type 'done' when finished entering all rows")

        data = []
        sample_num = 1

        while True:
            row_input = input(f"\nRow {sample_num} (or 'done' to finish): ").strip()
            if row_input.lower() == 'done':
                if sample_num == 1:
                    print("No data entered. Using example dataset...")
                    return get_example_dataset()
                break

            row_data = [x.strip() for x in row_input.split(',')]

            if len(row_data) != n_features:
                print(f"[WARN]  Expected {n_features} features, got {len(row_data)}. Please re-enter.")
                continue

            # Try to convert to numeric, keep as string if fails
            processed_row = []
            for val in row_data:
                try:
                    processed_row.append(float(val))
                except:
                    processed_row.append(val)

            data.append(processed_row)
            sample_num += 1

        # Get target labels
        print("\nEnter target labels for each sample:")
        labels = []
        for i in range(len(data)):
            label = input(f"Label for sample {i+1}: ").strip()
            labels.append(label)

        X = np.array(data)
        y = np.array(labels)

        return X, y, feature_names


def get_csv_input():
    """Load dataset from CSV file"""
    import os

    print("\n--- CSV File Input ---")
    print("Tip: You can paste the file path directly, with or without quotes")
    print("Example: C:\\Users\\durga\\Downloads\\health_data.csv")
    print("      or: \"C:\\Users\\durga\\Downloads\\health_data.csv\"")

    file_path = input("\nEnter path to CSV file: ").strip()

    # Remove quotes from both ends if present
    if file_path.startswith('"') and file_path.endswith('"'):
        file_path = file_path[1:-1]
    elif file_path.startswith("'") and file_path.endswith("'"):
        file_path = file_path[1:-1]

    # Remove any remaining quotes
    file_path = file_path.strip('"').strip("'")

    # Normalize path (handle Windows paths better)
    file_path = os.path.normpath(file_path)

    # Check if file exists
    if not os.path.exists(file_path):
        print(f"\n❌ Error: File not found at: {file_path}")
        print("Please check:")
        print("  1. The file path is correct")
        print("  2. The file exists at that location")
        print("  3. You have permission to read the file")

        retry = input("\nWould you like to try again? (y/n): ").strip().lower()
        if retry == 'y':
            return get_csv_input()
        else:
            print("Using example dataset instead...")
            return get_example_dataset()

    # Check if it's a CSV file
    if not file_path.lower().endswith('.csv'):
        print(f"\n[WARN]  Warning: File '{file_path}' doesn't have .csv extension")
        proceed = input("Do you want to continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            return get_csv_input()

    try:
        print(f"\nLoading file: {file_path}")
        df = pd.read_csv(file_path)
        print(f"✅ Successfully loaded dataset with shape: {df.shape}")
        print(f"   - Rows: {df.shape[0]}")
        print(f"   - Columns: {df.shape[1]}")
        print("\nFirst few rows:")
        print(df.head())

        # Ask which column is the target
        print(f"\nAvailable columns: {list(df.columns)}")
        target_col = input("Enter name of target column (or press Enter to use last column): ").strip()

        if not target_col:
            target_col = df.columns[-1]
            print(f"Using last column '{target_col}' as target.")

        if target_col not in df.columns:
            print(f"[WARN]  Error: '{target_col}' not found in columns.")
            print(f"Using last column '{df.columns[-1]}' as target.")
            target_col = df.columns[-1]

        # Separate features and target
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols].values
        y = df[target_col].values

        print(f"\n✅ Dataset prepared successfully!")
        print(f"   - Features: {len(feature_cols)} ({feature_cols})")
        print(f"   - Target: {target_col}")
        print(f"   - Classes: {np.unique(y)}")

        return X, y, feature_cols

    except pd.errors.EmptyDataError:
        print(f"\n❌ Error: The CSV file is empty: {file_path}")
        print("Using example dataset instead...")
        return get_example_dataset()
    except pd.errors.ParserError as e:
        print(f"\n❌ Error: Could not parse CSV file: {e}")
        print("The file might be corrupted or not a valid CSV file.")
        print("Using example dataset instead...")
        return get_example_dataset()
    except PermissionError:
        print(f"\n❌ Error: Permission denied. Cannot read file: {file_path}")
        print("Please check file permissions.")
        print("Using example dataset instead...")
        return get_example_dataset()
    except Exception as e:
        print(f"\n❌ Error loading file: {e}")
        print(f"Error type: {type(e).__name__}")
        print("\nTroubleshooting tips:")
        print("  1. Make sure the file path is correct")
        print("  2. Ensure the file is a valid CSV file")
        print("  3. Check if the file is open in another program")
        print("  4. Try copying the file to a different location")
        retry = input("\nWould you like to try again? (y/n): ").strip().lower()
        if retry == 'y':
            return get_csv_input()
        else:
            print("Using example dataset instead...")
            return get_example_dataset()


def get_example_dataset():
    """Generate example dataset"""
    print("\n--- Example Dataset ---")
    print("Generating example dataset with mixed numerical and categorical features...")

    np.random.seed(42)
    n_samples = 150

    # Numerical features
    X_num = np.random.randn(n_samples, 2)
    X_num[:, 0] = X_num[:, 0] * 3 + 10  # Feature 1: mean=10
    X_num[:, 1] = X_num[:, 1] * 2 + 5   # Feature 2: mean=5

    # Categorical features
    X_cat = np.random.choice(['Low', 'Medium', 'High'], size=(n_samples, 2))

    # Combine
    X = np.column_stack([X_num, X_cat])

    # Create target labels based on some pattern
    y = []
    for i in range(n_samples):
        if X_num[i, 0] > 10 and X_cat[i, 0] == 'High':
            y.append('Success')
        elif X_num[i, 1] < 5 and X_cat[i, 1] == 'Low':
            y.append('Success')
        else:
            y.append('Failure')
    y = np.array(y)

    feature_names = ['Income', 'Age', 'Education_Level', 'Experience']

    print(f"Generated dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Classes: {np.unique(y)}")

    return X, y, feature_names


def main():
    """Main interactive function"""
    print("\n" + "="*70)
    print("INTERACTIVE NAIVE BAYES CLASSIFIER")
    print("="*70)
    print("\nThis tool implements Naive Bayes theorem with automatic detection")
    print("of numerical (Gaussian) and categorical (Multinomial) features.")
    print("\nFeatures:")
    print("  - Automatic feature type detection")
    print("  - Supports mixed numerical and categorical data")
    print("  - Comprehensive visualizations")
    print("  - Detailed model summary")

    # Get dataset
    X, y, feature_names = get_user_dataset()

    print(f"\nDataset loaded successfully!")
    print(f"  Shape: {X.shape}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Feature names: {feature_names}")

    # Split data into train and test sets to prevent overfitting
    try:
        from sklearn.model_selection import train_test_split
        print("\n[Split] Splitting data into training and test sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"   Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"   Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        print(f"   Using stratified split to maintain class distribution")
        use_test_split = True
    except Exception as e:
        print(f"\n[WARNING] Could not split data: {e}")
        print("   Using all data for training (not recommended for evaluation)")
        X_train, X_test, y_train, y_test = X, X, y, y
        use_test_split = False

    # Detect feature types and train on training data only
    print("\nDetecting feature types...")
    nb = NaiveBayes()
    nb.fit(X_train, y_train, feature_names=feature_names)

    print("\nDetected feature types:")
    for i, (name, ftype) in enumerate(zip(feature_names, nb.feature_types)):
        print(f"  {name}: {ftype}")

    # Print model summary
    print_model_summary(nb)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualize_choice = input("Generate visualizations? (y/n): ").strip().lower()

    if visualize_choice == 'y':
        save_path = input("Enter path to save visualization (or press Enter to skip saving): ").strip()
        if not save_path:
            save_path = None
        visualize_naive_bayes(nb, X, y, save_path=save_path)

    # Make predictions on test samples
    print("\n" + "="*70)
    print("TESTING PREDICTIONS")
    print("="*70)

    test_choice = input("\nDo you want to test predictions? (y/n): ").strip().lower()

    if test_choice == 'y':
        print("\nEnter test sample(s):")
        print(f"Format: feature1,feature2,...,feature{len(feature_names)} (one sample per line)")
        print(f"Expected {len(feature_names)} features: {', '.join(feature_names)}")
        print("Type 'done' when finished")

        test_samples = []
        sample_num = 1

        while True:
            sample_input = input(f"\nTest sample {sample_num}: ").strip()
            if sample_input.lower() == 'done':
                break

            if not sample_input:
                print("[WARN]  Empty input. Please enter data or type 'done' to finish.")
                continue

            row_data = [x.strip() for x in sample_input.split(',')]

            # Check if we have the right number of features
            if len(row_data) != len(feature_names):
                print(f"[WARN]  Error: Expected {len(feature_names)} features, got {len(row_data)}")
                print(f"   Please enter exactly {len(feature_names)} values separated by commas")
                print(f"   Features expected: {', '.join(feature_names)}")
                continue

            # Process similar to manual input
            processed_row = []
            for val in row_data:
                try:
                    processed_row.append(float(val))
                except:
                    processed_row.append(val)

            test_samples.append(processed_row)
            sample_num += 1

        if test_samples:
            # Validate and fix test samples
            expected_features = len(feature_names)
            valid_samples = []

            print(f"\n{'='*70}")
            print("VALIDATING TEST SAMPLES")
            print('='*70)

            for idx, sample in enumerate(test_samples, 1):
                if len(sample) != expected_features:
                    print(f"[WARN]  Sample {idx}: Expected {expected_features} features, got {len(sample)}")
                    print(f"   Sample data: {sample}")
                    print(f"   Skipping this sample...")
                    continue
                valid_samples.append(sample)

            if not valid_samples:
                print("\n❌ No valid test samples found. All samples were rejected.")
                print("Please make sure each sample has exactly", expected_features, "features.")
            else:
                print(f"\n✅ {len(valid_samples)} valid test sample(s) found out of {len(test_samples)} entered")

                # Convert to numpy array with proper handling for mixed types
                try:
                    # Try to create a regular array first
                    test_X = np.array(valid_samples)
                except (ValueError, TypeError):
                    # If that fails (mixed types), use object dtype
                    test_X = np.array(valid_samples, dtype=object)

                # Always show detailed calculations
                print("\n" + "-"*70)
                print("🔍 DETAILED PROBABILITY CALCULATIONS WILL BE DISPLAYED")
                print("-"*70)
                print("The output will show:")
                print("  [OK] Step-by-step probability calculations")
                print("  [OK] Importance and interpretation of Mean and Standard Deviation")
                print("  [OK] Gaussian probability formula breakdown")
                print("  [OK] Distance analysis (Z-scores)")
                print("  [OK] Final normalized probabilities")
                print("-"*70)

                # Show detailed calculations for each sample
                for i, x in enumerate(test_X):
                    print(f"\n{'#'*70}")
                    print(f"SAMPLE {i+1} OF {len(test_X)}")
                    print('#'*70)

                    # Ensure x is a 1D array
                    if x.ndim > 1:
                        x = x.flatten()

                    probabilities = nb.predict_with_calculations(x, verbose=True)
                    predictions = [max(probabilities, key=probabilities.get)]

                    # Add a separator
                    if i < len(test_X) - 1:
                        print(f"\n{'='*70}")
                        print("Press Enter to see next sample...")
                        input()

    # Calculate and display accuracy with proper evaluation
    print("\n" + "="*70)
    print("MODEL PERFORMANCE (Anti-Overfitting Evaluation)")
    print("="*70)
    
    try:
        from sklearn.metrics import confusion_matrix, classification_report
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        
        # Training accuracy
        train_accuracy = nb.score(X_train, y_train)
        print(f"\n[Training] Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"   (Evaluated on {len(X_train)} training samples)")
        
        if use_test_split:
            # Test accuracy (main metric)
            test_accuracy = nb.score(X_test, y_test)
            print(f"\n[Test] Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"   (Evaluated on {len(X_test)} UNSEEN test samples)")
            
            # Check for overfitting
            overfitting_gap = train_accuracy - test_accuracy
            if overfitting_gap > 0.10:
                print(f"\n[WARNING] Potential Overfitting Detected!")
                print(f"   Training-Test Gap: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")
            elif overfitting_gap > 0.05:
                print(f"\n[CAUTION] Moderate Overfitting Detected")
                print(f"   Training-Test Gap: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")
            else:
                print(f"\n[OK] No Significant Overfitting")
                print(f"   Training-Test Gap: {overfitting_gap:.4f} ({overfitting_gap*100:.2f}%)")
            
            # Cross-validation
            print("\n[Cross-Validation] Performing 5-fold cross-validation...")
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(nb, X, y, cv=cv, scoring='accuracy')
            print(f"   Mean CV Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
            print(f"   Std CV Accuracy:  {cv_scores.std():.4f} ({cv_scores.std()*100:.2f}%)")
            
            # Confusion matrices
            train_predictions = nb.predict(X_train)
            test_predictions = nb.predict(X_test)
            
            print("\n[Matrix] Training Set Confusion Matrix:")
            train_cm = confusion_matrix(y_train, train_predictions)
            print(train_cm)
            
            print("\n[Matrix] Test Set Confusion Matrix:")
            test_cm = confusion_matrix(y_test, test_predictions)
            print(test_cm)
            
            print("\n[Report] Training Set Classification Report:")
            print(classification_report(y_train, train_predictions))
            
            print("\n[Report] Test Set Classification Report:")
            print(classification_report(y_test, test_predictions))
        else:
            # Fallback to training only
            predictions = nb.predict(X_train)
            cm = confusion_matrix(y_train, predictions)
            print("\n[Matrix] Confusion Matrix (Training Set):")
            print(cm)
            print("\n[Report] Classification Report (Training Set):")
            print(classification_report(y_train, predictions))
            
    except ImportError:
        print("\nNote: Install scikit-learn for detailed metrics and cross-validation")
        train_accuracy = nb.score(X_train, y_train)
        print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    except Exception as e:
        print(f"\n[WARNING] Could not calculate detailed metrics: {e}")
        train_accuracy = nb.score(X_train, y_train)
        print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")

    print("\n" + "="*70)
    print("Thank you for using Naive Bayes Classifier!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


