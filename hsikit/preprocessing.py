import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array
from scipy.signal import savgol_filter

class SNV(BaseEstimator, TransformerMixin):
    """
    Standard Normal Variate (SNV) preprocessing class.
    Each spectrum is centered and scaled individually.
    """
    def fit(self, X, y=None):
        X = check_array(X)
        return self
    
    def transform(self, X, y=None):
        X = check_array(X)
        X = X.copy()

        means = X.mean(axis=1, keepdims=True)
        stds = X.std(axis=1, keepdims=True)

        stds[stds == 0] = 1e-12

        Xsnv = (X - means) / stds
        return Xsnv

class MSC(BaseEstimator, TransformerMixin):
    """
    Multiplicative Scatter Correction (MSC) preprocessing class.
    Mean spectrum as reference if None.
    """
    def __init__(self, reference=None):
        self.reference = reference

    def fit(self, X, y=None):

        X = check_array(X)
        X = X.copy()

        # Mean centering
        X_centered = X - X.mean(axis=1, keepdims=True)

        # Set reference
        if self.reference is None:
            self.reference_ = X_centered.mean(axis=0)
        else:
            self.reference_ = np.asarray(self.reference)
            if self.reference_.shape[0] != X.shape[1]:
                raise ValueError("reference spectrum must match number of features")

        return self
    
    def transform(self, X, y=None):
        check_is_fitted(self, "reference_")

        X = check_array(X)
        X = X.copy()

        # Mean centering
        X_centered = X - X.mean(axis=1, keepdims=True)

        r = self.reference_

        # Center reference
        r_mean = r.mean()
        r_centered = r - r_mean

        # Precompute denominator (scalar)
        denom = np.dot(r_centered, r_centered)

        # Compute slopes (vector of shape[n_samples])
        slopes = (X_centered @ r_centered) / denom
        slopes[slopes == 0] = 1e-12

        # Compute intercepts
        X_means = X_centered.mean(axis=1)
        intercepts = X_means - slopes * r_mean

        # Apply correction
        Xmsc = (X_centered - intercepts[:, np.newaxis]) / slopes[:, np.newaxis]

        return Xmsc
    
class SavitzkyGolay(BaseEstimator, TransformerMixin):
    """
    Savitzky-Golay preprocessing class.
    Define window_length, polyorder and deriv when initializing.
    """
    def __init__(self, window_length=11, polyorder=2, deriv=0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv

    def fit(self, X, y=None):
        X = check_array(X)

        # Basic parameter validation
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        
        if self.window_length <= self.polyorder:
            raise ValueError("window_length must be > polyorder")
        
        return self
    
    def transform(self, X, y=None):
        X = check_array(X)

        # Apply filter along feature axis
        Xsg = savgol_filter(X, window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv, axis=1)

        return Xsg