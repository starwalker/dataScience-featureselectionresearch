from dcor import partial_distance_correlation, partial_distance_covariance, distance_correlation
import numpy as np
import itertools
import logging
import pandas as pd
from warnings import warn
import logging
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import StandardScaler
from itertools import chain
from collections import Counter
from FeatureSelectors.setup_logger import logger
from scipy.sparse import coo_matrix

class IsObserved(BaseEstimator, ClassifierMixin):
    '''
    Class that returns a array of one hot encoded values that
    represent whether a value has been observed or not

    Can be used on strings or numerics

    exampe:
        x = np.random.choice([0,.5, -5., 10,  1, None, np.nan], size=(100,10))
        s = IsObserved().fit(x)
        results = s.transform(x)
    '''

    def __init__(self, return_sparse=False, suffix= '_is_obsrvd'):
        self.coef_ = None
        self.inputs_ = None
        self.suffix = suffix
        self.return_sparse =  return_sparse

    def __call__(self, X):
        '''
        param X: array or pandas data frame
        param y: array of outcomes (unsused for compatatbility with scikit learn pipelines)
        return numpy array
        '''
        df = pd.DataFrame(X)\
        .replace({'':None})\
        .isna()\
        .replace({True:0, False:1})
        return df.values

    def fit(self, X, y=None):
        '''
        param X: array or pandas data frame
        param y: array of outcomes (unsused for compatatbility with scikit learn pipelines)
        return self

        '''
        df = pd.DataFrame(X)
        self.coef_ = 1-df.isna().sum(axis=0)/df.shape[0]
        self.inputs_ = df.columns
        return self

    def transform(self, X, y=None):
        '''
        param X: array or pandas data frame
        param y: array of outcomes (unsused for compatatbility with scikit learn pipelines)
        return numpy array or coo_matrix is self.resturn_sparse=True

        '''
        check_is_fitted(self)
        if X.shape[1] != len(self.inputs_):
            logger.warning(F'number of input columns: {len(self.inputs_)} does not match fitted: {X.shape[1]}')
        df = pd.DataFrame(X)\
        .replace({'':None})\
        .isna()\
        .replace({True:0, False:1})
        if self.return_sparse:
            return coo_matrix(df.values)
        else:
            return df.values

    def fit_transform(self, X, y=None):
        '''
        param X: array or pandas data frame
        param y: array of outcomes (unsused for compatatbility with scikit learn pipelines)
        return numpy array or coo_matrix is self.resturn_sparse=True

        '''
        self = self.fit(X)
        return self.transform(X)

    def get_feature_names(self):
        '''
        returns a list of feature Names
        '''
        check_is_fitted(self)
        return [str(v) + self.suffix for v in self.inputs_]

    def get_inputs(self):
        '''
        returns a list of expected inputs
        '''
        check_is_fitted(self)
        list(self.inputs_)

class Scaler(BaseEstimator, ClassifierMixin):
    '''
    Scaler Method that does not require imputatation work,
    Params are similart Scikitlearn StandardScaler
    example:
        x = np.random.choice([0,.5, -5., 10,  1, None, np.nan], size=(100,10))
        s = Scaler().fit(x)
        results = s.inverse_transform(s.transform(x, sample_size=10))

    '''

    def __init__(self, with_mean=True, with_std=True):
        self.with_mean = with_mean
        self.with_std = with_std
        self._scalers = []
        self.scale_ = []
        self.mean_ = []
        self.var_ = []
        self.n_samples_seen_ = []

    def _fit_scaler_gen(self, X):
        array_list = np.hsplit(X, X.shape[1])
        for array in array_list:
            x = pd.Series(array.flatten(), dtype=np.float32).dropna().values
            x = np.reshape(x, (-1,1))
            s = StandardScaler(copy=False, with_mean=self.with_mean, with_std=self.with_std)
            yield s.fit(x)

    def fit(self, X, y=None):
        '''
        param X: array or pandas data frame
        param y: array of outcomes (unsused for compatatbility with scikit learn pipelines)
        return self

        '''
        try:
            X = X.values
        except AttributeError:
            pass
        self._scalers = list(self._fit_scaler_gen(X))
        for scaler in self._scalers:
            try:
                self.scale_.append(scaler.scale_[0])
            except IndexError:
                pass
            try:
                self.mean_.append(scaler.mean_[0])
            except IndexError:
                pass
            try:
                self.mean_.append(scaler.var_[0])
            except IndexError:
                pass
            try:
                self.n_samples_seen_.append(scaler.n_samples_seen_ [0])
            except IndexError:
                pass

        return self

    def _transform_scaler_gen(self, X):
        array_list = np.hsplit(X, X.shape[1])
        results = np.zeros
        for i, array in enumerate(array_list):
            x = np.reshape(array, (-1,1))
            results = self._scalers[i].transform(x)
            yield results

    def _inverse_transform_scaler_gen(self, X):
        array_list = np.hsplit(X, X.shape[1])
        results = np.zeros
        for i, array in enumerate(array_list):
            x = np.reshape(array, (-1,1))
            results = self._scalers[i].inverse_transform(x)
            yield results

    def _transform(self, X, y=None):
        check_is_fitted(self)
        return np.hstack(list(self._transform_scaler_gen(X)))

    def fit_transform(self, X, y=None):
        self = self.fit(X)
        return self.transform(X)

    def _inverse_transform(self, X):
        try:
            X = X.values
        except AttributeError:
            pass
        return np.hstack(list(self._inverse_transform_scaler_gen(X)))

    def transform(self, X, sample_size=1000):
        '''
        Transform Method Scales on array X
        param X: array or pandas data frame
        param sampe_size int (size of chunks for parallel processing )
        return self

        '''
        try:
            X = X.values
        except AttributeError:
            pass
        n_splits = np.max((int(X.shape[0]/sample_size), 2))
        x_chunks =  np.array_split(X, n_splits)
        results = Parallel()(delayed(self._transform)(chunck) for chunck in x_chunks)
        return np.vstack(results)

    def inverse_transform(self, X, sample_size=1000):
        '''
        Reverses Transform Method Scales on array X
        param X: array or pandas data frame
        param sampe_size int (size of chunks for parallel processing )
        return self

        '''
        n_splits = np.max((int(X.shape[0]/sample_size), 2))
        x_chunks =  np.array_split(X, n_splits)
        results = Parallel()(delayed(self._inverse_transform)(chunck) for chunck in x_chunks)
        return np.vstack(results)


def _test_Scaler():
    x = np.random.choice([0,.5, -5., 10,  1, None, np.nan], size=(100,10))
    s = Scaler().fit(x)
    results= s.transform(x)
    logger.info('Scaler Transformer estimator test completed')
    results = s.inverse_transform(s.transform(x, sample_size=10))


def _test_IsObserved():
    x = np.random.choice([0,.5, -5., 10,  1, None, np.nan], size=(100,10))
    s = IsObserved().fit(x)
    results= s.transform(x)
    x = np.random.choice(['val', None,''], size=(10,3))
    x = pd.DataFrame(x, columns = ['test0', 'test1', 'test2'])
    s = IsObserved(return_sparse=True).fit(x)
    results= s.transform(x)
    logger.debug(s.get_feature_names())
    logger.info('IsObserved transformer Tested complete')
