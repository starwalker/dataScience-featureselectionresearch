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

def get_best_distance_correlation_indicies_rfa(x, y, lr=0.001):
    '''
    Best Distance Correlation calcuated the distance correlation between every feature in array x and
    the array y, and adds them recursively to a set of features to include untill the distnace correlation
    fails to increase

    param x: numpy array without missing values
    param y: numpy array without missing values
    '''

    indices = np.arange(x.shape[1])
    distances = np.zeros(len(indices))
    for i in indices:
        distances[i] = distance_correlation(x[:,  i],y)
    distances = pd.Series(distances, index=indices)
    distances = distances.loc[distances > 0]
    if len(distances) == 0:
        logger.warning('No Indices returned a dsitance_correlation that was positive ')
        return np.array([])
    else:
        distances = distances.sort_values(ascending=False)
        d = distances.iloc[0]
        keep_set = set([distances.index[0]])
        for i in distances.index:
            test_indicies = list(set(list(keep_set) + [i]))
            logger.debug(F'testing indicies {test_indicies}')
            new_d = distance_correlation(x[:, test_indicies], y)
            if new_d - d >= lr:
                logger.debug(F'distance improved, d: {d} new_d: {new_d}')
                keep_set.update(test_indicies)
                d = new_d
            else:
                logger.debug('distance failed to improve')
        keep_list = list(keep_set)
        keep_list.sort()
        return np.array(keep_list)

def reduce_w_pdcor(x, y, proposed_index, lr=0.001):
    '''
    reduce_w_pdcor is function that takes a set of features (x) and an array of labels
    (y) with a proposed best set of feature indicies, and uses partial_distance_corrlation
    to test whether removing feature from the proposed index reduces the pdcor


    param x: numpy array without missing values
    param y: numpy array without missing values
    param lr: number of digits to round pdcor calculation (normally 2)
        setting higher will make system include more features
    returns: np array of feature indicies in the markov blanket
    '''

    drop_index = np.array([v for v in range(x.shape[1]) if v not in proposed_index])
    d = partial_distance_correlation(y,  x[:, drop_index], x[:, proposed_index])
    logger.debug(F'PartialDistanceCorrlation {d} for proposed feature set {proposed_index}')
    keep_set = set(proposed_index)
    for i in proposed_index:
        test_indicies_to_keep = np.array([v for v in list(keep_set) if v != i])
        logger.debug(F'testing indicies to keep: {test_indicies_to_keep}')
        test_indicies_to_drop =  np.array([v for v in range(x.shape[1]) if v not in test_indicies_to_keep])
        logger.debug(F'testing indicies to drop: {test_indicies_to_drop}')
        try:
            new_d = partial_distance_correlation(y,
                                                  x[:, test_indicies_to_drop],
                                                  x[:, test_indicies_to_keep])
            logger.debug(F'partial distance corr d: {d} new_d: {new_d} with removing i :{i}')
            if d - new_d >= lr:
                keep_set.remove(i)
                logger.debug(F'distance improved, d: {d} new_d: {new_d}')
                d = new_d
            else:
                logger.debug('partial distance correlation failed to descrease')
        except IndexError:
            logger.debug(F'skipping pdcor indicies to drop: {test_indicies_to_drop},testing indicies to keep: {test_indicies_to_keep} ')
    keep_list = list(keep_set)
    keep_list.sort()
    return np.array(keep_list)


def get_mb(x, y, lr=.001):
    '''
    This method uses get_best_distance_correlation_indicies_rfa
    followed by reduce_w_pdcor to get a feature set that meets markov blanket
    contitions
    param x: numpy array without missing values
    param y: numpy array without missing values
    param r: number of digits to round pdcor calculation (normally 2)
        setting higher will make system include more features
    returns: np array of feature indicies in the markov blanket

    '''
    if x.shape[0]!=y.shape[0]:
        raise ValueError(F'X shape {x.shape[0]} and y {y.shape[0]}  need to have the same number of observations' )
    if np.sum(np.isnan(x)) > 0:
        raise ValueError('x cannot contain null values')
    if np.sum(np.isnan(y)) > 0:
        raise ValueError('y cannot contain null values')
    fwd_pass_index = get_best_distance_correlation_indicies_rfa(x, y)
    logger.debug(F'best fwd pass {fwd_pass_index }')
    if len(fwd_pass_index) == 0:
        logger.debug('no feature selected, pdcor step skipped')
        return np.array([])
    elif len(fwd_pass_index) == x.shape[1]:
        logger.debug('All features Selected, pdcor step skipped')
        return np.arange(x.shape[1])
    else:
        logger.debug('pdcor method reducing features ')
        best_features = reduce_w_pdcor(x, y, fwd_pass_index, lr=lr)
        if len(fwd_pass_index) == len(best_features):
            logger.warning(F'PDCor did not remove any features with max_features  try setting a larger max_features')
        else:
            logger.debug(F'PDCor reduced feature set size from {len(fwd_pass_index )} to {len(best_features)}')
        return best_features



def mb_search(x, y,  lr=2, sample_size=1000):
    '''
    This method uses get_best_distance_correlation_indicies_rfa
    followed by reduce_w_pdcor to get a feature set that meets markov blanket
    contitions
    Uses Joblib Parallel to discover a markov blanket per sample

    param x: numpy array without missing values
    param y: numpy array without missing values
    param r: number of digits to round pdcor calculation (normally 2)
        setting higher will make system include more features
    param sample_size: sample size used to spit arrays into
    returns: list of np arrays of feature indicies in the markov blanket
    '''
    n_splits = np.max((int(x.shape[0]/sample_size), 2))
    x_chunks =  np.array_split(x, n_splits)
    y_chunks = np.array_split(y, n_splits)
    results = Parallel()(delayed(get_mb)(*v) for v in zip(x_chunks, y_chunks))
    return results


class Selector(BaseEstimator, ClassifierMixin):
    '''
    Distance Correlation Feature Selection
    param: r int number of digits to round using n.round, when compairing distances
    param: sample_size int, size of the sample (selection method splits arrays row wise)
    param: min_freq int, min number of markov blankets a feature must appear in
    Arrays are split a mimimum twice, and two markov blankets are created, the
    mim_freq of 2 means that features have to appear in at least 2 mbs.

    This method uses get_best_distance_correlation_indicies_rfa
    followed by reduce_w_pdcor to get a feature set that meets markov blanket
    contitions

    Example:
        x =  np.random.normal(1, size=(100, 10))
        y =  np.random.choice([0,1], size=(100)).astype(np.float32)
        results = Selector().fit_transform(x, y)
        print(results)
    '''

    def __init__(self, r=2, sample_size=1000, min_freq=2):
        self.r = r
        self.min_freq = min_freq
        self.sample_size = sample_size
        self.markov_blankets_ = []
        self.selected_features = []

    def fit(self, X, y):
        '''
        Fit Method for Distance Correlation Feature Selection

        param X: numpy array with (rows, columns)
        param y: numpy array (rows)
        returns: self
        '''
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        try:
            X = X.values
        except AttributeError:
            pass
        try:
            y = y.values
        except AttributeError:
            pass
        self.y_ = y
        self.markov_blankets_ = mb_search(X.astype(float), y.astype(float), self.r, self.sample_size)
        counts = Counter(list(chain.from_iterable(self.markov_blankets_)))
        indicies = Parallel([v for (v, c) in sorted(counts.items()) if c>= self.min_freq])
        self.selected_features = list(set(chain.from_iterable(self.markov_blankets_)))
        return self

    def transform(self, X):
        '''
        Fit Method for Distance Correlation Feature Selection

        param X: numpy array with (rows, columns)
        returns: numpy array with (rows, selected_features)
        '''
        check_is_fitted(self)
        X = check_array(X)
        try:
            X = X.values
        except  AttributeError:
            pass
        return X[:, self.selected_features]

    def fit_transform(self, X, y):
        self = self.fit(X, y)
        return self.transform(X)

    def get_support(self):
        return self.selected_features

def _test_selection_method():
    np.random.seed(2012)
    size = 1000
    n_features = 10
    y = np.random.choice([0,1], size=(size )).astype(float)
    x = np.random.normal(0, .1, size=(size ,n_features))
    x[:, 0] = y + np.random.normal(.3, size=(size))
    x[:, 1] = y + np.random.normal(.3, size=(size))
    x[:, 2] = y + np.random.normal(.3, size=(size))
    x[:, 3] = np.random.choice([0,1], size=(size))
    x[:, 4] = np.random.choice([0,1], size=(size))
    x[:, 5] = np.random.choice([0,1], size=(size))
    x[:, 6] = np.random.choice([0,1], size=(size), p = [.9, .1])
    x[:, 7] = np.random.choice([0,1], size=(size), p = [.9, .1])
    #best_features = get_mb(x, y)
    #print(best_features)
    #assert len(best_features) == 3
    #assert all(best_features == np.array([0,1,2]))
    logger.info('testing parallel methods:')
    results  = mb_search(x,y,  sample_size=100)
    logger.info(F'parallel results {results}')
    results = Selector().fit_transform(x, y)
    assert results.shape[0] == x.shape[0]
    assert results.shape[1] == 3
    results  = mb_search(np.zeros(shape=(size, 10)),y,  sample_size=100)
    logger.info('selection methods test completed')
    print('test complete')
