if __name__ == "__main__":
    import logging
    from FeatureSelectors.setup_logger import logger
    from FeatureSelectors.Distancecorrelation import _test_selection_method
    from FeatureSelectors.feature_extraction import _test_Scaler, _test_IsObserved
    logger.setLevel(logging.INFO)
    _test_IsObserved()
    _test_selection_method()
    _test_Scaler()
    logger.info('all tests complete')
