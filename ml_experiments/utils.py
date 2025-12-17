

# ## Hosptial Mortality Classifcation
# this notebookes creates classifers that predict probablity that a patient died in the hospital based on lab values. It uses Phyisio Mimic III as a data source and uses the python evalML to evaluate classifers. m

import numpy as np
import pandas as pd

from sklearn.metrics import *
from evalml.automl import AutoMLSearch
from sklearn.model_selection import cross_validate,  StratifiedKFold
import evalml
import os
import re
from pyspark.sql.types import *
from pyspark.sql.functions import *
import mlflow
from evalml.model_understanding.prediction_explanations import explain_predictions
from mlflow.models.signature import infer_signature
import json
import os

def load_data():
    from pyspark.sql import SparkSession
    MAX_MEMORY = "32g"
    data_dir = os.getenv('PHYSIO_HOME')
    # #### Data Loading
    # Data is loaded from Phyiso MimiIII amd saved as paquet to pyspark data frames

    # #### Data Egneineering
    # creates a features data frame using max and min lab values during hospital stays

    # reads all the csvs and writes them to parquet filesspark = SparkSession.builder \
    spark = SparkSession.builder     .appName("HostpitalMortalityClassifier")     .config("spark.executor.memory", MAX_MEMORY)     .config("spark.driver.memory", MAX_MEMORY)     .getOrCreate()

    LABEVENTS =  spark.read.parquet(data_dir + '/LABEVENTS.parquet')
    D_LABITEMS =  spark.read.parquet(data_dir + '/D_LABITEMS.parquet')
    ADMISSIONS =   spark.read.parquet(data_dir + '/ADMISSIONS.parquet')

    # sets the number of features to section by frequency
    n_features = 100

    # gets the top n_features most frequent features
    top_features = LABEVENTS\
    .join(D_LABITEMS, on = 'ITEMID', how='inner')\
    .dropna(subset=['VALUENUM'])\
    .groupby('LABEL')\
    .count()\
    .sort('count', ascending=False)\
    .limit(n_features).drop('count')

    ## Data Transformations
    ## gets the max and min value from the top n_features
    ## groups by hospital admit id
    ## creates a flag where the patient died "Expired" in the hosptial
    data = LABEVENTS.join(D_LABITEMS, on = 'ITEMID', how='inner')\
    .join(top_features, on='label', how='inner')\
    .dropna(subset=['VALUENUM'])\
    .groupby('HADM_ID').pivot('LABEL')\
    .agg(max('VALUENUM').alias('max'), min('VALUENUM').alias('min'))\
    .join(ADMISSIONS.select('HADM_ID', col('HOSPITAL_EXPIRE_FLAG').alias('label')), on='HADM_ID', how='inner')\
    .filter('label in (0,1)')

    ## data Extraction to Pandas
    df = data.toPandas().set_index('HADM_ID')

    ## create arrays for training model
    y = df.loc[:, 'label'].values
    X = df.drop('label', axis=1).values

    spark.stop()

    # #### Data Splitting
    # Data splitting via Statified Shuffle Split
    splitter = StratifiedKFold(shuffle=True)
    train_index, test_index = next(splitter.split(X, y))
    return df, X, y, train_index, test_index

def build_model(X_train, y_train, X_test, y_test, experiment_id, run_name, selector_type):

    assert X_train.shape[0] ==y_train.shape[0]
    assert X_test.shape[0] ==y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]

    automl = AutoMLSearch(X_train=X_train, y_train=y_train, problem_type='binary')
    automl.search()
    model = automl.best_pipeline

    # #### Model Performance
    # Calcuates Model Peformace on Test Set

    # predicts the test data
    test_preds = model.predict_proba(X_test).iloc[:, 1]
    test_pred_labels = model.predict(X_test)

    ## predicts the training data
    train_preds = model.predict_proba(X_train).iloc[:, 1]
    train_pred_labels = model.predict(X_train)

    # calcuates metrics on test data
    test_f1 = f1_score(y_test, test_pred_labels)
    test_acc_balanced = balanced_accuracy_score(y_test, test_pred_labels)
    test_acc = accuracy_score(y_test, test_pred_labels)
    test_precision = precision_score(y_test, test_pred_labels)
    test_recall = recall_score(y_test, test_pred_labels)
    test_auc_score = roc_auc_score(y_test,  test_preds)
    print(F'roc_auc_score: {test_auc_score } on test')

    # calculates metrics on training data
    train_f1 = f1_score(y_train, train_pred_labels)
    train_acc_balanced = balanced_accuracy_score(y_train, train_pred_labels)
    train_acc = accuracy_score(y_train, train_pred_labels)
    train_precision = precision_score(y_train, train_pred_labels)
    train_recall = recall_score(y_train, train_pred_labels)
    train_auc_score = roc_auc_score(y_train, train_preds)
    print(F'roc_auc_score: {train_auc_score} on train')

    # gets params Artifacts for logging mlflow model
    n_test_cases = np.sum(y_test == 1)
    n_test_controls = np.sum(y_test == 0)
    n_train_cases = np.sum(y_train == 1)
    n_train_controls = np.sum(y_train == 0)

    n_train_obs = X_train.shape[0]
    n_test_obs = X_test.shape[0]


    train_label_prob = y_train.mean()
    test_label_prob = y_test.mean()
    desc = str(model.describe())
    model_type = str(type(model))
    input_example = pd.DataFrame(X_train).head(5).fillna(0)
    signature = infer_signature(input_example,  model.predict_proba(input_example))
    n_features =input_example.shape[1]

    # #### Feature Importance
    # save feature importance to a dictionary for later logging
    imp = model.feature_importance.set_index('feature')

    # dumps feature importance to a dictionary for logging as an artifact
    imp_dict = imp.to_dict()['importance']
    imp_json_path = 'feature_importance.json'
    with open(imp_json_path, 'w') as f:
        json.dump(imp_dict,f)
    print(F'model build completed, best model {desc}')
    print('Flogging run to experiment_id {experiment_id}')
    artifact_path = 'Model'
    data_grain = 'HADM_ID'
    label_name = 'HOSPITAL_EXPIRE_FLAG'
    data_source = 'PhysioMimicIII'
    tracking_uri = "http://localhost:5000"
    mlflow.set_tracking_uri(tracking_uri)
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:

        tracking_uri = mlflow.get_tracking_uri()
        artifact_uri = mlflow.get_artifact_uri()
        print("Tracking uri: {}".format(tracking_uri))
        print("Artifact uri: {}".format(artifact_uri))
        mlflow.sklearn.log_model(model,
                             artifact_path=artifact_path,
                             signature=signature,
                             input_example=input_example
                            )
        mlflow.log_artifact(imp_json_path)
        mlflow.log_param('_run_name', run_name)
        mlflow.log_param('data_source', data_source)
        mlflow.log_param('label_name', label_name)
        mlflow.log_param('data_grain', data_grain)
        mlflow.log_param('n_test_cases', n_test_cases)
        mlflow.log_param('n_test_controls', n_test_controls)
        mlflow.log_param('n_train_cases', n_train_cases)
        mlflow.log_param('n_train_controls', n_train_controls)
        mlflow.log_param('n_train_obs', n_train_obs)
        mlflow.log_param('n_test_obs', n_test_obs)
        mlflow.log_param('n_features', n_features)
        mlflow.log_param('train_label_prob', train_label_prob)
        mlflow.log_param('test_label_prob', test_label_prob)
        mlflow.log_param('desc', desc)
        mlflow.log_param('model_type',model_type)
        mlflow.log_param('feature_selection', selector_type)
        mlflow.log_metric('train_f1', train_f1)
        mlflow.log_metric('train_acc_balanced', train_acc_balanced)
        mlflow.log_metric('train_acc', train_acc)
        mlflow.log_metric('train_precision', train_precision)
        mlflow.log_metric('train_recall', train_recall)
        mlflow.log_metric('train_auc_score', train_auc_score)
        mlflow.log_metric('test_f1', test_f1)
        mlflow.log_metric('test_acc_balanced', test_acc_balanced)
        mlflow.log_metric('test_acc', test_acc)
        mlflow.log_metric('test_precision', test_precision)
        mlflow.log_metric('test_recall', test_recall)
        mlflow.log_metric('test_auc_score', test_auc_score)
        mlflow.log_param('features', '|'.join(imp.index))
        run_id = run.info.run_id
        mlflow.end_run()
    print(F'logged experiment_id: "{experiment_id}" run_id :"{run_id}" completed')
