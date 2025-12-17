#!/usr/bin/env python
# coding: utf-8

# ## Hosptial Mortality Classifcation
# this notebookes creates classifers that predict probablity that a patient died in the hospital based on lab values. It uses Phyisio Mimic III as a data source and uses the python evalML to evaluate classifers. m

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate,  StratifiedKFold
from sklearn.metrics import *
import argparse
from evalml.automl import AutoMLSearch
import evalml
import os
import re
import mlflow
from evalml.model_understanding.prediction_explanations import explain_predictions
from mlflow.models.signature import infer_signature
import json
import os
import warnings


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB



MAX_MEMORY = "32g"
data_dir = os.getenv('PHYSIO_HOME')


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-n',default=100, type=int,  help="integer Number of features to select")
args = parser.parse_args()

# sets the number of features to section by frequency
n_features_to_select = args.n


# #### Data Loading
# Data is loaded from Phyiso MimiIII amd saved as paquet to pyspark data frames

# #### Data Egneineering
# creates a features data frame using max and min lab values during hospital stays

# In[ ]:


# reads all the csvs and writes them to parquet filesspark = SparkSession.builder \
spark = SparkSession.builder     .appName("HostpitalMortalityClassifier")     .config("spark.executor.memory", MAX_MEMORY)     .config("spark.driver.memory", MAX_MEMORY)     .getOrCreate()

LABEVENTS =  spark.read.parquet(data_dir + '/LABEVENTS.parquet')
D_LABITEMS =  spark.read.parquet(data_dir + '/D_LABITEMS.parquet')
ADMISSIONS =   spark.read.parquet(data_dir + '/ADMISSIONS.parquet')


# sets the number of features to section by frequency
n_features = 100

# gets the top n_features most frequent features
top_features = LABEVENTS                .join(D_LABITEMS, on = 'ITEMID', how='inner')                .dropna(subset=['VALUENUM'])                .groupby('LABEL')                .count().sort('count', ascending=False)                .limit(n_features).drop('count')


## Data Transformations
## gets the max and min value from the top n_features
## groups by hospital admit id
## creates a flag where the patient died "Expired" in the hosptial
data = LABEVENTS.join(D_LABITEMS, on = 'ITEMID', how='inner').join(top_features, on='label', how='inner').dropna(subset=['VALUENUM']).groupby('HADM_ID').pivot('LABEL').agg(max('VALUENUM').alias('max'), min('VALUENUM').alias('min')).join(ADMISSIONS.select('HADM_ID', col('HOSPITAL_EXPIRE_FLAG').alias('label')), on='HADM_ID', how='inner').filter('label in (0,1)')

## data Extraction to Pandas
df = data.toPandas().set_index('HADM_ID')

## create arrays for training model
y = df.loc[:, 'label'].values
X = df.drop('label', axis=1).values

n_rows = X.shape[0]
n_features = X.shape[1]
feature_names_all = np.array(list(df.drop('label', axis=1).columns))
label_prob = y.mean()
print(F' n_rows: {n_rows}, n_features: {n_features}, label_prob {np.round(label_prob , 3)}')
print(F'features: {feature_names_all}')

spark.stop()


# #### Basic Data Statics
#

# In[ ]:


data_stats_path = 'data_stats.csv'
data_stats = df.describe()
data_stats.to_csv(data_stats_path)
data_stats


# In[ ]:


# #### Data Splitting
# Data splitting via Statified Shuffle Split
#
# #### Feature Selection pyImpetous

# In[ ]:




splitter = StratifiedKFold(shuffle=True)
train_index, test_index = next(splitter.split(X, y))
imputer = SimpleImputer()
selector = RFE(DecisionTreeClassifier(min_samples_leaf=5, max_depth=10) ,step=5, n_features_to_select=n_features_to_select )

#featuer selection via recussive feature elemination
selector= selector.fit(pd.DataFrame(imputer.fit_transform(X[train_index, :]), columns=feature_names_all).fillna(0),
                               y[train_index])

support_index = selector.get_support()
feature_names = feature_names_all[support_index]

imputer = SimpleImputer()

X_train = imputer.fit_transform(df.iloc[train_index, :].loc[:, feature_names])
X_test = imputer.fit_transform(df.iloc[test_index, :].loc[:, feature_names])
y_train = y[train_index]
y_test = y[test_index]


print(F'Selected Feature Names {feature_names}')


# In[ ]:


from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression(max_iter=5000)
estimator.fit(X_train, y_train)


# #### Modeliing Fitting Using AutoML
# Searchs through models to find best

model = Pipeline(steps=[('imp', imputer),('classifier', estimator)])


# #### Model Performance
# Calcuates Model Peformace on Test Set

# In[ ]:


# predicts the test data
test_preds = estimator.predict_proba(X_test)[:, 1]
test_pred_labels = estimator.predict(X_test)

# calcuates metrics on test data
test_f1 = f1_score(y_test, test_pred_labels)
test_acc_balanced = balanced_accuracy_score(y_test, test_pred_labels)
test_acc = accuracy_score(y_test, test_pred_labels)
test_precision = precision_score(y_test, test_pred_labels)
test_recall = recall_score(y_test, test_pred_labels)
test_auc_score = roc_auc_score(y[test_index], test_preds)
print(F'roc_auc_score: {test_auc_score } on test')

## predicts the training data
train_preds = model.predict_proba(X_train)[:, 1]
train_pred_labels = model.predict(X_train)

# calculates metrics on training data
train_f1 = f1_score(y_train, train_pred_labels)
train_acc_balanced = balanced_accuracy_score(y_train, train_pred_labels)
train_acc = accuracy_score(y_train, train_pred_labels)
train_precision = precision_score(y_train, train_pred_labels)
train_recall = recall_score(y_train, train_pred_labels)
train_auc_score = roc_auc_score(y_train, train_preds)
print(F'roc_auc_score: {train_auc_score} on train')


# In[ ]:


# gets params Artifacts for logging mlflow model
n_cases = np.sum(y == 1)
n_controls = np.sum(y == 0)
n_train_obs = X_train.shape[0]
n_test_obs = X_test.shape[0]
n_features = X_train.shape[1]
train_label_prob = y_train.mean()
test_label_prob = y_test.mean()
model_type = type(estimator)
split_type = type(splitter)
input_example = pd.DataFrame(X_train[:10, :], columns=feature_names).fillna(0)
signature = infer_signature(input_example , model.predict_proba(input_example))

print(F'using {n_features }')
print(F'Model Type {model_type }')
print(F'Selector Type {type(selector) }')

# #### Feature Importance
# save feature importance to a dictionary for later logging

# In[ ]:


imp = pd.Series(estimator.coef_[0], index=feature_names).sort_values(ascending=False)

# dumps feature importance to a dictionary for logging as an artifact
imp_dict = imp.to_dict()
imp_json_path = 'feature_importance.json'
with open(imp_json_path, 'w') as f:
    json.dump(imp_dict,f)

imp.head()


# #### Model Tracking
# Uses an mlflow tracking server to save the model, parameters and metrics

# In[ ]:


artifact_path = 'Model'
data_grain = 'HADM_ID'
label_name = 'HOSPITAL_EXPIRE_FLAG'
data_source = 'PhysioMimicIII'
run_name = 'rfe_log_reg'
tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(tracking_uri)

with mlflow.start_run(run_name=run_name, experiment_id=1) as run:

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
    mlflow.log_artifact(data_stats_path)
    mlflow.log_param('data_source', data_source)
    mlflow.log_param('label_name', label_name)
    mlflow.log_param('data_grain', data_grain)
    mlflow.log_param('n_cases', n_cases)
    mlflow.log_param('n_controls', n_controls)
    mlflow.log_param('n_train_obs', n_train_obs)
    mlflow.log_param('n_test_obs', n_test_obs)
    mlflow.log_param('n_features', n_features)
    mlflow.log_param('train_label_prob', train_label_prob)
    mlflow.log_param('test_label_prob', test_label_prob)
    mlflow.log_param('desc', 'rfe_selector_with_log_reg')
    mlflow.log_param('model_type',model_type)
    mlflow.log_param('split_type',split_type)
    mlflow.log_param('feature_selection', type(selector))
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
    experiment_id = run.info.experiment_id
    mlflow.end_run()
    print(F'logging experiment_id: "{experiment_id}" run_id :"{run_id}" completed')


# In[ ]:





# In[ ]:
