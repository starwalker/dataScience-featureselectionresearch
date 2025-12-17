from utils import load_data, build_model
import argparse
from prince import MCA
from sklearn.tree import DecisionTreeClassifier
from FeatureSelectors.feature_extraction import IsObserved
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from FeatureSelectors.Distancecorrelation import Selector
from FeatureSelectors.feature_extraction import Scaler
import numpy as np
import pandas as pd
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-experiment_id',default=1, type=int,  help="ExperimentId")
args = parser.parse_args()
experiment_id = args.experiment_id

run_name = 'evalML_dcor_scaled'

df, X, y, train_index, test_index = load_data()
input_feature_names = np.array(list(df.drop('label',axis=1).columns))

imputer = SimpleImputer().fit(X[train_index, :] )
transformers = [ ('imp', imputer, input_feature_names )]

pipe = ColumnTransformer(transformers).fit(df.iloc[train_index, :].loc[:, input_feature_names ])

X_train_initial = pipe.transform(df.iloc[train_index, :].loc[:, input_feature_names ])
y_train = y[train_index]
y_test = y[test_index]

transformed_feature_names = np.array( list(input_feature_names))
scaler = Scaler()
selector = Selector()

#feature selection via dcor
print(F'selecting features with {type(selector)}')
selector = selector.fit(scaler.fit_transform(X_train_initial), y_train)
support_index = selector.get_support()
best_features = transformed_feature_names[support_index]

# primary feature transformation and subseting of features
X_train = pd.DataFrame(pipe.transform(df.iloc[train_index,:].loc[:, input_feature_names ])[:, support_index ], columns=best_features)
X_test =  pd.DataFrame(pipe.transform(df.iloc[test_index,:].loc[:, input_feature_names ])[:, support_index ], columns=best_features)

##builds the model
build_model(X_train, y_train, X_test, y_test, 1, run_name, 'dcor')
