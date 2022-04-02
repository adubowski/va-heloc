import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE

import time


def get_data():
    start = time.time()
    # Read data
    features = pd.read_csv('heloc_model/heloc_dataset_v1.csv')
    # impute special values with nan
    features.replace([-9], np.nan, inplace=True)

    # Remove rows with missing values
    features = features.dropna() 
    # features = features.reset_index(drop=True)
    # drop columns with more than 1000 special values
    #features.drop(features.columns[features.isnull().sum() > 1000], axis=1, inplace=True)

    X = features[features.columns[1:]]
    y = features["RiskPerformance"]
    # print(X.index == y.index)
    # print(X.index)

    # columns categorization
    categorical = ['MaxDelqEver', 'MaxDelq/PublicRecLast12M']
    numerical = [col for col in X.columns if col not in categorical]
    
    # Code labels
    y_code = y.astype("category").cat.codes
    
    for cat in categorical:
        features[cat] = features[cat].astype("category")

    # model
    model = RandomForestClassifier(n_estimators=1500)

    num_pipe = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), OrdinalEncoder())
    transformer = ColumnTransformer(transformers=[('num', num_pipe, numerical), ('cat', cat_pipe, categorical)])

    X_transform = pd.DataFrame(transformer.fit_transform(X), columns=list(numerical) + list(categorical))
    X_train, X_test, y_train, y_test = train_test_split(X, y_code, stratify=y_code, train_size=0.9, random_state=0)
    X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed = \
        train_test_split(X_transform, y_code, stratify=y_code, train_size=0.9, random_state=0)
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    print(time.time()-start)
    return X_test_transformed, X_test, y_test, features, y_pred_prob


def tsne(X_test):
    X_embed = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(X_test)
    return X_embed