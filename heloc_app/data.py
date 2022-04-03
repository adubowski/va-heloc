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

    # Remove rows with missing values
    df = features.drop(["RiskPerformance"], axis=1)

    # Remove rows with the same value (all -9s)
    rows_with_missing_values = df[df.apply(lambda x: min(x) == max(x), 1)]
    features = features.drop(rows_with_missing_values.index.tolist())
    features = features.reset_index(drop=True)

    # Drop columns with correlation over 0.8 with lower feature importance
    to_remove = ['NumTotalTrades', 'NumTrades90Ever/DerogPubRec', 'NumInqLast6Mexcl7days']
    features = features.drop(to_remove, axis=1)

    X = features[features.columns[1:]]
    y = features["RiskPerformance"]

    # columns categorization
    categorical = ['MaxDelqEver', 'MaxDelq/PublicRecLast12M']
    numerical = [col for col in X.columns if col not in categorical]
    
    # Code labels
    y_code = y.astype("category").cat.codes
    
    for cat in categorical:
        features[cat] = features[cat].astype("category")

    # model
    model = RandomForestClassifier(n_estimators=200)

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
    return X_test_transformed, X_test, y_test, features, y_pred_prob, X_train, y_train, model, numerical


def tsne(X_test):
    X_embed = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(X_test)
    return X_embed