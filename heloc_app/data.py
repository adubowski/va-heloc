import plotly.express as px
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def get_data():
    # Read data
    #df = px.data.iris()
    #read csv
    features = pd.read_csv('heloc_model/heloc_dataset_v1.csv')
    #impute special values with nan
    features.replace([-9, -8, -7], np.nan, inplace=True)
    #y label
    labelDimension = "RiskPerformance"
    # Remove rows with more than 10 missing values
    features = features.dropna(thresh=10) 
     # drop columns with more than 1000 special values
    features.drop(features.columns[features.isnull().sum() > 1000], axis=1, inplace=True)
    
    #X-y 
    X = features[features.columns[1:]]
    y = features[labelDimension]

    #columns
    categorical = ['MaxDelqEver', 'MaxDelq/PublicRecLast12M']
    numerical = [col for col in X.columns if col not in categorical]
    
    # Categorize
    y_code = y.astype("category").cat.codes
    
    for cat in categorical:
        features[cat] = features[cat].astype("category")

    # get the model
    model = RandomForestClassifier(n_estimators = 1500)

    #pipeline the model
    num_pipe = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), OrdinalEncoder())
    transformer = ColumnTransformer(transformers=[('num', num_pipe, numerical), ('cat', cat_pipe, categorical)])

    X_transform = pd.DataFrame(transformer.fit_transform(X), columns=list(numerical) + list(categorical))
    X_train, X_test, y_train, y_test = train_test_split(X, y_code, stratify=y_code, train_size = 0.7, random_state=0)
    X_train_transformed, X_test_transformed, y_train_transformed, y_test_transformed = train_test_split(X_transform, y_code, stratify=y_code, train_size = 0.7, random_state=0)
    model.fit(X_train_transformed, y_train_transformed)
    
    return X_test_transformed, y_test_transformed, features
