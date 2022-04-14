import json

import pandas as pd
from dice_ml import Model, Dice, Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


CATEGORICAL = ['MaxDelqEver', 'MaxDelq/PublicRecLast12M']


def get_data():
    """ Loads and preprocesses the general dataset

    :return: preprocessed original 'features' df
    """
    features = pd.read_csv('heloc_model/heloc_dataset_v1.csv')

    # Remove rows with all the same values (missing values)
    df = features.drop(["RiskPerformance"], axis=1)
    rows_with_missing_values = df[df.apply(lambda x: min(x) == max(x), 1)]
    features = features.drop(rows_with_missing_values.index.tolist())
    features = features.reset_index(drop=True)

    # Drop columns with correlation over 0.8 with lower feature importance
    to_remove = ['NumTotalTrades', 'NumTrades90Ever/DerogPubRec',
                 'NumInqLast6Mexcl7days']
    features = features.drop(to_remove, axis=1)
    return features


def get_x_y(features):
    X = features[features.columns[1:]]
    y = features["RiskPerformance"]

    # columns categorization
    numerical = [col for col in X.columns if col not in CATEGORICAL]

    # Code labels
    y_code = y.astype("category").cat.codes

    for cat in CATEGORICAL:
        features[cat] = features[cat].astype("category")

    num_pipe = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),
                             OrdinalEncoder())
    transformer = ColumnTransformer(transformers=[('num', num_pipe, numerical),
                                                  ('cat', cat_pipe,
                                                   CATEGORICAL)])

    X_transform = pd.DataFrame(transformer.fit_transform(X),
                               columns=list(numerical) + list(CATEGORICAL))
    X_train, X_test, y_train, y_test = train_test_split(X, y_code,
                                                        stratify=y_code,
                                                        train_size=0.9,
                                                        random_state=0)
    _, X_test_transformed, _, _ = train_test_split(
        X_transform, y_code, stratify=y_code, train_size=0.9, random_state=0)

    return X_test_transformed, X_test, y_test, X_train, y_train, numerical


def get_fitted_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def get_predictions(model, X_test):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_prob


def get_counterfactual_df(X_train, y_train, model, numerical, X_test,
                          point_index):
    # DiCE counterfactual explanations
    df = X_train.copy()
    df['y'] = y_train.copy()
    data = Data(
        dataframe=df,
        continuous_features=numerical,
        outcome_name='y'
    )
    m = Model(model=model, backend='sklearn')
    dice = Dice(data, m, method='random')
    e = dice.generate_counterfactuals(X_test.loc[[point_index]], total_CFs=1,
                                      desired_class="opposite")

    d = json.loads(e.to_json())
    cfs_list = d['cfs_list'][0][0][:20]
    test_data = d['test_data'][0][0][:20]
    cf_df = pd.DataFrame(
        [test_data, cfs_list],
        columns=d['feature_names'],
        index=['Actual', 'Closest CounterFactual']
    )
    unique_cols = cf_df.nunique()
    cols_to_drop = unique_cols[unique_cols == 1].index

    output = cf_df.drop(cols_to_drop, axis=1)
    output['index'] = output.index.tolist()
    return output
