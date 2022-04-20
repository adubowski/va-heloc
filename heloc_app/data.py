import json
from typing import Tuple, List

import pandas as pd
from dice_ml import Model, Dice, Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def get_data() -> pd.DataFrame:
    """ Loads and preprocesses the HELOC dataset

    :return: preprocessed original 'features' df with HELOC data
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


def get_x_y(features: pd.DataFrame, subset: List[str] = None) \
        -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                 pd.DataFrame]:
    """Splits the given features dataset into train and test sets, imputes
    missing data and scales numerical values. Can take a subset of the data
    if needed. Defines lists of numerical and categorical columns in the data
    based on the final columns.

    :param features: full original HELOC dataset
    :param subset: list of columns to be used for the X sets
    :return: Tuple of dataframes with respectively: transformed test data,
    original test data, labels for test data, train data, labels for train data
    """
    categorical = []
    X = features[features.columns[1:]]
    y = features["RiskPerformance"]

    if subset and len(subset) > 0:
        X = X[subset]

    if 'MaxDelqEver' in X.columns.tolist():
        categorical.append('MaxDelqEver')
    if 'MaxDelq/PublicRecLast12M' in X.columns.tolist():
        categorical.append('MaxDelq/PublicRecLast12M')

    # columns categorization
    numerical = [col for col in X.columns if col not in categorical]

    # Code labels
    y_code = y.astype("category").cat.codes

    for cat in categorical:
        features[cat] = features[cat].astype("category")

    num_pipe = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    cat_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'),
                             OrdinalEncoder())
    transformer = ColumnTransformer(transformers=[('num', num_pipe, numerical),
                                                  ('cat', cat_pipe,
                                                   categorical)])

    X_transform = pd.DataFrame(
        transformer.fit_transform(X),
        columns=list(numerical) + list(categorical)
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y_code,
                                                        stratify=y_code,
                                                        train_size=0.9,
                                                        random_state=0)
    _, X_test_transformed, _, _ = train_test_split(
        X_transform, y_code, stratify=y_code, train_size=0.9, random_state=0)

    return X_test_transformed, X_test, y_test, X_train, y_train


def get_numerical_cols(X: pd.DataFrame) -> List[str]:
    """Retrieves a list of numerical columns from the provided X set

    :param X: dataset to retrieve the columns from
    :return: list of numerical columns
    """
    return list(X.select_dtypes(include='int64').columns)


def get_fitted_model(X_train, y_train) -> RandomForestClassifier:
    """ Returns model trained on the provided data

    :param X_train: training dataset for the model
    :param y_train: labels for the training dataset
    :return: Fitted Random Forest classifier model
    """
    model = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def get_scatterplot_df(X_test_transformed, X_test, y_test, y_pred_prob):
    """ Prepares dataset for the scatterplot using TSNE to reduce
    dimensionality of the provided transformed dataset.

    :param X_test_transformed: Scaled test dataset
    :param X_test: Original test dataset
    :param y_test: labels for the test data
    :param y_pred_prob: predicted probabilities on the test data
    :return: Dataframe prepared for the scatterplot
    (with TSNE embeddings and original data)
    """
    X_embed = TSNE(n_components=2, learning_rate='auto', init='pca') \
        .fit_transform(X_test_transformed)
    dic = {
        "y_predict": y_pred_prob,
        "y_test": y_test.astype(str),
        "Embedding 1": X_embed[:, 0],
        "Embedding 2": X_embed[:, 1]
    }
    X_copy = X_test.copy()
    for i, k in dic.items():
        X_copy[i] = k
    return X_copy


def get_predictions(model, X_test) -> Tuple[list, list]:
    """Returns predictions and their probabilities from the given binary
    model on the given test set.

    :param model: binary classifier
    :param X_test: test set for the predictions
    :return: two lists, for class predictions and probabilities, respectively
    """
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    return y_pred, y_pred_prob


def get_counterfactual_df(X_train, y_train, model, numerical, X_test,
                          point_index, include_all_cols=False) -> \
        pd.DataFrame:
    """Provides a dataframe with DiCE counterfactual explanations for the given
    test point using provided classifier.

    :param X_train: train data of the model
    :param y_train: labels of the train data
    :param model: fitted classifier
    :param numerical: list of numerical columns in the dataset. Required
    because they are handled differently than categorical in DiCE
    :param X_test: test dataset
    :param point_index: index of the point in the test dataset
    :param include_all_cols: flag whether to include all columns of the data
    :return: Dataframe with Actual and Counterfactual values for the point.
    Unless include_all_cols is set to True, only changed columns are included.
    """
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
    test_data = d['test_data'][0][0][:20]  # TODO: Is it always 20?
    cf_df = pd.DataFrame(
        [test_data, cfs_list],
        columns=d['feature_names'],
        index=['Actual', 'Closest CounterFactual']
    )
    if include_all_cols:
        cf_df['Value'] = cf_df.index.tolist()
        return cf_df

    # Drop columns with the same values
    unique_cols = cf_df.nunique()
    cols_to_drop = unique_cols[unique_cols == 1].index
    output = cf_df.drop(cols_to_drop, axis=1)
    output['Value'] = output.index.tolist()
    return output
