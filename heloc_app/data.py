import base64
from dash import html, dcc
import io
import json
from typing import Tuple, List

import pandas as pd
import shap
from dice_ml import Model, Dice, Data
from matplotlib import pyplot as plt
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
    # Sample half the data to make app smoother
    X, _, y, _ = train_test_split(X, y, stratify=y, train_size=0.5)
    if subset and len(subset) > 0:
        X = X[subset]

    if 'MaxDelqEver' in X.columns:
        categorical.append('MaxDelqEver')
    if 'MaxDelq/PublicRecLast12M' in X.columns:
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
                                                        train_size=0.8,
                                                        random_state=0)
    _, X_test_transformed, _, _ = train_test_split(
        X_transform, y_code, stratify=y_code, train_size=0.8, random_state=0)

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
    model = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        max_features='sqrt'
    )
    model.fit(X_train, y_train)
    return model


def get_scatterplot_df(X_test_transformed, X_test, y_test, model):
    """ Prepares dataset for the scatterplot using TSNE to reduce
    dimensionality of the provided transformed dataset.

    :param X_test_transformed: Scaled test dataset
    :param X_test: Original test dataset
    :param y_test: labels for the test data
    :param model: model to get predicted probabilities on the test data from
    :return: Dataframe prepared for the scatterplot
    (with TSNE embeddings and original data)
    """
    y_pred, y_pred_prob = get_predictions(model, X_test)

    X_embed = TSNE(n_components=2, learning_rate='auto', init='pca') \
        .fit_transform(X_test_transformed)
    dic = {
        "y_pred_prob": y_pred_prob,
        "y_pred": y_pred,
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
                          point_index, include_all_cols=False,
                          return_index=True):
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
    :param return_index: flag whether to return also the index of the
    dataframe as separate column
    :return: Dataframe with Actual and Counterfactual values for the point.
    Unless include_all_cols is set to True, only changed columns are included.
    """
    # DiCE counterfactual explanations
    df = X_train.copy()
    df['y'] = y_train.copy()

    # Calculates reasonable ranges for each feature using the training dataset
    permitted_range_dict = {}
    for col in numerical:
        col_min = max(min(X_train[col]), 0)
        col_max = max(X_train[col])
        permitted_range_dict[col] = [col_min, col_max]

    data = Data(
        dataframe=df,
        continuous_features=numerical,
        outcome_name='y'
    )
    m = Model(model=model, backend='sklearn')
    dice = Dice(data, m, method='random')
    e = dice.generate_counterfactuals(
        X_test.loc[[point_index]],
        total_CFs=1,
        desired_class="opposite",
        permitted_range=permitted_range_dict
    )

    d = json.loads(e.to_json())
    n_x_cols = len(d['feature_names'])
    cfs_list = d['cfs_list'][0][0][:n_x_cols]
    test_data = d['test_data'][0][0][:n_x_cols]
    cf_df = pd.DataFrame(
        [test_data, cfs_list],
        columns=d['feature_names'],
        index=['Current', 'Alternative']
    )
    if not include_all_cols:
        # Drop columns with the same values
        unique_cols = cf_df.nunique()
        cols_to_drop = unique_cols[unique_cols == 1].index
        cf_df = cf_df.drop(cols_to_drop, axis=1)

    if return_index:
        cf_df.insert(0, 'Value', cf_df.index.tolist())
    return cf_df


def get_shap_plot(model, X_test, plot_type='bar'):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)[1]
    if plot_type == 'bar':
        shap.summary_plot(
            shap_values,
            X_test,
            show=False,
            plot_type='bar'
        )
    elif plot_type == 'summary':
        shap.summary_plot(
            shap_values,
            X_test,
            show=False
        )
    elif plot_type == 'decision':
        shap.decision_plot(
            explainer.expected_value[0],
            shap_values,
            X_test.iloc,
            show=False
        )
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches='tight')
    encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()
    return html.Img(src=f"data:image/png;base64, {encoded}")
