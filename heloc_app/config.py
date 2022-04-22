from .data import get_data

features = get_data()
DATA_COLS = features[features.columns[1:]].columns.tolist()
Y_COLS = ["y_pred", "y_pred_prob", "y_test"]

GLOBAL_PLOT_TYPES = [
    "SHAP Bar Plot",
    "SHAP Summary Plot",
    "SHAP Decision Plot"
]
# Scatterplot columns
SSC_COLS = Y_COLS + DATA_COLS
DATA_GROUP_TYPES = ['Units', 'Months', 'Percentage',
                    'Net Fraction']

DATA_GRAPH_TYPES = ["Scatterplot Matrix", "Histogram", "Boxplot"]
COL_GROUPS = ["Trade", "Inquiry", "Delinquency"]

DATA_COLORS = ['RiskPerformance'] + DATA_COLS