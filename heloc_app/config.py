from .data import get_data

features = get_data()
DATA_COLS = features[features.columns[1:]].columns.tolist()
Y_COLS = ["y_pred", "y_pred_prob", "y_test"]

# Scatterplot columns
SSC_COLS = Y_COLS + DATA_COLS
GROUP_TYPES = ['Number of', 'Number of months', 'Percentage', 'Net Fraction']

GRAPH_TYPES = ["Scatterplot Matrix", "Histogram", "Boxplot"]
COL_GROUPS = ["Trade", "Inquiry", "Delinquency"]

DATA_COLORS = ['RiskPerformance'] + DATA_COLS