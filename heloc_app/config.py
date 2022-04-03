# Here you can add any global configuations
import pandas as pd

features = pd.read_csv('heloc_model/heloc_dataset_v1.csv')
to_remove = ['NumTotalTrades', 'NumTrades90Ever/DerogPubRec', 'NumInqLast6Mexcl7days']
features = features.drop(to_remove, axis=1)

graph_type = ["Scatterplot Matrix", "Histogram", "Boxplot"]
columns = features[features.columns[1:]].columns.tolist()
color_type = ["y_test", "y_predict"]
col_group = ["Trade", "Inquiry", "Delinquency"]

colorssc = color_type + columns
group_type = ['Number of', 'Number of months', 'Percentage', 'Net Fraction']