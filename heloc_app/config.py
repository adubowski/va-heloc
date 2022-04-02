# Here you can add any global configuations
from heloc_app.data import get_data
import pandas as pd

features = pd.read_csv('heloc_model/heloc_dataset_v1.csv')
graph_type = ["Histogram","Boxplot"]
columns = features[features.columns[1:]].columns.tolist()
color_type = ["y_test", "y_predict"]
col_group = ["Trade", "Inquiry", "Delinquency"]

