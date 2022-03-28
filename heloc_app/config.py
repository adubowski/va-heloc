# Here you can add any global configuations
from heloc_app.data import get_data

_ , _, features = get_data()
graph_type = ["Bar Chart", "Histogram"]
columns = features[features.columns[1:]].columns.tolist()
color_list1 = ["green", "blue"]
color_list2 = ["red", "purple"]
