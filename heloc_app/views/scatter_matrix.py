import pandas as pd
from dash import dcc, html
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import cross_val_predict


class DataScatterMatrix(html.Div):
    def __init__(self, name, df, model):
        """
       :param name: name of the plot
       :param df: dataframe
       """
        self.html_id = name.lower().replace(" ", "-")
        self.name = name
        self.df = df
        self.model = model
        self.title_id = self.html_id + "-t"
        # Equivalent to `html.Div([...])`
        super().__init__(
            className="graph_card",
            children=[
                html.H6(id=self.title_id,
                        children="Data Exploration Matrix"),
                dcc.Graph(id=self.html_id),

            ],
        )

    def update(self, selected_color):
        cols = [
            "NumTrades60Ever/DerogPubRec",
            "NumTradesOpeninLast12M",
            "NumInqLast6M",
            "NumRevolvingTradesWBalance",
            "NumInstallTradesWBalance",
            "NumBank/NatlTradesWHighUtilization",
            "NumSatisfactoryTrades"
        ]
        self.fig = px.scatter_matrix(
            data_frame=self.df,
            dimensions=cols,
            color=selected_color
        )

        self.fig.update_layout(
            yaxis_zeroline=False,
            xaxis_zeroline=False,
            dragmode='select',
            width=1600,
            height=1100,
            hovermode='closest',
        )
        self.fig.update_xaxes(fixedrange=True)
        self.fig.update_yaxes(fixedrange=True)

        for annotation in self.fig['layout']['annotations']:
            annotation['textangle'] = 60
        return self.fig
