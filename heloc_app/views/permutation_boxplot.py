import numpy as np
import pandas as pd
from dash import dcc, html
import plotly.express as px
from sklearn.inspection import permutation_importance


class PermutationBoxplot(html.Div):
    def __init__(self, name, df):
        """
       :param name: name of the plot
       :param df: dataframe
       """
        self.html_id = name.lower().replace(" ", "-")
        self.df = df
        self.name = name
        self.fig = None
        self.title_id = self.html_id + "-t"

        # Equivalent to `html.Div([...])`
        super().__init__(
            className="graph_card",
            children=[
                html.H6(id=self.title_id,
                        children="Permutation Importances (might take a while)"
                        ),
                dcc.Graph(id=self.html_id),
            ],
        )

    def update(self, model, X_global_test, y_global_test):
        perm_importance = permutation_importance(
            model, X_global_test, y_global_test, n_repeats=7, random_state=0
        )  # Takes a long time
        sorted_idx = perm_importance.importances_mean.argsort()
        columns = list(X_global_test.columns)
        df = pd.DataFrame(
            perm_importance.importances[sorted_idx].T,
            columns=np.array(columns)[sorted_idx]
        )
        self.fig = px.box(df, orientation="h")

        self.fig.update_layout(
            height=800,
            yaxis_zeroline=False,
            xaxis_zeroline=False,
        )
        self.fig.update_xaxes(fixedrange=True)
        self.fig.update_yaxes(fixedrange=True)

        return self.fig
