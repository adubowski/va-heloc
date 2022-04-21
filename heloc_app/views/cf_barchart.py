import pandas as pd
from dash import dcc, html
import plotly.express as px
from ..data import get_counterfactual_df, get_numerical_cols


class CFBarchart(html.Div):
    def __init__(self, name, point, X_train, y_train, X_test, model):
        """Barchart presenting CounterFactual (CF) values for a given point
        :param name: name of the plot
        :param point: ID of the point for which to calculate CFs
        :param X_train: train dataset used in the model
        :param y_train: labels of the train dataset used in the model
        :param X_test: test dataset used in the model
        :param model: fitted classifier model
        """
        self.html_id = name.lower().replace(" ", "-")
        self.name = name
        self.point = point
        self.title_id = self.html_id + "-t"
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.model = model

        # Equivalent to `html.Div([...])`
        super().__init__(
            className="graph_card",
            children=[
                html.H6(id=self.title_id,
                        children="DiCE Counterfactual Explanations - What "
                                 "change is needed to achieve the opposite "
                                 "outcome?"
                        ),
                dcc.Graph(id=self.html_id),
            ],
        )

    def update(self, point):
        if not point:
            point = self.point

        cf_df = get_counterfactual_df(
            self.X_train,
            self.y_train,
            self.model,
            get_numerical_cols(self.X_test),
            self.X_test,
            point,
            return_index=False
        )
        scenarios = []
        cols = []
        values = []
        for col in cf_df.columns:
            scenarios.extend(cf_df.index.tolist())
            cols.extend(len(cf_df.index) * [col])
            values.extend(cf_df[col])

        d = {
            "Scenario": scenarios,
            "Feature": cols,
            "Value": values
        }

        self.fig = px.bar(
            data_frame=pd.DataFrame(d),
            x="Feature",
            y="Value",
            orientation='v',
            color="Scenario",
            barmode="group"
        )

        self.fig.update_layout(
            yaxis_zeroline=False,
            xaxis_zeroline=False,
            hovermode='closest',
        )
        self.fig.update_xaxes(fixedrange=True)
        self.fig.update_yaxes(fixedrange=True)

        # update titles
        self.fig.update_layout(
            xaxis_title="Feature",
            yaxis_title="Values",
        )

        return self.fig
