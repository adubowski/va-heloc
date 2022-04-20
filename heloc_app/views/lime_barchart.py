import pandas as pd
from dash import dcc, html
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer


class LimeBarchart(html.Div):
    def __init__(self, name, point, X_train, X_test, model):
        """
       :param name: name of the plot
       :param point: ID of the point for which to calculate lime explanations
       :param df: dataframe
       """
        self.html_id = name.lower().replace(" ", "-")
        self.name = name
        self.point = point
        self.title_id = self.html_id + "-t"
        self.X_train = X_train
        self.X_test = X_test
        self.model = model
        # Equivalent to `html.Div([...])`
        super().__init__(
            className="graph_card",
            children=[
                html.H6(id=self.title_id,
                        children="LIME Probability Explained"
                        ),
                dcc.Graph(id=self.html_id),
            ],
        )

    def update(self, point):
        if not point:
            point = self.point

        explainer = LimeTabularExplainer(
            self.X_train.values,
            mode='classification',
            feature_names=self.X_train.columns,
            verbose=True
        )
        exp = explainer.explain_instance(self.X_test.loc[point],
                                         self.model.predict_proba)

        explanations = [e[0] for e in exp.as_list()]
        prob_values = [e[1] for e in exp.as_list()]
        impact = ['Higher RiskPerformance Score' if e[1] > 0
                  else 'Lower RiskPerformance Score' for e in exp.as_list()]

        d = {
            "Explanation": explanations,
            "Probability attributed": prob_values,
            "Impact": impact
        }

        self.fig = px.bar(
            data_frame=pd.DataFrame(d),
            y="Explanation",
            x="Probability attributed",
            orientation='h',
            color="Impact"
        )

        self.fig.update_layout(
            yaxis_zeroline=False,
            xaxis_zeroline=False,
            dragmode='select',
            hovermode='closest',
        )
        self.fig.update_xaxes(fixedrange=True)
        self.fig.update_yaxes(fixedrange=True)

        # update titles
        self.fig.update_layout(
            xaxis_title="Explanation",
            yaxis_title="Probability attributed",
        )

        return self.fig
