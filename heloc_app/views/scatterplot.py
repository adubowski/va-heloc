from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


class Scatterplot(html.Div):
    def __init__(self, name, feature_x, feature_y, input_df):
        self.html_id = name.lower().replace(" ", "-")
        self.df = input_df
        self.feature_x = feature_x
        self.feature_y = feature_y

        # Equivalent to `html.Div([...])`
        super().__init__(
            className="graph_card",
            children=[
                html.H6(name + " - 1 stands for Good RiskPerformance"),
                dcc.Graph(id=self.html_id)
            ],
        )

    def update(self, scplt_color, input_df):

        categorical = []
        if 'MaxDelqEver' in input_df.columns:
            categorical.append('MaxDelqEver')
        if 'MaxDelq/PublicRecLast12M' in input_df.columns:
            categorical.append('MaxDelq/PublicRecLast12M')

        scplot_cmap = \
            'RdYlGn' if scplt_color in ('y_pred_prob', 'y_pred', 'y_test') \
            else px.colors.qualitative.G10 if scplt_color in categorical \
            else 'Oranges'

        self.fig = px.scatter(
            input_df,
            x=self.feature_x,
            y=self.feature_y,
            hover_data=input_df.columns,
            custom_data=[input_df.index],
            color=scplt_color,
            color_continuous_scale=scplot_cmap,
            color_discrete_sequence=px.colors.qualitative.G10
        )

        # Estimate model's decision boundary using Voronoi tesselation
        # Source: https://stackoverflow.com/a/61225622/9994398

        # create meshgrid
        res = 80
        X2d_xmin = np.min(input_df[self.feature_x])
        X2d_xmax = np.max(input_df[self.feature_x])
        X2d_ymin = np.min(input_df[self.feature_y])
        X2d_ymax = np.max(input_df[self.feature_y])
        xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, res),
                             np.linspace(X2d_ymin, X2d_ymax, res))

        X_embed = input_df[[self.feature_x, self.feature_y]]

        # approximate Voronoi tesselation using KNN
        background_model = KNeighborsClassifier(n_neighbors=10).fit(
            X_embed,
            input_df.y_pred
        )
        voronoiBackground = background_model.predict_proba(
            np.c_[xx.ravel(), yy.ravel()]
        )[:, 1]
        voronoiBackground = voronoiBackground.reshape((res, res))

        self.fig.add_trace(
            go.Contour(
                x=np.linspace(X2d_xmin, X2d_xmax, res),
                y=np.linspace(X2d_ymin, X2d_ymax, res),
                z=voronoiBackground,
                colorscale='YlGn',
                hoverinfo='skip',
                line_smoothing=0.75
            )
        )

        self.fig.update_layout(
            yaxis_zeroline=False,
            xaxis_zeroline=False,
            dragmode='select',
            height=650,
            coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside")
        )
        self.fig.update_xaxes(visible=False, showticklabels=False)
        self.fig.update_yaxes(visible=False, showticklabels=False)

        return self.fig
