from dash import dcc, html
import plotly.express as px


class Scatterplot(html.Div):
    def __init__(self, name, feature_x, feature_y, df):
        self.html_id = name.lower().replace(" ", "-")
        self.df = df
        self.feature_x = feature_x
        self.feature_y = feature_y
        self.fig = px.scatter(
            self.df,
            x=self.feature_x,
            y=self.feature_y,
            hover_data=self.df.columns,
            custom_data=[self.df.index],
            color=self.df.columns[0],
            color_continuous_scale="redor"
        )

        # Equivalent to `html.Div([...])`
        super().__init__(
            className="graph_card",
            children=[
                html.H6(name + " - 1 stands for Good RiskPerformance"),
                dcc.Graph(id=self.html_id)
            ],
        )

    def update(self, sccolor):
        self.fig = px.scatter(
            self.df,
            x=self.feature_x,
            y=self.feature_y,
            hover_data=self.df.columns,
            custom_data=[self.df.index],
            color=sccolor,
            color_continuous_scale="redor"
        )

        # TODO: Port voronoi background (contourf) to plotly
        # Decision boundary using Voronoi tesselation
        # from sklearn.neighbors import KNeighborsClassifier

        # # create meshgrid
        # res = 80
        # X2d_xmin, X2d_xmax = np.min(X_embed[:, 0]), np.max(X_embed[:, 0])
        # X2d_ymin, X2d_ymax = np.min(X_embed[:, 1]), np.max(X_embed[:, 1])
        # xx, yy = np.meshgrid(np.linspace(X2d_xmin, X2d_xmax, resolution),
        #                      np.linspace(X2d_ymin, X2d_ymax, resolution))
        #
        # # approximate Voronoi tesselation using KNN
        # background_model = KNeighborsClassifier(n_neighbors=5).fit(X_embed,
        #                                                            y_pred)
        # voronoiBackground = background_model.predict_proba(
        #     np.c_[xx.ravel(), yy.ravel()])[:, 1]
        # voronoiBackground = voronoiBackground.reshape((res, res))
        #
        # # plot
        # plt.contourf(xx, yy, voronoiBackground, cmap='magma')
        # plt.scatter(X_embed[:, 0], X_embed[:, 1], c=y_pred_prob, s=5,
        #             cmap='magma')
        # plt.colorbar()

        self.fig.update_traces(mode='markers', marker_size=5)
        self.fig.update_layout(
            yaxis_zeroline=False,
            xaxis_zeroline=False,
            dragmode='select'
        )
        self.fig.update_xaxes(visible=False, showticklabels=False)
        self.fig.update_yaxes(visible=False, showticklabels=False)

        return self.fig
