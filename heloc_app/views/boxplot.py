from dash import dcc, html
import plotly.express as px


class Boxplot(html.Div):
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
                        children=self.name
                        ),
                dcc.Graph(id=self.html_id),
            ],
        )

    def update(self, cols):
        colss = ["RiskPerformance"] + cols
        self.fig = px.box(self.df[colss], color="RiskPerformance")

        self.fig.update_layout(
            height=800,
            yaxis_zeroline=False,
            xaxis_zeroline=False,
            dragmode='select'
        )
        self.fig.update_xaxes(fixedrange=True)
        self.fig.update_yaxes(fixedrange=True)

        return self.fig
