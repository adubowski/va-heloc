from dash import dcc, html
import plotly.express as px


class Histogram(html.Div):
    def __init__(self, name, col1, col2, df):
        """
       :param name: name of the plot
       :param df: dataframe
       """
        self.html_id = name.lower().replace(" ", "-")
        self.df = df
        self.col1 = col1
        self.col2 = col2
        self.name = name
        self.title_id = self.html_id + "-t"
        self.fig = px.histogram(self.df, x=self.col1, y=self.col2,
                                color="RiskPerformance",
                                marginal="violin",
                                hover_data=self.df.columns)

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

    def update(self, selected_col1, selected_col2):
        if selected_col1:
            self.col1 = selected_col1
        if selected_col2:
            self.col2 = selected_col2

        self.fig = px.histogram(self.df, x=self.col1, y=self.col2,
                                color="RiskPerformance",
                                marginal="violin",  # or violin, rug
                                hover_data=self.df.columns)

        self.fig.update_layout(
            height = 800,
            yaxis_zeroline=False,
            xaxis_zeroline=False,
            dragmode='select'
        )
        self.fig.update_xaxes(fixedrange=True)
        self.fig.update_yaxes(fixedrange=True)

        # update titles
        self.fig.update_layout(
            xaxis_title=self.col1,
            yaxis_title=self.col2,
        )

        return self.fig
