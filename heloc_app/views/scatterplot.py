from dash import dcc, html
from numpy import size
import plotly.express as px
import plotly.graph_objects as go


class Scatterplot(html.Div):
    def __init__(self, name, feature_x, feature_y, df):
        self.html_id = name.lower().replace(" ", "-")
        self.df = df
        self.feature_x = feature_x
        self.feature_y = feature_y

        # Equivalent to `html.Div([...])`
        super().__init__(
            className="graph_card",
            children=[
                html.H6(name),
                dcc.Graph(id=self.html_id)
            ],
        )

    def update(self, sccolor, selected_data):

        # cols = [c for c in self.df.columns if c not in ["Embedding 1","Embedding 2"]]
        df1 = self.df.copy()
        df1 = df1.drop(columns=["Embedding 1", "Embedding 2"])
        self.fig = px.scatter(
            self.df,
            x=self.feature_x, 
            y=self.feature_y,
            hover_data=[df1.index],
            color=sccolor,
        )

        # x_values = self.df[self.feature_x]
        # y_values = self.df[self.feature_y]
        self.fig.update_traces(mode='markers', marker_size=5)
        self.fig.update_layout(
            yaxis_zeroline=False,
            xaxis_zeroline=False,
            dragmode='select'
        )
        self.fig.update_xaxes(fixedrange=True, showticklabels=False)
        self.fig.update_yaxes(visible=False, showticklabels=False)

        # highlight points with selection other graph
        # if selected_data is None:
        #     print("No selected data scatterplot")
        #     selected_index = self.df.index  # show all
        # else:
        #     print("Scatterplot selected: ", selected_data)
        #     selected_index = [  # show only selected indices
        #         x.get('pointIndex', None)
        #         for x in selected_data['points']
        #     ]
        # print("first sc",self.fig.data[0])
        # self.fig.data[0].update(
        #     selectedpoints=selected_index,

        #     # color of selected points
        #     selected=dict(marker=dict(color="purple")),

        #     # color of unselected pts
        #     unselected=dict(marker=dict(color='rgb(200,200,200)', opacity=0.9))
        # )
        # print("updated sc",self.fig.data[0])
        # update axis titles
        # self.fig.update_layout(
        #     xaxis_title=self.feature_x,
        #     yaxis_title=self.feature_y,
        # )

        return self.fig
