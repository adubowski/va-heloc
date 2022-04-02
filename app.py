from heloc_app.main import app
from heloc_app.views.menu import generate_control_card, generate_description_card, feature_selection
from heloc_app.views.scatterplot import Scatterplot
from heloc_app.views.boxplot import Boxplot
from heloc_app.views.barchart import Barchart
from heloc_app.views.histogram import Histogram
from heloc_app.data import get_data, tsne
from dash import html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output


if __name__ == '__main__':
    # Create data
    X_test_transformed, X_test, y_test, features, y_predict_prob = get_data()
    columns = features[features.columns[1:]].columns.tolist()
    X_embed = tsne(X_test_transformed)
    dic = {
        "y_test": y_test,
        "y_predict": y_predict_prob,
        "Embedding 1": X_embed[:, 0],
        "Embedding 2": X_embed[:, 1]
    }
    X_copy = X_test.copy()
    for i, k in dic.items():
        X_copy[i] = k

    graph_types = {
        "Scatterplot": Scatterplot("Scatterplot", "Embedding 1", "Embedding 2", X_copy),
        "Histogram": Histogram("Histogram", columns[0], columns[1], features),
        "Boxplot": Boxplot("Boxplot", None, None, features)
    }

    # Initialization
    plot1 = graph_types.get("Scatterplot")
    plot2 = Histogram("Histogram", columns[0], columns[1], features)

    app.layout = html.Div(
        id="app-container",
        children=[
            # Left column
            html.Div(
                id="left-column",
                className="three columns",
                children=[
                    generate_description_card(),
                    generate_control_card(),
                    feature_selection()
                ]
            ),

            # Right column
            html.Div(
                id="right-column",
                className="nine columns",
                children=[
                    plot1,
                    plot2,
                ],
            ),
        ],
    )

    # Define interactions
    @app.callback(
        Output(plot1.html_id, "figure"), [
            Input("color-type-1", "value"),
            # Input("columns-1", "value"),
            # Input("columns-2", "value"),
            # Input(plot1.html_id, 'selectedData')

        ])
    def update_first(sccolor):
        print("hello")
        # plot1 = graph_types.get(graph_type)
        return plot1.update(sccolor)


    @app.callback(
        Output(plot2.html_id, "figure"), 
        Output("div-hist", "style"),
        Output("div-box", "style"),
        [
        Input("graph-type-2", "value"),
        Input("columns-3", "value"),
        Input("columns-4", "value"),
        Input("col-group", "value"),
        #Input(scatterplot1.html_id, 'selectedData')
    ])
    def update_second(graph_type, col1, col2, colgroup):
        hide = {"display": "none"}
        show = {"display": "block"}
        plot2 = graph_types.get(graph_type)
        if graph_type == "Histogram":
            
            return plot2.update(col1, col2),show,hide
        
        elif graph_type == "Boxplot":
            if colgroup == "Trade":
                cols = ["MSinceOldestTradeOpen", "MSinceMostRecentTradeOpen"]
            elif colgroup == "Inquiry":
                cols = ["NumInqLast6M", "MSinceMostRecentInqexcl7days"]
            else:
                cols = ["MaxDelq/PublicRecLast12M", "MaxDelqEver"]
            return plot2.update(cols),hide,show


    @app.callback(
        Output('modal', 'style'), [
            Input("features-button", "n_clicks")
        ])
    def show_feature(n):
        print(n)
        if n > 0:
            print("here")
            return {"display": "block"}
        return {"display": "none"}


    # Close modal by resetting info_button click to 0
    @app.callback(Output('features-button', 'n_clicks'),
                  [Input('modal-close-button', 'n_clicks')])
    def close_feature(n):
        return 0


    app.run_server(debug=False, dev_tools_ui=False)
