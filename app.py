from sklearn.manifold import TSNE

from heloc_app.main import app
from heloc_app.views.menu import generate_description_card, \
    local_interactions, data_interactions
from heloc_app.views.scatterplot import Scatterplot
from heloc_app.views.boxplot import Boxplot
from heloc_app.views.lime_barchart import LimeBarchart
from heloc_app.views.scatter_matrix import DataScatterMatrix
from heloc_app.views.histogram import Histogram
from heloc_app.data import get_data, get_fitted_model, get_predictions, \
    get_counterfactual_df, get_x_y
from dash import html, dash_table, dcc
from dash.dependencies import Input, Output
from heloc_app.config import DATA_COLS
# ignore known warnings
from warnings import simplefilter
import time

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

if __name__ == '__main__':
    start = time.time()
    # Create data
    features = get_data()
    X_test_transformed, X_test, y_test, X_train, y_train, numerical = \
        get_x_y(features)
    model = get_fitted_model(X_train, y_train)
    y_pred, y_pred_prob = get_predictions(model, X_test)
    X_embed = TSNE(n_components=2, learning_rate='auto', init='pca') \
        .fit_transform(X_test)
    dic = {
        "y_predict": y_pred_prob,
        "y_test": y_test.astype(str),
        "Embedding 1": X_embed[:, 0],
        "Embedding 2": X_embed[:, 1]
    }
    X_copy = X_test.copy()
    for i, k in dic.items():
        X_copy[i] = k

    graph_types = {
        # Local Explanations tab
        "Scatterplot": Scatterplot("Scatterplot", "Embedding 1", "Embedding 2",
                                   X_copy),
        "LimeBarchart": LimeBarchart("LimeBarchart", X_test.index[0], X_train,
                                     X_test, model),
        # Data tab
        "Scatterplot Matrix": DataScatterMatrix("DataScatterMatrix", features,
                                                model),
        "Boxplot": Boxplot("Boxplot", features),
        "Histogram": Histogram("Histogram", DATA_COLS[0], DATA_COLS[1],
                               features),
    }

    # Initialization
    plot1 = graph_types.get("Scatterplot")
    cf_df = get_counterfactual_df(X_train, y_train, model, numerical, X_test,
                                  X_test.index[0])
    plot2 = dash_table.DataTable(
        data=cf_df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in cf_df.columns],
        id='tbl'
    )
    plot3 = graph_types.get("LimeBarchart")
    data_plot = graph_types.get("Scatterplot Matrix")

    app.layout = html.Div(
        id="app-container",
        children=[
            # Left column
            html.Div(
                id="left-column",
                className="two columns",
                children=[
                    generate_description_card(),
                    data_interactions(),
                ]
            ),

            # Right column
            html.Div(
                dcc.Tabs(id='tabs', value='local_exp', children=[
                    dcc.Tab(label='Local explanations', value='local_exp',
                            children=[plot1, plot3, plot2]),
                    dcc.Tab(label='Data', value='data', children=[data_plot]),
                ]),
                id="right-column",
                className="ten columns",
            ),
        ],
    )

    # Define interactions
    @app.callback(
        Output(plot1.html_id, "figure"), [
            Input("color-type-1", "value"),
        ])
    def update_first(selected_color):
        return plot1.update(selected_color)

    @app.callback(
        Output(data_plot.html_id, "figure"),
        Output("div-hist", "style"),
        Output("div-color", "style"),
        Output("div-group", "style"), [
            Input("color-selector-data", "value"),
            Input("graph-type-2", "value"),
            Input("group-type-2", "value"),
            Input("columns-3", "value"),
            Input("columns-4", "value"),
        ])
    def update_data_plot(selected_color, graph, group, col_x, col_y):
        data_plot = graph_types.get(graph)
        cols = DATA_COLS
        show = {"display": "block"}
        hide = {"display": "none"}

        if group == 'Number of':
            cols = [
                "NumTrades60Ever/DerogPubRec",
                "NumTradesOpeninLast12M",
                "NumInqLast6M",
                "NumRevolvingTradesWBalance",
                "NumInstallTradesWBalance",
                "NumBank/NatlTradesWHighUtilization",
                "NumSatisfactoryTrades"
            ]
        elif group == 'Number of months':
            cols = [
                "MSinceOldestTradeOpen",
                "MSinceMostRecentTradeOpen",
                "AverageMInFile",
                "MSinceMostRecentDelq",
                "MSinceMostRecentInqexcl7days",
            ]
        elif group == 'Percentage':
            cols = [
                "PercentTradesNeverDelq",
                "PercentInstallTrades",
                "PercentTradesWBalance",
            ]
        elif group == 'Net Fraction':
            cols = [
                "NetFractionRevolvingBurden",
                "NetFractionInstallBurden",
            ]

        if graph == "Scatterplot Matrix":
            return data_plot.update(cols, selected_color), hide, show, show
        elif graph == "Histogram":
            return data_plot.update(col_x, col_y), show, hide, hide
        else:  # Boxplot
            return data_plot.update(cols), hide, hide, show

    @app.callback(
        Output(plot3.html_id, "figure"),
        [Input(plot1.html_id, "clickData")]
    )
    def update_third(clicked):
        print("Update LIME barchart")
        print(clicked)
        if clicked is not None:
            return plot3.update(clicked['points'][0].get('customdata')[0])
        return plot3.update(X_test.index[0])

    @app.callback(
        Output("tbl", "data"),
        Output("tbl", "columns"), [
            Input(plot1.html_id, "clickData"),
        ]
    )
    def update_table(clicked):
        if clicked is not None:
            print("Data table updated")
            print(clicked)
            cf_df = get_counterfactual_df(
                X_train,
                y_train,
                X_test,
                model,
                numerical,
                clicked['points'][0].get('customdata')[0]
            )
            data = cf_df.to_dict('records')
            cols = [{"name": i, "id": i} for i in cf_df.columns]
            return data, cols

    @app.callback(
        Output("left-column", "children"), [
            Input("tabs", "value"),
        ]
    )
    def update_interactions(tab):
        if tab == 'data':
            children = [generate_description_card(), data_interactions()]
        else:
            children = [generate_description_card(), local_interactions()]
        return children

    # TODO: Implement feature selection
    # @app.callback(
    #     Output('modal', 'style'), [
    #         Input("features-button", "n_clicks")
    #     ])
    # def show_feature(n):
    #     print(n)
    #     if n > 0:
    #         print("here")
    #         return {"display": "block"}
    #     return {"display": "none"}
    #
    # # Close modal by resetting info_button click to 0
    # @app.callback(Output('features-button', 'n_clicks'),
    #               [Input('modal-close-button', 'n_clicks')])
    # def close_feature(n):
    #     return 0

    print("App startup time (s): ", time.time() - start)
    app.run_server(debug=False, dev_tools_ui=False)
