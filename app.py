from heloc_app.main import app
from heloc_app.views.menu import generate_description_card, \
    local_interactions, data_interactions, feature_selection
from heloc_app.views.scatterplot import Scatterplot
from heloc_app.views.boxplot import Boxplot
from heloc_app.views.lime_barchart import LimeBarchart
from heloc_app.views.scatter_matrix import DataScatterMatrix
from heloc_app.views.histogram import Histogram
from heloc_app.data import get_data, get_fitted_model, get_predictions, \
    get_counterfactual_df, get_x_y, get_scatterplot_df, get_numerical_cols
from dash import html, dash_table, dcc
from dash.dependencies import Input, Output
from heloc_app.config import DATA_COLS, SSC_COLS
# ignore known warnings
from warnings import simplefilter
import time

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

if __name__ == '__main__':
    start = time.time()
    # Create data
    features = get_data()
    X_transformed, X_test, y_test, X_train, y_train = get_x_y(features)
    numerical = get_numerical_cols(X_test)
    model = get_fitted_model(X_train, y_train)
    scatterplot_X = get_scatterplot_df(X_transformed, X_test, y_test,
                                       model)
    graph_types = {
        # Local Explanations tab
        "Scatterplot": Scatterplot("Scatterplot", "Embedding 1", "Embedding 2",
                                   scatterplot_X),
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

    @app.callback(
        Output(plot1.html_id, "figure"),
        Output("color-type-1", "options"),
        [
            Input("color-type-1", "value"),
            Input('modal-close-button', 'n_clicks'),
            Input("features-button", "n_clicks"),
            Input('f-checklist', 'value'),
        ])
    def update_first(scplt_color, close, selected_features, cols):
        if close == 0:
            options = [{"label": i, "value": i} for i in SSC_COLS]
            return plot1.update(scplt_color, scatterplot_X), options
        elif close == selected_features:
            # Get new train test split on selected cols
            X_trans, X_test, y_test, X_train, y_train = get_x_y(features, cols)
            # Retrain model on new data
            model = get_fitted_model(X_train, y_train)
            new_df = get_scatterplot_df(X_trans, X_test, y_test, model)
            cols_dropdown = ['y_pred', 'y_pred_prob', 'y_test'] + cols
            options = [{"label": i, "value": i} for i in cols_dropdown]
            return plot1.update(scplt_color, new_df), options

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
            cf_df = get_counterfactual_df(
                X_train,
                y_train,
                model,
                numerical,
                X_test,
                clicked['points'][0].get('customdata')[0]
            )
        else:  # added to get rid of errors
            cf_df = get_counterfactual_df(X_train, y_train, model, numerical,
                                          X_test,
                                          X_test.index[0])
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
            children = [generate_description_card(), local_interactions(),
                        feature_selection()]
        return children

    @app.callback(
        Output('modal', 'style'), [
            Input("features-button", "n_clicks"),
            Input('modal-close-button', 'n_clicks'),
            Input('f-checklist', 'value'),
        ])
    def show_feature(open, close, values):
        if open > close:
            return {"display": "block"}
        else:
            return {"display": "none"}

    print("App startup time (s): ", time.time() - start)
    app.run_server(debug=False, dev_tools_ui=False)
