from heloc_app.main import app
from heloc_app.views.menu import generate_description_card, \
    local_interactions, data_interactions, feature_selection, \
    global_interactions
from heloc_app.views.scatterplot import Scatterplot
from heloc_app.views.boxplot import Boxplot
from heloc_app.views.permutation_boxplot import PermutationBoxplot
from heloc_app.views.lime_barchart import LimeBarchart
from heloc_app.views.cf_barchart import CFBarchart
from heloc_app.views.scatter_matrix import DataScatterMatrix
from heloc_app.views.histogram import Histogram
from heloc_app.data import get_data, get_fitted_model, get_x_y, \
    get_scatterplot_df, get_numerical_cols, get_shap_plot
from dash import html, dcc
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
    origin = get_data()
    X_transformed, X_test, y_test, X_train, y_train = get_x_y(origin)
    sample = 250
    X_global_test, y_global_test = X_test[:sample], y_test[:sample]
    columns = list(X_global_test.columns)
    numerical = get_numerical_cols(X_test)
    model = get_fitted_model(X_train, y_train)
    scatterplot_X = get_scatterplot_df(X_transformed, X_test, y_test, model)
    graph_types = {
        # Global Explanations tab
        "Permutation Importances":
            PermutationBoxplot("Permutation Importances", origin),
        # Local Explanations tab
        "Scatterplot": Scatterplot("Scatterplot", "Embedding 1", "Embedding 2",
                                   scatterplot_X),
        "LimeBarchart": LimeBarchart("LimeBarchart", X_test.index[0], X_train,
                                     X_test, model),
        "CFBarchart": CFBarchart("CFBarchart", X_test.index[0], X_train,
                                 y_train, X_test, model),
        # Data tab
        "Scatterplot Matrix": DataScatterMatrix("DataScatterMatrix", origin,
                                                model),
        "Boxplot": Boxplot("Boxplot", origin),
        "Histogram": Histogram("Histogram", DATA_COLS[0], DATA_COLS[1], origin),
    }

    # Initialization
    scatterplot = graph_types.get("Scatterplot")
    cf_barchart = graph_types.get("CFBarchart")
    lime_barchart = graph_types.get("LimeBarchart")
    data_plot = graph_types.get("Scatterplot Matrix")
    global_boxplot = graph_types.get("Permutation Importances")
    # shap_summary_plot = graph_types.get("SHAP Summary Plot")
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
                dcc.Tabs(
                    id='tabs',
                    value='local_exp',
                    style={'height': '5vh'},
                    children=[
                        dcc.Tab(
                            id='global_plots',
                            label='Global explanations',
                            value='global_exp',
                            children=[get_shap_plot(model, X_global_test)]
                        ),
                        dcc.Tab(label='Local explanations',
                                value='local_exp',
                                children=[
                                    scatterplot,
                                    html.Div(
                                        id='lime',
                                        className="seven columns",
                                        children=[lime_barchart]
                                    ),
                                    html.Div(
                                        id='cf',
                                        className="five columns",
                                        children=[cf_barchart]
                                    )]
                                ),
                        dcc.Tab(label='Data',
                                value='data',
                                children=[data_plot]),
                    ]),
                id="right-column",
                className="ten columns",
            ),
        ],
    )


    @app.callback(
        Output(scatterplot.html_id, "figure"),
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
            return scatterplot.update(scplt_color, scatterplot_X), options
        elif close == selected_features:
            # Get new train test split on selected cols
            X_trans, X_test, y_test, X_train, y_train = get_x_y(origin, cols)
            # Retrain model on new data
            model = get_fitted_model(X_train, y_train)
            new_df = get_scatterplot_df(X_trans, X_test, y_test, model)
            cols_dropdown = ['y_pred', 'y_pred_prob', 'y_test'] + cols
            options = [{"label": i, "value": i} for i in cols_dropdown]
            return scatterplot.update(scplt_color, new_df), options


    @app.callback(
        Output(data_plot.html_id, "figure"),
        Output("div-hist", "style"),
        Output("div-color", "style"),
        Output("div-group", "style"), [
            Input("color-selector-data", "value"),
            Input("graph-type-2", "value"),
            Input("group-type-2", "value"),
            Input("DATA_COLS-3", "value"),
            Input("DATA_COLS-4", "value"),
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
        Output(lime_barchart.html_id, "figure"),
        [Input(scatterplot.html_id, "clickData")]
    )
    def update_lime(clicked):
        if clicked is not None:
            return lime_barchart.update(
                clicked['points'][0].get('customdata')[0])
        return lime_barchart.update(X_test.index[0])


    @app.callback(
        Output(cf_barchart.html_id, "figure"),
        [Input(scatterplot.html_id, "clickData")]
    )
    def update_cf(clicked):
        if clicked is not None:
            return cf_barchart.update(clicked['points'][0].get('customdata')[0])
        return cf_barchart.update(X_test.index[0])


    @app.callback(
        Output("left-column", "children"), [
            Input("tabs", "value"),
        ]
    )
    def update_interactions(tab):
        if tab == 'data':
            children = [generate_description_card(), data_interactions()]
        elif tab == 'local_exp':
            children = [generate_description_card(), local_interactions(),
                        feature_selection()]
        else:
            children = [generate_description_card(),
                        global_interactions(),
                        feature_selection()
                        ]
        return children


    @app.callback(
        Output("global_plots", "children"), [
            Input("global-plot-selector", "value"),
        ]
    )
    def update_global_plot(plot_selected):
        if plot_selected == 'SHAP Bar Plot':
            children = [get_shap_plot(model, X_global_test)]
        elif plot_selected == 'SHAP Decision Plot':
            children = [get_shap_plot(model, X_global_test,
                                      plot_type='decision')]
        elif plot_selected == 'SHAP Summary Plot':
            children = [get_shap_plot(model, X_global_test,
                                      plot_type='summary')]
        else:
            children = [global_boxplot.update(model, X_global_test,
                                              y_global_test)]
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
