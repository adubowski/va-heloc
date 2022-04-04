from heloc_app.main import app
from heloc_app.views.menu import generate_description_card, local_interactions, data_interactions
from heloc_app.views.scatterplot import Scatterplot
from heloc_app.views.boxplot import Boxplot
from heloc_app.views.lime_barchart import LimeBarchart
from heloc_app.views.scatter_matrix import DataScatterMatrix
from heloc_app.views.histogram import Histogram
from heloc_app.data import get_data, tsne
from dash import html, dash_table, dcc
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output
import json
from dice_ml import Data, Model, Dice
# ignore known warnings
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

if __name__ == '__main__':
    # Create data
    X_test_transformed, X_test, y_test, features, y_predict_prob, X_train, y_train, \
    model, numerical = get_data()
    columns = features[features.columns[1:]].columns.tolist()
    X_embed = tsne(X_test_transformed)
    dic = {
        "y_test": y_test.astype(str),
        "y_predict": y_predict_prob,
        "Embedding 1": X_embed[:, 0],
        "Embedding 2": X_embed[:, 1]
    }
    X_copy = X_test.copy()
    for i, k in dic.items():
        X_copy[i] = k

    graph_types = {
        "Scatterplot": Scatterplot("Scatterplot", "Embedding 1", "Embedding 2", X_copy),
        "LimeBarchart": LimeBarchart("LimeBarchart", X_test.index[0], X_train, X_test, model),
        ###### For Data tab
        "Scatterplot Matrix": DataScatterMatrix("DataScatterMatrix", features, model),
        "Boxplot": Boxplot("Boxplot", features),
        "Histogram": Histogram("Histogram", columns[0], columns[1], features),
    }


    def counter(X_train, y_train, X_test, model, numerical, pointindex):
        # DiCE counterfactual explanations
        df = X_train.copy()
        df['y'] = y_train.copy()
        data = Data(
            dataframe=df,
            continuous_features=numerical,
            outcome_name='y'
        )
        m = Model(model=model, backend='sklearn')
        dice = Dice(data, m, method='random')
        e = dice.generate_counterfactuals(X_test.loc[[pointindex]], total_CFs=1,
                                          desired_class="opposite")

        d = json.loads(e.to_json())
        cfs_list = d['cfs_list'][0][0][:20]
        test_data = d['test_data'][0][0][:20]
        cf_df = pd.DataFrame(
            [test_data, cfs_list],
            columns=d['feature_names'],
            index=['Actual', 'Closest CounterFactual']
        )
        nunique = cf_df.nunique()
        cols_to_drop = nunique[nunique == 1].index

        output = cf_df.drop(cols_to_drop, axis=1)
        output['index'] = output.index.tolist()
        return output


    # Initialization
    plot1 = graph_types.get("Scatterplot")
    df = counter(X_train, y_train, X_test, model, numerical, X_test.index[0])
    plot2 = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
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
                dcc.Tabs(id='tabs', value='local exp', children=[
                    dcc.Tab(label='Local explanations', value='local exp', children=[plot1, plot3, plot2]),
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
    def update_first(sccolor):
        return plot1.update(sccolor)


    # Define interactions
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
    def update_data_plot(sccolor, graph, group, col_x, col_y):
        data_plot = graph_types.get(graph)

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
            return data_plot.update(cols, sccolor), hide, show, show
        elif graph == "Histogram":
            return data_plot.update(col_x, col_y), show, hide, hide
        else:  # Boxplot
            return data_plot.update(cols), hide, hide, show


    # Define interactions
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
            print("Update data table")
            print(clicked)
            df = counter(X_train, y_train, X_test, model, numerical,
                         clicked['points'][0].get('customdata')[0])
            data = df.to_dict('records')
            columns = [{"name": i, "id": i} for i in df.columns]
            return data, columns


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

    app.run_server(debug=False, dev_tools_ui=False)
