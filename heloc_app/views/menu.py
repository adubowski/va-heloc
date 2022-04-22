from dash import dcc, html
from ..config import DATA_GRAPH_TYPES, DATA_COLS, SSC_COLS, DATA_GROUP_TYPES, \
    DATA_COLORS, GLOBAL_PLOT_TYPES


def generate_description_card():
    """
    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("HELOC Group 22"),
            html.Div(
                id="intro",
                children="Analyse local and global predictions of the model. "
                         "In the Local Explanations tab you can select a "
                         "point and (after a couple of seconds) you will "
                         "see local and counterfactual explanations. In the "
                         "data tab you can investigate the correlations in "
                         "the dataset while in the global tab you will see "
                         "different plots showing main contributors to the "
                         "model predictions. Note that due to computation "
                         "constraints, only a sample of test data is used for "
                         "model explanation plots.",
            ),
        ],
    )


def local_interactions():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.Label("Select features for coloring:"),
            dcc.Dropdown(
                id="color-type-1",
                options=[{"label": i, "value": i} for i in SSC_COLS],
                value=SSC_COLS[0],
                clearable=False,
            ),
            html.Br(),
            html.Button('Select Features', id='features-button', n_clicks=0),
        ], style={"textAlign": "float-left"}
    )


def global_interactions():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.Label("Select plot:"),
            dcc.Dropdown(
                id="global-plot-selector",
                options=[{"label": i, "value": i} for i in GLOBAL_PLOT_TYPES],
                value=GLOBAL_PLOT_TYPES[0],
                clearable=False,
            ),
            html.Br(),
        ], style={"textAlign": "float-left"}
    )


def data_interactions():
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.Label("Select plot type:"),
            dcc.Dropdown(
                id="graph-type-2",
                options=[{"label": i, "value": i} for i in DATA_GRAPH_TYPES],
                value=DATA_GRAPH_TYPES[0],
                clearable=False,
            ),
            html.Div(
                id="div-color",
                children=[
                    html.Br(),
                    html.Label("Select features for coloring:"),
                    dcc.Dropdown(
                        id="color-selector-data",
                        options=[{"label": i, "value": i} for i in DATA_COLORS],
                        value=DATA_COLORS[0],
                        clearable=False,
                    ),
                ]
            ),
            html.Div(
                id="div-group",
                children=[
                    html.Br(),
                    html.Label("Select data group type:"),
                    dcc.Dropdown(
                        id="group-type-2",
                        options=[{
                            "label": i,
                            "value": i
                        } for i in DATA_GROUP_TYPES],
                        value=DATA_GROUP_TYPES[0],
                        clearable=False,
                    ),
                ]
            ),

            # Histogram
            html.Div(
                id="div-hist",
                children=[
                    html.Label("Select features for x axis:"),
                    dcc.Dropdown(
                        id="DATA_COLS-3",
                        options=[{"label": i, "value": i} for i in DATA_COLS],
                        value=DATA_COLS[0],
                        clearable=False,
                    ),
                    html.Label("Select features for y axis:"),
                    dcc.Dropdown(
                        id="DATA_COLS-4",
                        options=[{"label": i, "value": i} for i in DATA_COLS],
                        value=DATA_COLS[1],
                        clearable=False,
                    ),
                ],
                style={"display": "none"}
            ),
        ], style={"textAlign": "float-left"}
    )


def feature_selection():
    return html.Div([  # modal div
        html.Div([  # content div
            html.Div([
                'Feature Selection Menu',
            ]),
            dcc.Checklist(
                id="f-checklist",
                options=DATA_COLS,
                value=DATA_COLS
            ),
            html.Br(),
            html.Button('Close', id='modal-close-button', n_clicks=0)
        ],
            style={'textAlign': 'center', },
            className='modal-content',
        ),
    ],
        id='modal',
        className='modal',
        style={"display": "none"},
    )
