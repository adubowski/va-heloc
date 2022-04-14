from dash import dcc, html
from ..config import GRAPH_TYPES, DATA_COLS, SSC_COLS, GROUP_TYPES


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
                children="Analyse local predictions of the model "
                         "with the following DATA_COLS. "
                         "Select a point and wait a couple of seconds "
                         "to see local and counterfactual explanations.",
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
            html.Label("Select DATA_COLS for coloring:"),
            dcc.Dropdown(
                id="color-type-1",
                options=[{"label": i, "value": i} for i in SSC_COLS],
                value=SSC_COLS[0],
                clearable=False,
            ),
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
                options=[{"label": i, "value": i} for i in GRAPH_TYPES],
                value=GRAPH_TYPES[0],
                clearable=False,
            ),
            html.Div(
                id="div-color",
                children=[
                    html.Br(),
                    html.Label("Select DATA_COLS for coloring:"),
                    dcc.Dropdown(
                        id="color-selector-data",
                        options=[{"label": i, "value": i} for i in DATA_COLS],
                        value=DATA_COLS[0],
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
                        } for i in GROUP_TYPES],
                        value=GROUP_TYPES[0],
                        clearable=False,
                    ),
                ]
            ),

            # Histogram
            html.Div(
                id="div-hist",
                children=[
                    html.Label("Select DATA_COLS for x axis:"),
                    dcc.Dropdown(
                        id="DATA_COLS-3",
                        options=[{"label": i, "value": i} for i in DATA_COLS],
                        value=DATA_COLS[0],
                        clearable=False,
                    ),
                    html.Label("Select DATA_COLS for y axis:"),
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
            dcc.Checklist(DATA_COLS, []),
            html.Br(),
            html.Button('Close', id='modal-close-button')
        ],
            style={'textAlign': 'center', },
            className='modal-content',
        ),
    ],
        id='modal',
        className='modal',
        style={"display": "none"},
    )
