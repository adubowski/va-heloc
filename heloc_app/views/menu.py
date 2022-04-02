from dash import dcc, html
from ..config import graph_type,columns,color_type


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
                children="HELOC",
            ),
        ],
    )


def generate_control_card():
    """

    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.Label("Graph Type"),
            dcc.Dropdown(
                id="color-type-1",
                options=[{"label": i, "value": i} for i in color_type],
                value=color_type[0],
                clearable=False,
            ),
            html.Br(),
            # html.Label("Columns 1.1"),
            # dcc.Dropdown(
            #     id="columns-1",
            #     options=[{"label": i, "value": i} for i in columns],
            #     value=columns[0],
            #     clearable=False,
            # ),
            # html.Label("Columns 1.2"),
            # dcc.Dropdown(
            #     id="columns-2",
            #     options=[{"label": i, "value": i} for i in columns],
            #     value=columns[1],
            #     clearable=False,
            # ),
            html.Button('Select Features', id="features-button", n_clicks=0),
            html.Br(),
            # html.Div(id='container-button-basic',
            #  children='Enter a value and press submit'),
            html.Br(),
            html.Br(),
            html.Br(),
            html.Label("Graph Type 2"),
            dcc.Dropdown(
                id="graph-type-2",
                options=[{"label": i, "value": i} for i in graph_type],
                value=graph_type[1],
                clearable=False,
            ),
            html.Br(),
            html.Label("Columns 2.1"),
            dcc.Dropdown(
                id="columns-3",
                options=[{"label": i, "value": i} for i in columns],
                value=columns[0],
                clearable=False,
            ),
            html.Label("Columns 2.2"),
            dcc.Dropdown(
                id="columns-4",
                options=[{"label": i, "value": i} for i in columns],
                value=columns[1],
                clearable=False,
            ),
        ], style={"textAlign": "float-left"}
    )

def feature_selection():
    return html.Div([  # modal div
                html.Div([  # content div
                    html.Div([
                'Feature Selection Menu',
            ]),
            dcc.Checklist(columns,[]),
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