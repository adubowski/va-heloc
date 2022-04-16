from dash import dcc, html
from ..config import graph_type, columns, color_type, col_group, colorssc, group_type


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
                children="Analyse local predictions of the model with the following columns. "
                         "Select a point and wait a couple of seconds to see local and counterfactual explanations.",
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
            html.Label("Select column for coloring:"),
            dcc.Dropdown(
                id="color-type-1",
                options=[{"label": i, "value": i} for i in colorssc],
                value=colorssc[0],
                clearable=False,
            ),
            html.Br(),
            html.Button('Select Features', id='features-button', n_clicks=0),
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
            options=[{"label": i, "value": i} for i in graph_type],
            value=graph_type[0],
            clearable=False,
            ),
            html.Div(
                id = "div-color",
                children= [
                    html.Br(),
                    html.Label("Select column for coloring:"),
                    dcc.Dropdown(
                        id="color-selector-data",
                        options=[{"label": i, "value": i} for i in columns],
                        value=columns[0],
                        clearable=False,
                    ),
                ]
            ),
            html.Div(
                id = "div-group",
                children = [
                    html.Br(),
                    html.Label("Select data group type:"),
                    dcc.Dropdown(
                    id="group-type-2",
                    options=[{"label": i, "value": i} for i in group_type],
                    value=group_type[0],
                    clearable=False,
                    ),
                ]
            ),
            
            ########### Histogram Div #########
            html.Div(
                id = "div-hist",
                children = [
                    html.Label("Select column for x axis:"),
                    dcc.Dropdown(
                        id="columns-3",
                        options=[{"label": i, "value": i} for i in columns],
                        value=columns[0],
                        clearable=False,
                    ),
                    html.Label("Select column for y axis:"),
                    dcc.Dropdown(
                        id="columns-4",
                        options=[{"label": i, "value": i} for i in columns],
                        value=columns[1],
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
            dcc.Checklist(id = "f-checklist", options = columns, value = columns),
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