from heloc_app.main import app
from heloc_app.views.menu import generate_control_card,generate_description_card
from heloc_app.views.scatterplot import Scatterplot
from heloc_app.views.barchart import Barchart
from heloc_app.views.histogram import Histogram
from heloc_app.data import get_data
from dash import html
import plotly.express as px
from dash.dependencies import Input, Output
from heloc_app.config import columns

if __name__ == '__main__':
    # Create data
    #df = px.data.iris()
    X_test_transformed, y_test_transformed, features = get_data()
    

    graph_types = {
        "Bar Chart" : Barchart("Bar chart", columns[0], columns[1], features),
        "Histogram" : Histogram("Histogram", columns[0], columns[1], features),
    }

    #X-y 
    X = features[features.columns[1:]]
    y = features["RiskPerformance"]

    
    # Initialization
    plot1 = Barchart("Bar chart", columns[0], columns[1], features)
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
                    generate_control_card()
                ]
            ),

            # Right column
            html.Div(
                id="right-column",
                className="nine columns",
                children=[
                    plot1,
                    plot2
                ],
            ),
        ],
    )

   # Define interactions
    @app.callback(
        Output(plot1.html_id, "figure"), [
        Input("graph-type-1", "value"),
        Input("columns-1", "value"),
        Input("columns-2", "value"),
        #Input(plot1.html_id, 'selectedData')

    ])
    def update_first(graph_type, col1, col2):
        plot1 = graph_types.get(graph_type)
        return plot1.update(col1, col2)

    @app.callback(
        Output(plot2.html_id, "figure"), [
        Input("graph-type-2", "value"),
        Input("columns-3", "value"),
        Input("columns-4", "value"),
        #Input(scatterplot1.html_id, 'selectedData')
    ])
    def update_first(graph_type, col1, col2):
        plot2 = graph_types.get(graph_type)
        return plot2.update(col1, col2)


    app.run_server(debug=False, dev_tools_ui=False)