# HELOC model explainer Group 22

This project aims to provide a Visual Analytics solution explaining the 
predictions of the HELOC prediction model. The app is built with a Dash 
template provided in the Visualization (JBI100) course and a number of 
freely available Python libraries, specified in requirements.txt file.

[comment]: <> (TODO: Add abstract summary)

## Prerequisites
In order to successfully run the application, a number of Python libraries 
are required. Firstly, make sure you have PIP installed and then run:

```commandline
pip install -r requirements.txt
```

## Run the application
The application can be started by running the following command from the 
repository's root directory:

```commandline
python app.py
```

or 

```commandline
python -m app
```

After the server has been started, one can access the app via web-browser at 
[http://127.0.0.1:8050/](http://127.0.0.1:8050/).

## Other
Please note that the application has been developed as a research project 
and is therefore not suitable for production deployment in the current state.
Recommended viewing resolution is 1080p (Full HD) as we found it challenging 
to set dynamic resolution in the plotly library.