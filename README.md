# HELOC model explainer Group 22

This project aims to provide a Visual Analytics solution explaining the 
predictions of the HELOC prediction model. The app is built with a Dash 
template provided in the Visualization (JBI100) course and a number of 
freely available Python libraries, specified in requirements.txt file.

Abstract of the problem statement and suggest work:
The complex and "black box" nature of the machine learning methods hinders 
the connection between the human touch (providing domain expertise)
and the efficiency of the automated decision making methods. 
The challenge that was tackled is about understanding 
and explaining the gap between the machine learning models 
and the ability to understand these models to address the outcome of 
the Home Equity Line of Credit (HELOC) applications. 
In order to introduce explainability to this problem, 
HELOC Model Explainer Group 22 was developed. 
Visual analytics techniques were used to develop this solution
in order to provide understanding, diagnosing and refining to the model. 
The motivation is to enable the users to understand the model 
and thus provide reasoning and gather insights of the decisions made by the model. 
As the insurance and finance sector have high-risk, 
the decisions need to reasoned very well. 
As a result, the solution was able to address the user tasks which were to provide explanations 
(local and global) of the prediction result by also introducing interactivity for the user, 
provide the importance of each attribute and provide an explanation of the dataset.

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
