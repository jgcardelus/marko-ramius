import numpy as np
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, dash_table

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

def plot_coeff_importance(df, title):
    coeff_importance_figure = go.Figure()
    feature_importance_trace = go.Bar(x=df["variable"], y=df["weight"], marker_color="blue", width=np.repeat(0.65, len(df[:])))
    coeff_importance_figure.add_trace(
        feature_importance_trace
    )
    return coeff_importance_figure
    
data = pd.read_csv("./data.csv", delimiter=";")
var_importance = pd.read_csv("./var_importance.csv")
predictions = pd.read_csv("./predictions.csv")
duration_success = pd.read_csv("./duration_success.csv")
age_success = pd.read_csv("./age_success.csv")

numerical_variables = data.select_dtypes(include='number')
categorical_variables = data.loc[:, ["job", "marital", "education", "default", "housing", "loan", "contact", "y"]]

conf_matrix = confusion_matrix(predictions["y_real"], predictions["y_pred"])
corr = numerical_variables[numerical_variables.columns].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

app = Dash("Analysis of Data")

numerical_analysis = html.Div([
    html.H2("Histograms of numerical variables"),
    dcc.Dropdown(numerical_variables.columns, value=numerical_variables.columns[0], id="numerical-dropdown"),
    dcc.Graph(id="numerical-histogram", figure={})
])

categorical_analysis = html.Div([ 
    html.H2("Pie chart of categorical variables"),
    dcc.Dropdown(categorical_variables.columns, value=categorical_variables.columns[0], id="categorical-dropdown"),
    dcc.Graph(id="categorical-pie", figure={})
])

app.layout = html.Div(
    [
        html.H1("Bank Account Stats"),
        html.Hr(),
        numerical_analysis,
        categorical_analysis,
        html.H2("Variable Importance"),
        html.P("What variables better predict success?"),
        dcc.Graph(figure=plot_coeff_importance(var_importance, "Variable Importance")),
        html.H2("Confusion Matrix of Classifier"),
        dcc.Graph(figure=px.imshow(conf_matrix, text_auto=True, labels=dict(x="Predicted", y="Actual"))),
        html.H2("Correlation of Numerical Variables"),
        dcc.Graph(figure=go.Figure(go.Heatmap(
            z=corr.mask(mask),
            x=corr.columns,
            y=corr.columns,
            colorscale=px.colors.diverging.RdBu,
            zmin=-1,
            zmax=1
        ))),
        html.H2("Duration Success"),
        html.P("How do different duration values predict success?"),
        dcc.Graph(figure=px.histogram(duration_success, x="duration", y="y", histfunc="avg")),
        html.H2("Age Success"),
        html.P("How do different age values predict success?"),
        dcc.Graph(figure=px.histogram(age_success, x="age", y="y", histfunc="avg")),
        html.Hr(),
        html.H1("Raw Data"),
        dash_table.DataTable(data.to_dict("records"), [{"name": i, "id": i} for i in data.columns]),
    ],
    style={
        "font-family": "verdana",
    }
)

@callback(
    Output(component_id="numerical-histogram", component_property="figure"),
    Input(component_id="numerical-dropdown", component_property="value")
)
def update_numerical_histogram(variable):
    print(variable)
    return px.histogram(data, x=variable)

@callback(
    Output(component_id="categorical-pie", component_property="figure"),
    Input(component_id="categorical-dropdown", component_property="value")
)
def update_categorical_pie(variable):
    print(variable)
    return px.pie(data, names=variable)

if __name__ == "__main__":
    app.run_server(debug=True)