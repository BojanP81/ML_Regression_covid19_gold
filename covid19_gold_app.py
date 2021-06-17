import pandas as pd
import numpy as np
import plotly.express as px
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yahoo_fin.stock_info as yf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
covid19 = pd.read_csv(url)

covid19_world = covid19[covid19['location'] == 'World']
covid19_world = covid19_world[['new_cases', 'date']].set_index('date')
covid19_world.index = pd.to_datetime(covid19_world.index)

end_data = datetime.now().date()
gold = yf.get_data('GC=F', datetime(2020, 1, 22), end_date=end_data, interval='1d')
gold.rename(columns={'close': 'Gold_close_price'}, inplace=True)
gold.index.rename('date', inplace=True)

merged = covid19_world.merge(gold, left_index=True, right_index=True)
merged = merged.dropna(axis=0, how='any')

x = np.array(merged.new_cases)
y = np.array(merged.Gold_close_price)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LinearRegression()
lr.fit(x_train.reshape(-1, 1), y_train)
y_pred_lr = lr.predict(x_test.reshape(-1, 1))

R2_lr = metrics.r2_score(y_test, y_pred_lr)
mae_lr = metrics.mean_absolute_error(y_test, y_pred_lr)
mse_lr = metrics.mean_squared_error(y_test, y_pred_lr)

y_pred_lr_df = pd.DataFrame({'Actual value Gold_close_price': y_test, 'Predicted value Gold_close_price': y_pred_lr})
fig = px.line(y_pred_lr_df, title='Linear Regression: Actual vs predicted Gold close price',
              color_discrete_map={'Actual value Gold_close_price': 'rgb(55, 83, 109)',
                                  'Predicted value Gold_close_price': 'rgb(50, 212, 215)'})
fig.update_layout(legend=dict(
        x=0.5,
        y=-0.3,
        xanchor="center",
        yanchor="top",
        orientation="h",
    ),
        legend_title_text='')

rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
rf_regressor.fit(x_train.reshape(-1, 1), y_train)
y_pred_rf = rf_regressor.predict(x_test.reshape(-1, 1))

R2_rf = metrics.r2_score(y_test, y_pred_rf)
mae_rf = metrics.mean_absolute_error(y_test, y_pred_rf)
mse_rf = metrics.mean_squared_error(y_test, y_pred_rf)

y_pred_rf_df = pd.DataFrame({'Actual value Gold_close_price': y_test, 'Predicted value Gold_close_price': y_pred_rf})
fig1 = px.line(y_pred_rf_df, title='Random Forest Regression: Actual vs predicted Gold close price',
               color_discrete_map={'Actual value Gold_close_price': 'rgb(55, 83, 109)',
                                   'Predicted value Gold_close_price': 'rgb(50, 212, 215)'})
fig1.update_layout(legend=dict(
        x=0.5,
        y=-0.3,
        xanchor="center",
        yanchor="top",
        orientation="h",
    ),
        legend_title_text='')
dt_regressor = DecisionTreeRegressor(random_state=0)
dt_regressor.fit(x_train.reshape(-1, 1), y_train)
y_pred_dt = dt_regressor.predict(x_test. reshape(-1, 1))

R2_dt = metrics.r2_score(y_test, y_pred_dt)
mae_dt = metrics.mean_absolute_error(y_test, y_pred_dt)
mse_dt = metrics.mean_squared_error(y_test, y_pred_dt)

y_pred_dt_df = pd.DataFrame({'Actual value Gold_close_price': y_test, 'Predicted value Gold_close_price': y_pred_dt})
fig2 = px.line(y_pred_dt_df, title='Decision Tree Regression: Actual vs predicted Gold close price',
               color_discrete_map={'Actual value Gold_close_price': 'rgb(55, 83, 109)',
                                   'Predicted value Gold_close_price': 'rgb(50, 212, 215)'})
fig2.update_layout(legend=dict(
        x=0.5,
        y=-0.3,
        xanchor="center",
        yanchor="top",
        orientation="h",
    ),
        legend_title_text='')
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x.reshape(-1, 1))
model = LinearRegression()
model.fit(x_poly, y)
y_poly_pred = model.predict(x_poly)

mae_poly = metrics.mean_absolute_error(y, y_poly_pred)
mse_poly = metrics.mean_squared_error(y, y_poly_pred)
rmse_poly = np.sqrt(mean_squared_error(y, y_poly_pred))
R2_poly = r2_score(y, y_poly_pred)

y_pred_poly_reg_df = pd.DataFrame({'Actual value Gold_close_price': y, 'Predicted value Gold_close_price': y_poly_pred})
fig3 = px.line(y_pred_poly_reg_df, title='Polynomial Features: Actual vs predicted Gold close price',
               color_discrete_map={'Actual value Gold_close_price': 'rgb(55, 83, 109)',
                                   'Predicted value Gold_close_price': 'rgb(50, 212, 215)'})
fig3.update_layout(legend=dict(
        x=0.5,
        y=-0.3,
        xanchor="center",
        yanchor="top",
        orientation="h",
    ),
        legend_title_text='')
sv_regressor = SVR(kernel='rbf')
sv_regressor.fit(x_train.reshape(-1, 1), y_train)
y_pred_svm = sv_regressor.predict(x_test.reshape(-1, 1))

mae_sv = metrics.mean_absolute_error(y_test, y_pred_svm)
mse_sv = metrics.mean_squared_error(y_test, y_pred_svm)
R2_sv = metrics.r2_score(y_test, y_pred_svm)

y_pred_dt_svm = pd.DataFrame({'Actual value Gold_close_price': y_test, 'Predicted value Gold_close_price': y_pred_svm})
fig4 = px.line(y_pred_dt_svm, title='Support Vector Regression: Actual vs predicted Gold close price',
               color_discrete_map={'Actual value Gold_close_price': 'rgb(55, 83, 109)',
                                   'Predicted value Gold_close_price': 'rgb(50, 212, 215)'})
fig4.update_layout(legend=dict(
        x=0.5,
        y=-0.3,
        xanchor="center",
        yanchor="top",
        orientation="h",
    ),
        legend_title_text='')

cv = KFold(n_splits=5, random_state=1, shuffle=True)
lr_kfd_scores = cross_val_score(lr, x.reshape(-1, 1), y, cv=cv)
lr_cv_score = lr_kfd_scores.mean()

cv = KFold(n_splits=5, random_state=1, shuffle=True)
rf_kfd_scores = cross_val_score(rf_regressor, x.reshape(-1, 1), y, cv=cv)
rf_cv_score = rf_kfd_scores.mean()

cv = KFold(n_splits=5, random_state=1, shuffle=True)
dt_kfd_scores = cross_val_score(dt_regressor, x.reshape(-1, 1), y, cv=cv)
dt_cv_score = dt_kfd_scores.mean()

cv = KFold(n_splits=5, random_state=1, shuffle=True)
poly_kfd_scores = cross_val_score(model, x_poly, y, cv=cv)
poly_cv_score = poly_kfd_scores.mean()

cv = KFold(n_splits=5, random_state=1, shuffle=True)
sv_kfd_scores = cross_val_score(sv_regressor, x.reshape(-1, 1), y, cv=cv)
sv_cv_score = sv_kfd_scores.mean()

R2 = [R2_lr, R2_rf, R2_dt, R2_poly, R2_sv]
MAE = [mae_lr, mae_rf, mae_dt, mae_poly, mae_sv]
MSE = [mse_lr, mse_rf, mse_dt, mse_poly, mse_sv]
CV = [lr_cv_score, rf_cv_score, dt_cv_score, poly_cv_score, sv_cv_score]

col = {'R2 score': R2, 'Mean Absolute Error': MAE, 'Mean Square Error': MSE, 'K-Fold Cross Validation mean score': CV}
models = ['Mutiple Lin. Reg.', 'Random Forest Reg.', 'Decision Tree Reg.', 'Polynominal Reg.',
          'Support Vector Regression']
metrics = ['R2 score', 'Mean Absolute Error', 'Mean Square Error', 'K-Fold Cross Validation mean score']
performance_df = pd.DataFrame(
    columns=['R2 score', 'Mean Absolute Error', 'Mean Square Error', 'K-Fold Cross Validation mean score'],
    index=models, data=col)

fig_perf = make_subplots(rows=1, cols=4)
fig_perf.add_trace(go.Bar(x=performance_df.index,
                          y=performance_df['R2 score'],
                          name='R2 score',
                          marker=dict(color='rgb(55, 83, 109)'),
                          ), 1, 1)
fig_perf.add_trace(go.Bar(x=performance_df.index,
                          y=performance_df['Mean Absolute Error'],
                          name='Mean Absolute Error',
                          marker=dict(color='rgb(50, 212, 215)'),
                          ), 1, 2)
fig_perf.add_trace(go.Bar(x=performance_df.index,
                          y=performance_df['Mean Square Error'],
                          name='Mean Square Error',
                          marker=dict(color='rgb(49, 170, 155)'),
                          ), 1, 3)
fig_perf.add_trace(go.Bar(x=performance_df.index,
                          y=performance_df["K-Fold Cross Validation mean score"],
                          name="K-Fold Cross Validation mean score",
                          marker=dict(color='rgb(134, 245, 8)'),
                          ), 1, 4)

fig_perf.update_layout(
    title='Performance charts',
    xaxis_tickfont_size=14,
    yaxis=dict(
        titlefont_size=16,
        tickfont_size=12,
    ),
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1
)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

header_card_content = [
    dbc.CardHeader("Welcome to my ML Dashboard"),
    dbc.CardBody(
        [
            html.H5('Overview', className='card-title'),
            html.P(
                "This is my personal data science project completely built in Python in which I study the relationship "
                "between Covid19 new cases and the price of the gold in the NYSE. "
                "In this project I'm using Supervised Machine Learning models to predict a continuous variable, "
                "in this case, Gold close price. "
                "I'm utilizing the pandas, scikit-learn, numpy and yahoo_fin libraries to collect, explore and "
                "pre-process data used in building machine learning models. "
                "For this particular regression problem, I'm comparing the five most commonly used regression models "
                "according to their prediction accuracy. "
                "In the final step of the project, I'm using the Dash Plotly library to show gained insights to the "
                "end-user. ",
                className="card-text",
            ),
            html.P(
                "Please note that this is very simplified example (which can be seen in poor accuracy scores)"
                " as my goal was not to make a highly accurate prediction app but to connect data science techniques "
                "and machine learning within web app dashboard.",
                className="card-text",
            ),
        ]
    ),
]

drop_button_card_content = [
    dbc.CardHeader('Regression models menu'),
    html.Br(),
    dbc.CardBody(
        [
            html.H5('Select one of the regression models:', className='card-title'),
            dcc.Dropdown(id='model_dropdown',

                         options=[
                                {'label': 'Linear Regression', 'value': 'LR'},
                                {'label': 'Random Forest Regressor', 'value': 'RFR'},
                                {'label': 'Decision Tree Regressor', 'value': 'DTR'},
                                {'label': 'Polynomial Features', 'value': 'PF'},
                                {'label': 'Support Vector Regression', 'value': 'SVR'},
                                 ],
                         value='LR',
                         searchable=True,
                         style={'color': '#000000'}),
            html.Br(),
            html.Br(),
            html.Br(),
            html.H5('Enter number of new cases to predict Gold close price:'),
            dbc.Input(id='predict_input', placeholder='Enter number ', type='number'),
            html.Br(),
            dbc.Button(id='submit-button-state', children='Calculate', color="info"),
            html.Br(),
            html.Br(),
            html.Div(id='predict_output'),
        ]
    ),
]

chart1_card_content = [
    dbc.CardHeader('Accuracy'),
    dbc.CardBody(
        [
            html.H5('ML Regression Models performance', className='card-title'),
            dcc.Graph(id='my_bar1', figure=fig_perf)
        ]
    ),
]

chart2_card_content = [
    dbc.CardHeader('ML prediction'),
    dbc.CardBody(
        [
            html.H5('Actual vs predicted', className='card-title'),
            dcc.Graph(id='my_bar2', figure={})
        ]
    ),
]


def serve_layout():
    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(header_card_content, color='secondary', inverse=True)),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(chart1_card_content, color='secondary', inverse=True), width=12),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(drop_button_card_content, color='secondary', inverse=True), width=3),
                    dbc.Col(dbc.Card(chart2_card_content, color='secondary', inverse=True), width=9),
                ],
                className="mb-4",
            ),
        ], style={'padding': '0px 15px 15px 15px'},
    )


app.layout = serve_layout


@app.callback(
    Output('my_bar2', 'figure'),
    [Input(component_id='model_dropdown', component_property='value')]
)
def select_graph(value):
    if value is None:
        raise PreventUpdate
    elif value == 'LR':
        return fig
    elif value == 'RFR':
        return fig1
    elif value == 'DTR':
        return fig2
    elif value == 'PF':
        return fig3
    else:
        return fig4


@app.callback(
    Output(component_id='predict_output', component_property='children'),
    Input(component_id='submit-button-state', component_property='n_clicks'),
    Input(component_id='model_dropdown', component_property='value'),
    State(component_id='predict_input', component_property='value'),

)
def update_output_div(n_clicks, value, input_value):

    if input_value is None:
        raise PreventUpdate
    try:
        if value is None:
                raise PreventUpdate
        elif value == 'LR':
                output = lr.predict([[input_value]]).round(2)
                return u'Predicted price with Linear Regression is: {} $'.format(float(output))
        elif value == 'RFR':
                output = rf_regressor.predict([[input_value]]).round(2)
                return u'Predicted price with Random Forest Regressor is: {} $'.format(float(output))
        elif value == 'DTR':
                output = rf_regressor.predict([[input_value]]).round(2)
                return u'Predicted price with Decision Tree Regressor is: {} $'.format(float(output))
        elif value == 'PF':
                output = model.predict(poly_reg.fit_transform([[input_value]])).round(2)
                return u'Predicted price with Polynomial Features is: {} $'.format(float(output))
        else:
                output = sv_regressor.predict([[input_value]]).round(2)
                return u'Predicted price with Support Vector Regression is: {} $'.format(float(output))
    except ValueError:
            return 'Unable to give prediction'


if __name__ == '__main__':
    app.run_server(debug=True)
