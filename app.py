import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load and prepare data
df = pd.read_csv("medication_demand_data.csv", engine='python')

# Preprocessing
cols_to_impute = ['Temperature', 'Humidity', 'Flu_Cases', 'Pollen_Count', 'Google_Trends', 'Marketing_Spend']
df[cols_to_impute] = df[cols_to_impute].fillna(df[cols_to_impute].median())
df['Sales'] = df['Sales'].round()
df['Flu_Cases'] = df['Flu_Cases'].round()

# Feature Engineering
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
start_date = df['Date'].min()
df['Week'] = ((df['Date'] - start_date).dt.days // 7) + 1
df['Holiday'] = df['Holiday'].fillna('None')
df['Is_Holiday'] = (df['Holiday'] != 'None').astype(int)
df['Is_Flu_Season'] = df['Month'].apply(lambda x: 1 if (x >= 10 or x <= 4) else 0)

def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

df['Season'] = df['Month'].apply(get_season)

holiday_options = ['No holiday', 'New Year', 'Canada Day', 'Thanksgiving', 'Christmas']

# Encode categorical features
df_model = df.copy()
le_med = LabelEncoder()
le_reg = LabelEncoder()
le_season = LabelEncoder()
df_model['Medication'] = le_med.fit_transform(df_model['Medication'].astype(str))
df_model['Region'] = le_reg.fit_transform(df_model['Region'].astype(str))
df_model['Season'] = le_season.fit_transform(df_model['Season'].astype(str))

# Feature list
features = ['Medication', 'Region', 'Season', 'Google_Trends', 'Marketing_Spend',
            'Temperature', 'Flu_Cases', 'Humidity', 'Week', 'Is_Flu_Season',
            'Is_Holiday', 'Pollen_Count']

X = df_model[features]
y = df_model['Sales']

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=70, random_state=10),
    "AdaBoost": AdaBoostRegressor(n_estimators=80, random_state=10),
    "KNN": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(max_depth=7),
    "XGBoost": XGBRegressor(n_estimators=80, random_state=10)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    preds_train = model.predict(X_train)
    results[name] = {
        'model': model,
        'train_preds': preds_train,
        'test_preds': preds_test,
        'train_r2': r2_score(y_train, preds_train),
        'test_r2': r2_score(y_test, preds_test),
        'rmse': mean_squared_error(y_test, preds_test, squared=False)
    }

# Feature importance (Random Forest only)
rf_model = models['Random Forest']
rf_feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Dash App
app = Dash(__name__)
app.title = "Medication Demand Dashboard"
server = app.server 


app.layout = html.Div([
    html.H1("\ud83d\udcca Medication Demand Dashboard"),
    html.P(children='''
        This dashboard presents insights from a 2023 dataset exploring sales of various medications across Canadian cities. The Forecasting tab provides an overview of different machine learning models to forecast sales of medication based on various contextual factors (i.e. marketing spend, season, temprature, etc.)
    '''),
    dcc.Tabs([
        dcc.Tab(label='Data Insights', children=[
            dcc.Graph(figure=px.pie(df, values='Sales', names='Medication', title="Breakdown of Sales by Medication Type")),
            html.P(children = ''' The pie chart above provides an overview of sales by Medication Type; a similar breakdown in number of sales per Medication can be seen (~18% - 22%).'''),
            html.H3(children =''' Select a medication type to view google trend score and marketing spend:'''),
            dcc.Dropdown(
                id='trend-medication-selector',
                options=[{'label': str(med), 'value': str(med)} for med in sorted(df['Medication'].unique())],
                value=str(df['Medication'].unique()[0]),
                placeholder="Select a medication",
                style={'width': '50%', 'margin-bottom': '20px'}
            ),
            dcc.Graph(id='trend-medication-line-chart'),
            dcc.Graph(id='trend-marketing-line-chart'),
            html.P(children='''These plots show how Google search interest and marketing spend vary by month for the selected medication.''')
        ]),
        dcc.Tab(label='Seasonal Insights', children=[
            dcc.Graph(figure=px.pie(df, values='Sales', names='Season', title="Breakdown of Sales by Season")),
            html.P(children = ''' The pie chart above provides an overview of sales by seasons; the greatest number of sales are seen in Winter and the least are seen in Summer.'''),
            dcc.Graph(figure=px.box(df, y ='Sales', x ='Season', color = 'Medication', title="Sales Distribution by Season and Medication")),
            html.P(children = ''' The box chart above provides an overview of sales by season and medication type. In Winter, cold and fever medication had the highest sales volume. In Spring, allergy medication had the highest sales volume. In summer and fall, there are no clear differences in sales for medications.'''),
            dcc.Graph(figure=px.box(df, y ='Sales', x ='Is_Flu_Season', color = 'Medication', 
                    labels={
                     "y": "Sales",
                     "x": "Not Flu Season VS. Flu Season",
                    "color": "Medication Type"}, title="Flu vs Non- Flu Season Medication Sales")),
            html.P(children = ''' The box chart above provides an overview of sales for flu (1) vs non flu (0) season and medication type. Flu season (1) has greater sales for all medication types, except pain medication which is approximately the same.'''),
            html.H3(children =''' Select a medication type to view temprature and sales trends:'''),
            dcc.Dropdown(
                id='medication-selector',
                options=[{'label': str(med), 'value': str(med)} for med in sorted(df['Medication'].unique())],
                value=str(df['Medication'].unique()[0]),
                placeholder="Select a medication",
                style={'width': '50%', 'margin-bottom': '20px'}
            ),
            dcc.Graph(id='medication-line-chart'),
            html.P(children='''The line chart above shows the relationship between temperature and sales for the selected medication. 
                In Winter, cold, cough and fever medication had the highest sales volume. In Spring, allergy medication had the highest sales volume. 
                In summer and fall, there are no clear differences in sales for medications. Pain medication also had no seasonal differences in sales.'''),
        ]),
        dcc.Tab(label='Regional Insights', children=[
            dcc.Graph(figure=px.pie(df, values='Sales', names='Region', title="Sales Distribution by Region")),
            html.P(children = ''' The pie chart above provides an overview of sales by 4 regions in Canada; a similar breakdown in number of sales per region can be seen.'''),
            dcc.Graph(figure=px.box(df, y ='Sales', x ='Season', color = 'Region', title="Medication Sales Distribution by Region")),
            html.P(children = ''' The box chart highlights clear seasonal trends in medication sales, with Winter showing the highest and most varied demand—especially in Toronto and Montreal—due to flu season. Spring sees a moderate allergy-related spike, while Summer remains the most stable. Toronto consistently leads in median sales, underscoring the need to factor in both seasonal and regional dynamics in forecasting.'''),
        ]),
        dcc.Tab(label='Forecasting', children=[
            html.P("Six different machine learning models were created to test how well each model can learn from the data and generalize to new, unseen information and predict medication sales. Two evaluation metrics are calculated for each:"), 
            html.P("1. Root Mean Squared Error (RMSE): Measures how far off the predictions are from the actual sales values — lower is better."), 
            html.P("2. R² (R-squared Score): Measures how much of the variability in sales the model explains — closer to 1 is better."),
            html.H3(children=''' Select a model:'''),
            dcc.Dropdown(
                id='model-selector',
                options=[
                    {'label': 'Random Forest 🏆', 'value': 'Random Forest'},
                    *[
                        {'label': k, 'value': k} 
                        for k in models.keys() if k != 'Random Forest'
                    ]
                ],
                value='Random Forest',
                style={'width': '50%'}
            ),
            html.Div(id='metrics-output', style={'margin-top': 20}),
            dcc.Graph(id='prediction-graph'),
            html.H3("Feature Importance (Random Forest)"),
            html.P("The feature importance scores indicate which variables had the most influence on predicting medication sales. Features such as marketing spend, Google search trends, region, and flu case counts were among the top contributors in our strongest model, offering valuable insight into the key drivers of medication demand."),
            dcc.Graph(figure=px.bar(rf_feature_importance, x='Importance', y='Feature', orientation='h', title="Feature Importance Scores")),
        dcc.Tab(label='Predict a Medications Sales Volume', children = [
            html.H2("Predict Medication Demand Across Regions"),
    html.P("Use the selectors below to define your demand scenario. Predictions are based on your chosen values and the Random Forest Model above."),
    html.Div([
        html.H4("1. Medication Type"),
        html.P("Select the medication for which you want to predict demand."),
        dcc.Dropdown(
            id='medication-input',
            options=[{'label': str(med), 'value': str(med)} for med in sorted(df['Medication'].unique())],
            placeholder="Select Medication"
        ),

        html.H4("2. Google Trend Score"),
        html.P("Pick a level of search interest from Google Trends."),
        dcc.Dropdown(
            id='google-trends-input',
            options=[
                {'label': 'Low Score (25)', 'value': 25},
                {'label': 'Medium Score (55)', 'value': 55},
                {'label': 'High Score (95)', 'value': 95}
            ],
            placeholder="Google Trend Score"
        ),

        html.H4("3. Marketing Spend"),
        html.P("Choose a marketing spend range."),
        dcc.Dropdown(
            id='marketing-spend-input',
            options=[
                {'label': 'Low Spend ($1,000)', 'value': 1000},
                {'label': 'Medium Spend ($2,000)', 'value': 2000},
                {'label': 'High Spend ($4,000)', 'value': 4000}
            ],
            placeholder="Marketing Spend ($)"
        ),

        html.H4("4. Temperature (°C)"),
        html.P("Drag to select the temperature. The app will assign a season based on your input."),
        dcc.Slider(id='temperature-input', min=-30, max=35, step=1, value=10, marks=None, tooltip={"placement": "bottom", "always_visible": True}),

        html.H4("5. Flu Cases"),
        html.P("Enter the expected number of flu cases."),
        dcc.Slider(id='flu-cases-input', min=0, max=1000, step=10, value=100, marks=None, tooltip={"placement": "bottom", "always_visible": True}),

        html.H4("6. Humidity Level"),
        html.P("Choose a humidity level (scale: 0–100)."),
        dcc.Slider(id='humidity-input', min=0, max=100, step=1, value=5, marks=None, tooltip={"placement": "bottom", "always_visible": True}),

        html.H4("7. Week Number"),
        html.P("Select a week number (1–52). The app will use this to assign flu season context."),
        dcc.Slider(id='week-input', min=1, max=52, step=1, value=10, marks=None, tooltip={"placement": "bottom", "always_visible": True}),

        html.H4("8. Holiday"),
        html.P("Choose a holiday. If no holiday applies, select 'No holiday'."),
        dcc.Dropdown(
            id='holiday-input',
            options=[{'label': h, 'value': h} for h in holiday_options],
            value='No holiday',
            placeholder="Select Holiday"
        ),
    ], style={'marginBottom': '30px'}),
    html.Button("Predict Demand", id='predict-button', n_clicks=0, style={'marginTop': '20px'}),
    html.Div(id='region-demand-bar') 
])
        ])
    ])
])

# Google Trends & Marketing Spend medication selector
@app.callback(
    Output('trend-medication-line-chart', 'figure'),
    Output('trend-marketing-line-chart', 'figure'),
    Input('trend-medication-selector', 'value')
)
def update_trend_medication_line_chart(medication):
    filtered_df = df[df['Medication'] == medication]

    # Google Trends
    trends = filtered_df.groupby('Month', as_index=False)['Google_Trends'].mean()
    fig_trends = px.line(trends, x='Month', y='Google_Trends', title=f"Monthly Google Trend Score for {medication}")
    fig_trends.update_traces(mode='lines+markers')

    # Marketing Spend
    spend = filtered_df.groupby('Month', as_index=False)['Marketing_Spend'].mean()
    fig_spend = px.line(spend, x='Month', y='Marketing_Spend', title=f"Monthly Marketing Spend for {medication}")
    fig_spend.update_traces(mode='lines+markers')

    return fig_trends, fig_spend

# Medication selector for temperature vs sales
@app.callback(
    Output('medication-line-chart', 'figure'),
    Input('medication-selector', 'value')
)
def update_medication_line_chart(medication):
    filtered_df = df[df['Medication'] == medication]

    grouped = (
        filtered_df
        .groupby(['Season', 'Temperature'], as_index=False)
        .agg({'Sales': 'mean'})
        .sort_values(by='Temperature')
    )

    fig = px.line(
        grouped,
        x='Temperature',
        y='Sales',
        color='Season',
        title=f"Average Sales vs Temperature for {medication}",
        labels={'Sales': 'Average Sales', 'Temperature': 'Temperature (°C)'}
    )
    fig.update_traces(mode='lines+markers')
    return fig

# Forecasting model results
@app.callback(
    Output('metrics-output', 'children'),
    Output('prediction-graph', 'figure'),
    Input('model-selector', 'value')
)
def update_model_display(model_name):
    res = results[model_name]
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=y_test.values, mode='lines+markers', name='Actual'))
    fig.add_trace(go.Scatter(y=res['test_preds'], mode='lines+markers', name='Predicted'))
    fig.update_layout(title=f"{model_name} Predictions vs Actual", xaxis_title='Index', yaxis_title='Sales')

    metrics = html.Div([
        html.H4(f"{model_name} Performance"),
        html.P(f"Train R²: {res['train_r2']:.2f}"),
        html.P(f"Test R²: {res['test_r2']:.2f}"),
        html.P(f"Test RMSE: {res['rmse']:.2f}")
    ])
    return metrics, fig

# Prediction Callback 
@app.callback(
    Output('region-demand-bar', 'children'),
    Input('predict-button', 'n_clicks'),
    State('medication-input', 'value'),
    State('google-trends-input', 'value'),
    State('marketing-spend-input', 'value'),
    State('temperature-input', 'value'),
    State('flu-cases-input', 'value'),
    State('humidity-input', 'value'),
    State('week-input', 'value'),
    State('holiday-input', 'value')
)
def predict_by_region(n_clicks, medication, trend, spend, temp, flu_cases, humidity, week, holiday):
    if not all([medication, trend, spend, temp is not None, flu_cases is not None, humidity is not None, week]):
        return html.Div("Please fill in all fields to generate a prediction.")

    if temp <= -10:
        season = 'Winter'
    elif -9 <= temp <= 10:
        season = 'Fall'
    elif 11 <= temp <= 19:
        season = 'Spring'
    else:
        season = 'Summer'

    flu_season_weeks = list(range(1, 18)) + list(range(40, 53))
    is_flu_season = 1 if week in flu_season_weeks else 0
    is_holiday = 0 if holiday == 'No holiday' else 1

    region_names = df['Region'].unique()
    rows = []
    for region in region_names:
        rows.append({
            'Medication': medication,
            'Region': region,
            'Season': season,
            'Google_Trends': trend,
            'Marketing_Spend': spend,
            'Temperature': temp,
            'Flu_Cases': flu_cases,
            'Humidity': humidity,
            'Week': week,
            'Is_Flu_Season': is_flu_season,
            'Is_Holiday': is_holiday,
            'Pollen_Count': 250
        })

    df_pred = pd.DataFrame(rows)
    df_pred['Medication'] = le_med.transform(df_pred['Medication'])
    df_pred['Region'] = le_reg.transform(df_pred['Region'])
    df_pred['Season'] = le_season.transform(df_pred['Season'])
    X_input = scaler.transform(df_pred[features])
    df_pred['Predicted_Sales'] = rf_model.predict(X_input)
    df_pred['Region'] = region_names

    df_sorted = df_pred.sort_values(by='Predicted_Sales', ascending=False)
    max_value = df_sorted['Predicted_Sales'].max()

    styled_output = [
        html.Div([
            html.Span("🏆 Highest Demand! ", style={'color': '#e74c3c', 'fontWeight': 'bold'}) if row['Predicted_Sales'] == max_value else None,
            html.Span(f"{row['Region']} Predicted Sales: ${row['Predicted_Sales']:,.2f}",
                      style={'margin': '6px 0', 'color': '#2c3e50'})
        ], style={'padding': '5px 0'})
        for _, row in df_sorted.iterrows()
    ]

    return html.Div(styled_output, style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px'})

if __name__ == '__main__':
    app.run(debug=True, port=8051)

