import dash
import numpy as np
import pandas as pd
from math import sqrt
from plotly import tools
from pandas import Series
from datetime import datetime
import plotly.graph_objs as go
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import ExponentialSmoothing

app = dash.Dash(__name__)

train = pd.read_csv('Train_SU63ISt.csv')
test = pd.read_csv('Test_0qrQsBZ.csv')

train_original = train.copy()
test_original = test.copy()

train['Datetime'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
test['Datetime'] = pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
train_original['Datetime'] = pd.to_datetime(train_original.Datetime, format='%d-%m-%Y %H:%M')
test_original['Datetime'] = pd.to_datetime(test_original.Datetime, format='%d-%m-%Y %H:%M')

for tm in (train, test, train_original, test_original):
	tm['year'] = tm.Datetime.dt.year
	tm['month'] = tm.Datetime.dt.month
	tm['day'] = tm.Datetime.dt.day
	tm['Hour'] = tm.Datetime.dt.hour
# print(train_original.head())

#| monday = 0 | tuesday = 1 | wednesday = 2 | thursday = 3 | friday = 4 | saturday = 5 | sunday = 6 |
train['day of week'] = train['Datetime'].dt.dayofweek
temp = train['Datetime']

def applyer(row): # 1 for the weekday (5 or 6) else 0
	if row.dayofweek == 5 or row.dayofweek == 6:
		return 1
	else: return 0

temp2 = train['Datetime'].apply(applyer)
train['weekday'] = temp2

train.index = train['Datetime']
df = train.drop('ID', axis=1)
ts = df['Count']
# print(df.index)

years = list(set(train['year']))
year_means = list(train.groupby('year')['Count'].mean())
months = list(set(train['month']))
month_means = list(train.groupby('month')['Count'].mean())
weekdays = list(set(train['day of week']))
weekday_means = list(train.groupby('day of week')['Count'].mean())
peakhours = list(set(train['Hour']))
peakhour_means = list(train.groupby('Hour')['Count'].mean())

train = train.drop('ID', axis=1)
train['Timestamp'] = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
train.index = train['Timestamp']

hourly = train.resample('H').mean() # hourly time series, H--hour
daily = train.resample('D').mean() # converting to daily mean, D--daily
weekly = train.resample('W').mean() # converting to weekly mean, W--weekly
monthly = train.resample('M').mean() # converting to monthly mean, M--monthly

def stable_model_tm():
	trace0 = go.Scatter(
		x=hourly.index,
		y=hourly['Count'],
		name='Hourly'
	)
	trace1 = go.Scatter(
		x=daily.index,
		y=daily['Count'],
		name='Daily'
	)
	trace2 = go.Scatter(
		x=weekly.index,
		y=weekly['Count'],
		name='Weekly'
	)
	trace3 = go.Scatter(
		x=monthly.index,
		y=monthly['Count'],
		name='Monthly'
	)

	fig = tools.make_subplots(rows=4, cols=1, specs=[[{}], [{}], [{}], [{}]],
		shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.001)
	fig.append_trace(trace0, 1, 1)
	fig.append_trace(trace1, 2, 1)
	fig.append_trace(trace2, 3, 1)
	fig.append_trace(trace3, 4, 1)
	fig['layout'].update(height=700, width=1200)
	return fig

#########################################################
# splitting data - training and validation

test['Timestamp'] = pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
test.index = test['Timestamp']
test_daily = test.resample('D').mean() # daily mean
# print(test_daily.head())

train_daily = daily.copy() # daily mean
# print(train_daily.head())

Train = train_daily.loc['2012-08-25':'2014-06-24'] # rest of the data
valid = train_daily.loc['2014-06-25':'2014-09-25'] # last 3 months (validation)

tc = list(Train['Count'])
vc = list(valid['Count'])

### naive approach ###
vals = np.asarray(Train.Count)
y_hat = valid.copy()
y_hat['naive'] = vals[len(vals)-1]
naive_rms = sqrt(mean_squared_error(valid['Count'], y_hat['naive']))

### moving average technique ###

### exponential smoothing ###
y_hat_exp = valid.copy()
fit2 = SimpleExpSmoothing(np.asarray(Train['Count'])).fit(smoothing_level=0.6, optimized=False)
y_hat_exp['SES'] = fit2.forecast(len(valid)) # forecasting method
exp_rms = sqrt(mean_squared_error(valid['Count'], y_hat_exp['SES']))

### holt linear trend model ###
y_hat_holt = valid.copy()
fit1 = Holt(np.asarray(Train['Count'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
y_hat_holt['Holt_linear'] = fit1.forecast(len(valid)) # forecasting method
holt_rms = sqrt(mean_squared_error(valid['Count'], y_hat_holt['Holt_linear']))

### holt winter trend model ###
y_hat_winter = valid.copy()
fit3 = ExponentialSmoothing(np.asarray(Train['Count']), seasonal_periods=7, trend='add', seasonal='add').fit()
y_hat_winter['Holt_winter'] = fit3.forecast(len(valid))
holt_winter_rms = sqrt(mean_squared_error(valid['Count'], y_hat_winter['Holt_winter']))
#########################################################

app.layout = html.Div([
	html.Div(html.H2('Time Series Analysis', style={'textAlign' : 'center', 'margin-top' : 30})),
	
	html.Hr(),

	dcc.Tabs(id='time-series-analysis', children=[
		dcc.Tab(label='Understanding data', value='general-hypothesis', children=[
			html.Hr(),
			html.Div([
				html.Div([
					html.H4('Increasing Trend', style={'textAlign' : 'center'}),
					html.H6('There is an increasing trend in the series i.e., the number of count is increasing with respect to time. Certain points, there is a sudden increase in the number of counts. The possible reason behind this could be that on particular day, due to some event the traffic was high.',
						style={'margin-left' : 30, 'margin-top' : 30})
				], className='three columns'),
				html.Div([
					dcc.Graph(
						id='passeger-count',
						figure={
							'data' : [{'x' : df.index, 'y' : ts, 'type' : 'line', 'name' : 'Passenger count'}],
							'layout' : {
								'title' : 'Time Series', 
								'height' : 500,
								'width' : 1000,
								'xaxis' : {'title' : 'Time(year-month)'}, 
								'yaxis' : {'title' : 'Passenger count'}
							}
						}
					)
				], className='nine columns'),
			], className='row', style={'margin-top' : 30}),
			
			html.Hr(),

			html.Div([
				html.Div([
					dcc.Tabs(id='hypothesis-plots', value='hourly', children=[
						dcc.Tab(label='Yearly wise', value='yearly', children=[
							dcc.Graph(
								id='year-plot',
								figure={'data' : [{'x' : years, 'y' : year_means, 'type' : 'bar'}]}
							)
						]),
						dcc.Tab(label='Monthly wise', value='monthly', children=[
							dcc.Graph(
								id='month-plot',
								figure={'data' : [{'x' : months, 'y' : month_means, 'type' : 'bar'}]}
							)
						]),
						dcc.Tab(label='Weekly wise', value='weekly', children=[
							dcc.Graph(
								id='week-plot',
								figure={'data' : [{'x' : weekdays, 'y' : weekday_means, 'type' : 'bar'}]}
							)
						]),
						dcc.Tab(label='Peak hours', value='hourly', children=[
							dcc.Graph(
								id='hour-plot',
								figure={'data' : [{'x' : peakhours, 'y' : peakhour_means, 'type' : 'bar'}]}
							)
						]),
					])
				], className='nine columns'),
				html.Div([
					html.H4('General Hypothesis', style={'textAlign' : 'center'}),
					html.Div([
						html.H6('Traffic will increase as the years pass by.'),
						html.H6('Traffic will be high from May to October.'),
						html.H6('Traffic on Weekdays will be more and less on Weekends.'),
						html.H6('Traffic during the Peak Hours will be high.')
					], style={'margin-left' : 30}),
					html.H4('Conclusion', style={'textAlign' : 'center'}),
					html.Div(id='output-conclusion')
				], className='three columns'),
			], className='row'),
			
			html.Hr(),

			html.Div([
				html.H3('Stable Model of monthly series to reduce noise', style={'textAlign' : 'center'}),
				html.H6('As we have noticed there is a lot of noise in the hourly time series, the best approach is to aggregate the hourly time series into daily, weekly and monthly time series to reduce the noise and make it more stable which would be easier for a model to learn.', style={'textAlign' : 'center', 'margin-left' : 80, 'margin-right' : 80}),
				html.Div([
					dcc.Graph(id='stable-subplots', figure=stable_model_tm())
				], style={'margin-left' : 60}),
				html.H6('The time series becomes more stable when it is aggregated to daily, weekly and monthly basis. But it would be difficult to convert the monthly and weekly predictions to hourly predictions, so we consider daily time series.', style={'textAlign' : 'center', 'margin-left' : 80, 'margin-right' : 80}),
			]),
			
			html.Hr(),

		]),

		dcc.Tab(label='Time series models', value='split-data', children=[
			html.Hr(),
			html.Div([
				html.Div([
					dcc.Graph(
						id='train-valid-graph',
						figure={
							'data' : [
								{'x' : Train.index, 'y' : tc, 'type' : 'line', 'name' : 'Training'},
								{'x' : valid.index, 'y' : vc, 'type' : 'line', 'name' : 'Validation'}
							],
							'layout' : {
								'title' : 'Daily Ridership',
								'height' : 500,
								'width' : 1000,
								'xaxis' : {'title' : 'Datetime'},
								'yaxis' : {'title' : 'Passenger count'}
							}
						}
					)
				], className='nine columns'),
				html.Div([
					html.H6('Splitting is done by selecting last 3 months for Validation data and rest for Training data. We could have used Random splitting, but it will not work effectively on Validation dataset. It would be similar to predicting the old values based on the future values which is of no use.', style={'margin-top' : 60, 'margin-right' : 30}),
				], className='three columns')
			], className='row', style={'margin-top' : 30}),
			
			html.Hr(),

			html.Div([
				html.H3('Modelling Techniques', style={'textAlign' : 'center'}),
				dcc.Tabs(id='modelling-techs', value='naive-method', children=[
					dcc.Tab(label='Naive approach', value='naive-method', children=[
						html.Div([
							html.Div([
								html.Div([
									html.H6('Here, we assume that the next expected point is equal to the last observed point, and ultimately we get a straight horizontal line as the Prediction.'),
									html.H6('Accuracy can be checked by Root Mean Squared Error (Standard deviation of residuals).'),
									html.H6('RMSE = ' + str(naive_rms))
								], style={'margin-left' : 10})
							], className='three columns', style={'margin-top' : 50}),
							html.Div([
								dcc.Graph(
									id='naive-approach',
									figure={
										'data' : [
											{'x' : Train.index, 'y' : tc, 'type' : 'line', 'name' : 'Training'},
											{'x' : valid.index, 'y' : vc, 'type' : 'line', 'name' : 'Validation'},
											{'x' : y_hat.index, 'y' : y_hat['naive'], 'type' : 'line', 'name' : 'Naive forecasting'}
										],
										'layout' : {
											'title' : 'Naive Approach',
											'height' : 500,
											'width' : 1000
										}
									}
								)
							], className='nine columns'),
						], className='row', style={'margin-top' : 40})
					]),
					dcc.Tab(label='Moving average', value='mavg-method', children=[
						html.Div([
							html.Div([
								html.Div([
									html.H6('Here, we take the average of the Passenger counts for the last few periods and predictions are made based on those averages.'),
									html.H6('We took the average of last 10, 20, 50 observations and predicted based on that.'),
									html.Div(id='avg-rmse')
								], style={'margin-left' : 10})
							], className='three columns', style={'margin-top' : 140}),
							html.Div([
								html.Div([
									html.Div([
										html.H6('Select moving average type')
									], className='seven columns', style={'textAlign' : 'right'}),
									html.Div([
										dcc.Dropdown(
											id='avg-options',
											options=[{'label' : s, 'value' : s} for s in [10, 20, 50]],
											value=10
										)
									], className='two columns', style={'textAlign' : 'left'})
								], className='row', style={'margin-top' : 30}),
								html.Div([
									dcc.Graph(id='moving-averages')
								], style={'margin-top' : 10})
							], className='nine columns')
						], className='row')
					]),
					dcc.Tab(label='Exponential smoothing', value='esmoothing-method', children=[
						html.Div([
							html.H6('Exponential smoothing is a rule of thumb for smoothing time series data using the exponential window function. Whereas in the simple moving average the past observations are weighted equally, exponential functions as used to assign exponentially decreasing weights over time.')
						], style={'margin-top' : 40, 'margin-left' : 80, 'margin-right' : 80, 'textAlign' : 'center'}),
						html.Div([
							html.Div([
								html.Div([
									html.H6('In this technique, we assign larger weights to more recent observations than to observations from the distant past.'),
									html.H6('The weights decrease exponentially as observations come from further in the past. Smallest weights are associated to the oldest observations.'),
									html.H6('RMSE = ' + str(exp_rms))
								], style={'margin-left' : 10})
							], className='three columns', style={'margin-top' : 30}),
							html.Div([
								dcc.Graph(
									id='ex-smoothing',
									figure={
										'data' : [
											{'x' : Train.index, 'y' : tc, 'name' : 'Training', 'type' : 'line'},
											{'x' : valid.index, 'y' : vc, 'name' : 'Validation', 'type' : 'line'},
											{'x' : y_hat_exp.index, 'y' : y_hat_exp['SES'], 'name' : 'SES', 'type' : 'line'}
										],
										'layout' : {
											'title' : 'Smooth Exponential Approach',
											'height' : 500,
											'widht' : 1000
										}
									}
								)
							], className='nine columns')	
						], className='row', style={'margin-top' : 10})
					]),
					dcc.Tab(label="Holt's trend model", value='linear_trend-method', children=[
						html.Div([
							html.H4('Series is decomposed into 4 parts - Observed, Trend, Seasonal and Residual.'),
						], style={'textAlign' : 'center', 'margin-top' : 40}),
						html.Div([
							dcc.Tabs(id='holts-model', children=[
								dcc.Tab(label='Linear trend', value='trendy', children=[
									html.Div([
										html.Div([
											html.Div([
												html.H6('It is an extension of simple exponential smoothing that allows forecasting of the data with a trend. The forecast function in this method is a functionof level and trend.'),
												html.H6('We can see an inclined line that signifies trend in the time series.'),
												html.H6('RMSE = ' + str(holt_rms))
											], style={'margin-left' : 10, 'margin-top' : 30})
										], className='three columns'),
										html.Div([
											dcc.Graph(
												id='linear-trend-graph',
												figure={
													'data' : [
														{'x' : Train.index, 'y' : tc, 'type' : 'line', 'name' : 'Training'},
														{'x' : valid.index, 'y' : vc, 'type' : 'line', 'name' : 'Validation'},
														{'x' : y_hat_holt.index, 'y' : y_hat_holt['Holt_linear'], 
															'type' : 'line', 'name' : 'Holt linear model'}
													],
													'layout' : {
														'title' : 'Holt Linear Trend',
														'height' : 500,
														'width' : 1000
													}
												}
											)
										], className='nine columns')
									], className='row', style={'margin-top' : 40})
								]),
								dcc.Tab(label="Winter's model", value='winter', children=[
									html.Div([
										html.Div([
											html.Div([
												html.H6('Datasets which show a similar set of pattern after fixed intervals of a time period suffer from seasonality. Hence we need a method that takes both trend and seasonality to forecast future prices. Applies exponential smoothing to the seasonal components in addition to level and trend.'),
												html.H6('RMSE : ' + str(holt_winter_rms))
											], style={'margin-left' : 10, 'margin-top' : 30})
										], className='three columns'),
										html.Div([
											dcc.Graph(
												id='winter-trend-graph',
												figure={
													'data' : [
														{'x' : Train.index, 'y' : tc, 'type' : 'line', 'name' : 'Training'},
														{'x' : valid.index, 'y' : vc, 'type' : 'line', 'name' : 'Validation'},
														{'x' : y_hat_winter.index, 'y' : y_hat_winter['Holt_winter'], 
															'type' : 'line', 'name' : 'Holt winter model'}
													],
													'layout' : {
														'title' : 'Holt Winter Trend',
														'height' : 500,
														'width' : 1000
													}
												}
											)
										], className='nine columns')
									], className='row', style={'margin-top' : 40}),
								])
							])
						], style={'margin-left' : 15, 'margin-right' : 15, 'margin-top' : 30})
					]),					
				])
			]),
			
			html.Hr(),

		]),
		
		dcc.Tab(label='ARIMA modelling methods', value='parameter-arima', children=[
			html.Hr(),
		]),
	])
])

@app.callback(
	Output('output-conclusion', 'children'),
	[Input('hypothesis-plots', 'value')]
)
def conclude_this(tab):
	if tab == 'yearly':
		return html.P('There is an exponential growth in the traffic with respect to year which validates our Hypothesis.',
			style={'margin-left' : 30})
	elif tab == 'monthly':
		return html.P('There is a decrease in the mean of Passenger count in last three months since months 10, 11 and 12 are not present for the year 2014.',
			style={'margin-left' : 30})
	elif tab == 'weekly':
		return html.P('The days 5 and 6 represents Weekends that infers less Passenger count as compared to other days of week. Hence this validates the third Hypothesis',
			style={'margin-left' : 30})
	elif tab == 'hourly':
		return html.P('It can be inferred that the peak traffic is at 7 PM, and a decreasing trend till 5 AM. The Passenger count starts increasing again and peaks betweem 11 AM and 12.',
			style={'margin-left' : 30})

@app.callback(
	Output('moving-averages', 'figure'),
	[Input('avg-options', 'value')]
)
def moving_averages_bands(value):
	y_hat_avg = valid.copy()
	y_hat_avg['moving_avg'] = Train['Count'].rolling(value).mean().iloc[-1]
	train_trace = go.Scatter(
		x=Train.index,
		y=tc,
		mode='lines',
		name='Training'
	)
	valid_trace = go.Scatter(
		x=valid.index,
		y=vc,
		mode='lines',
		name='Validation'
	)
	mg_avg_trace = go.Scatter(
		x=y_hat_avg.index,
		y=y_hat_avg['moving_avg'],
		mode='lines',
		name='Moving Average ' + str(value) 
	)
	traces = [train_trace, valid_trace, mg_avg_trace]

	layout = go.Layout(
		title='Moving Average - ' + str(value),
		height=500,
		width=1000
	)

	return {'data' : traces, 'layout' : layout}

@app.callback(
	Output('avg-rmse', 'children'),
	[Input('avg-options', 'value')]
)
def mvg_avgs_rmse(value):
	y_hat_avg = valid.copy()
	y_hat_avg['moving_avg'] = Train['Count'].rolling(value).mean().iloc[-1]
	rmse = sqrt(mean_squared_error(valid['Count'], y_hat_avg['moving_avg']))
	return html.H6('RMSE for the last ' + str(value) + ' observations : ' + str(rmse))

external_css = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
for css in external_css:
	app.css.append_css({'external_url' : css})

external_js = ['https://cdnjs.cloudflare.com/ajax/libs/materialize/0.100.2/js/materialize.min.js']
for js in external_js:
	app.scripts.append_script({'external_url' : js})

if __name__ == '__main__':
	app.run_server(debug=True)

