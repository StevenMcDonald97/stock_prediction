# This file will retreive the full daily stock data for the given stock symbol 
# from the Alpha Vantage API and save it in a CSV format
# To usa call: python download_data.py <stock_symbol> 
# ex. python download_data.py AMZN

# from alpha_vantage.timeseries import TimeSeries
import os
import sys
import requests
import pandas as pd

api_key = os.getenv('ALPHAVANTAGE_API_KEY')
base_url = 'https://www.alphavantage.co/query?'
params = {'function': 'TIME_SERIES_DAILY',
		 'symbol': '%s' % (sys.argv[1]),
		 'outputsize':'full',
		 'datatype': 'csv',
		 'apikey': ''}

response = requests.get(base_url, params=params)

#Save CSV to file
with open(('./data/%s.csv'% (sys.argv[1])), 'wb') as file:
	file.write(response.content)


