# Stock Trading ML Algorithm

## Overview
Based off article: [Medium Article](https://medium.com/towards-data-science/getting-rich-quick-with-machine-learning-and-stock-market-predictions-696802da94fe)


## Requirements

-   Python 3.5+
-   alpha_vantage
-   pandas
-   numpy
-   sklearn
-   keras
-   tensorflow
-   matplotlib

## Documentation

[Blog Post](https://yacoubahmed.me/blog/stock-prediction-ml)



## Train your own model

1. Clone the repo
2. Pip install the requirements `pip install -r requirements.txt`
3. Save the stock price history to a csv file `python save_data_to_csv.py --help`
4. Edit one of the model files to accept the symbol you want
5. Edit model architecture
6. Edit dataset preprocessing / history_points inside util.py
7. Train the model `python tech_ind_model.py` or `python basic_model.py`
8. Try the trading algorithm on the newly saved model `python trading_algo.py`


