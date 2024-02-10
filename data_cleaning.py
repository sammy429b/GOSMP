import numpy as np
from bsedata.bse import BSE
from bsedata.exceptions import InvalidStockException  # Import the exception
import pandas as pd
import yfinance


import os


def setup():
    # check if data folder exists
    if not os.path.exists("data"):
        os.makedirs("data")

    # check if Equity.csv exists
    if not os.path.exists("Equity.csv"):
        print("Equity.csv not found. Please download the file and place it in the root directory.")
        exit(1)

    # check if close_prices.csv exists
    if not os.path.exists("close_prices.csv"):
        download_close_prices()


def load_and_clean(csv_file):
    timed_df = pd.read_csv(csv_file)
    timed_df = timed_df.replace(0, np.nan).fillna(method='ffill')
    timed_df = timed_df.replace(0, np.nan).fillna(method='bfill')

    timed_df['Date'] = pd.to_datetime(timed_df['Date'])
    timed_df.set_index('Date', inplace=True)

    # Drop columns where every entry is 0.0
    timed_df = timed_df.loc[:, (timed_df != 0).any(axis=0)]

    # # # Use the column selection to drop columns where less than the threshold number of values are non-zero
    threshold = 0.70 * len(timed_df)
    timed_df = timed_df.loc[:, (timed_df != 0).sum() >= threshold]

    # Iterate through each column
    for col in timed_df.columns:
        # Calculate the mean of the last 10 non-zero values using rolling and mean
        rolling_mean = timed_df[col].replace(0, np.nan).rolling(window=10, min_periods=1).mean()
        
        # Fill zero values with the calculated rolling mean
        timed_df[col] = timed_df.apply(lambda row: row[col] if row[col] != 0 else rolling_mean[row.name], axis=1)

    nifty_df = pd.read_csv("nifty.csv")
    nifty_df['Date'] = pd.to_datetime(nifty_df['Date'])
    nifty_df.set_index('Date', inplace=True)

    # Step 1: Calculate Nifty percentage returns for each rolling 6-month period
    nifty_returns = nifty_df['Close'].pct_change().rolling(window=126).sum().dropna()
    nifty_returns = nifty_returns.max()

    # Step 2: Check if each stock meets the condition
    retained_stocks = []

    for stock in timed_df.columns:
        stock_returns = timed_df[stock].pct_change().rolling(window=252).sum().dropna()
        
        # Check if any 6-month period has at least 50% of Nifty return
        if (stock_returns[stock_returns >= 1].count() >= 5):
            retained_stocks.append(stock)

    # Step 3: Retain only the columns of retained stocks in stock_df
    timed_df = timed_df[retained_stocks]

    return timed_df


def clean(timed_df, start_date='2010-01-05', end_date='2019-01-05', num_columns_to_keep=100):

    # Use loc to select rows within the specified date range
    timed_df = timed_df.loc[start_date:end_date]

    # random num columns
    timed_df = timed_df.sample(n=num_columns_to_keep, axis=1)
    return timed_df


def download_close_prices():
    bse = BSE(update_codes=True)

    strip_code_data = pd.read_csv("Equity.csv")
    strip_code_data.reset_index(inplace=True)
    tick_data = pd.DataFrame(strip_code_data[["index", "Issuer Name"]])

    def get_stock_quote(id):
        try:
            id = str(id)
            return bse.getQuote(id)
        except InvalidStockException as e:
            return None  # Return None to indicate an issue

    stock_data = pd.DataFrame()

    inactive_count = 0

    for id in tick_data["index"]:
        try:
            print(id)
            stock_quote = get_stock_quote(id)

            # Check if stock_quote is not None before processing
            if stock_quote is not None:
                # Convert the dictionary to a DataFrame row
                row_df = pd.DataFrame.from_dict(stock_quote, orient='index').T

                # Append the row to the result DataFrame
                stock_data = pd.concat([row_df, stock_data])
            else:
                # Increment the counter for inactive stocks
                inactive_count += 1

            print("Inactive Count is ", inactive_count)

        except Exception as e:
            print(e)
            print("Error for ", id)

    close_prices_df = _yf_download(stock_data)

    close_prices_df.to_csv("close_prices.csv")


def _yf_download(stock_data):
    inactive_count = 0
    close_prices_df = pd.DataFrame()

    for symbol in stock_data["securityID"]:
        try:
            d = yfinance.download(
                symbol + ".BO", period="max")  # get quote here

            if d is not None:
                temp_df = pd.DataFrame(d)  # make a df
                # select only one column from df
                close_prices_df[symbol] = temp_df['Close']
            else:
                inactive_count += 1
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            inactive_count += 1

    return close_prices_df
