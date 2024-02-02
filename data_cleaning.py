import pandas as pd


def load_and_clean(csv_file):
    timed_df = pd.read_csv(csv_file)
    timed_df.fillna(0, inplace=True)

    timed_df['Date'] = pd.to_datetime(timed_df['Date'])
    timed_df.set_index('Date', inplace=True)

    # Drop columns where every entry is 0.0
    timed_df = timed_df.loc[:, (timed_df != 0).any(axis=0)]

    # # # Use the column selection to drop columns where less than the threshold number of values are non-zero
    threshold = 0.70 * len(timed_df)
    timed_df = timed_df.loc[:, (timed_df != 0).sum() >= threshold]
    return timed_df


def clean(timed_df, start_date='2010-01-05', end_date='2019-01-05', num_columns_to_keep=50):
    # Set the desired start and end dates

    # Use loc to select rows within the specified date range
    timed_df = timed_df.loc[start_date:end_date]


    # first num columns
    # timed_df = timed_df.iloc[:, :num_columns_to_keep]

    # last num columns
    # timed_df = timed_df.iloc[:, -num_columns_to_keep:]

    # random num columns
    timed_df = timed_df.sample(n=num_columns_to_keep, axis=1)
    return timed_df