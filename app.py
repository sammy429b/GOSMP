import streamlit as st
import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from optimizer import withoutOptimization, withOptimization, backtest_with_nifty, totalReturn
from data_cleaning import load_and_clean, clean
from questions import questions, calculate_risk_score, calculate_risk_category


def main():
    st.title("Risk Assessment")

    answers = {}
    for question, data in questions.items():
        st.write(data["question"])
        for i, option in enumerate(data["options"], start=1):
            st.write(f"{i}. {option}")

        key = f"q{question}"  # Unique key for each question
        answer = st.radio(f"Select your choice for Question {question}", options=[
                          str(i) for i in range(1, len(data["options"]) + 1)], key=key)
        answers[question] = data["options"][int(answer) - 1]

    risk_score = calculate_risk_score(answers)
    risk_category = calculate_risk_category(risk_score)

    st.write(f"Risk Score: {risk_score}")
    st.write(f"Risk Category: {risk_category}")
    # store the risk category in the session state
    st.session_state.risk_category = risk_category
    # show a button that will redirect to a new page
    if st.button("Portfolio Optimization"):
        st.session_state.page = 1


def showOptimization(timed_df, exp_ret_type, cov_type, weight_type, invest_amount, start_date, num_days, nifty_csv_file):
    st.title("Portfolio without Optimization")
    st.write("stocks:")
    st.write(pd.DataFrame(timed_df.columns, columns=["Stocks"]))
    port_variance, port_volatility, port_annual_return, percent_var, percent_vols, percent_ret = withoutOptimization(
        timed_df)
    st.write(f"Portfolio Variance: {port_variance:.2f}")
    st.write(f"Portfolio Variance in Percentage: {percent_var}")
    st.write(f"Portfolio Volatility: {port_volatility: .2f}")
    st.write(f"Portfolio Volatility in Percentage: {percent_vols}")
    st.write(f"Portfolio Annual Return: {port_annual_return : .2f}")
    st.write(f"Portfolio Annual Return in Percentage: {percent_ret}")

    st.title("Portfolio with Optimization")
    performance, invested, weights, not_invested, r = withOptimization(
        timed_df, exp_ret_type, cov_type, weight_type, invest_amount, start_date)
    st.write(f"Expected annual return: {performance[0] * 100 :.2f}%")
    st.write(f"Annual volatility: {performance[1] * 100 :.2f}%")
    st.write(f"Sharpe ratio: {performance[2] :.2f}")

    st.write("Portfolio Allocation")
    we = pd.DataFrame(weights, index=[0])
    st.write(we)

    # show a pie chart of the portfolio allocation
    fig, ax = plt.subplots()
    ax.pie(we.values[0], labels=we.columns, autopct='%1.1f%%')
    st.write("Portfolio Allocation")
    st.pyplot(fig)

    st.write("Backtesting")
    dats = backtest_with_nifty(
        nifty_csv_file, invest_amount, start_date, num_days, timed_df, weights, not_invested, invested, r)

    # show table of units and price for each stock allocation
    st.write("Stock Allocation")
    st.write(pd.DataFrame(invested).T)

    st.write("Total Investment")
    st.write(
        f'{sum([invested[stock]["allocated"] for stock in invested]) : .2f} INR')

    dats.index = pd.to_datetime(dats['Date'])
    dats = dats.drop(["Date"], axis=1)
    st.write("Portfolio vs Nifty")
    dats.columns = ["Portfolio percent change","Nifty percent change"]
    st.write(dats)
    st.line_chart(data=dats)

    total_portfolio_return, nifty_return = totalReturn(
        nifty_csv_file, start_date, num_days, timed_df, invested)

    st.write("Total Portfolio Return")
    st.write(total_portfolio_return)
    st.write("Nifty Return")
    st.write(nifty_return)


def open_optimization_page():
    # show a button that will redirect to a new page
    st.write("## Portfolio Optimization")

    csv_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # input for date range
    start_date_df = st.date_input(
        "Enter the start date for DataFrame", datetime.date(2019, 1, 5), min_value=datetime.date(2002, 1, 5))

    end_date_df = st.date_input(
        "Enter the end date for DataFrame", datetime.date(2024, 2, 8))
    invest_amount = st.number_input(
        "Enter the amount you want to invest", value=1_00_000)

    start_date = st.date_input(
        "Enter the start date for optimization", datetime.date(2019, 9, 19))
    # ensure that the start_date is  between start_date_df and end_date_df
    if start_date_df > start_date:
        st.error(
            "Start date should be greater than or equal to start date for DataFrame")
        st.stop()

    end_invest_date = st.date_input(
        "Enter the end date for optimization", datetime.date(2024, 2, 1))

    num_days = (end_date_df - start_date_df).days

    # print the number of days
    st.write(f"Number of days: {num_days}")

    num_columns_to_keep = st.number_input(
        "Enter the number of columns to keep", value=100)

    exp_ret_type = {
        "type": "ema",
        "log_returns": True
    }

    cov_type = {
        "type": "exp_cov"
    }

    weight_type = {}

    risk_type = st.session_state.risk_category
    if risk_type == "Low risk":
        weight_type = {
            "type": "efficient_risk",
            "target_volatility": 0.1
        }
    elif risk_type == "Moderate risk":
        weight_type = {
            "type": "efficient_risk",
            "target_volatility": 0.2
        }
    elif risk_type == "High risk":
        weight_type = {
            "type": "efficient_risk",
            "target_volatility": 0.3
        }
    elif risk_type == "Very high risk":
        weight_type = {
            "type": "efficient_risk",
            "target_volatility": 0.4
        }
    else:
        weight_type = {
            "type": "efficient_risk",
            "target_volatility": 0.5
        }

    nifty_csv_file = "nifty.csv"

    # button click to show the optimization
    if st.button("Optimize"):
        timed_df = load_and_clean(csv_file)
        timed_df = clean(timed_df, start_date=start_date_df,
                         end_date=end_date_df, num_columns_to_keep=num_columns_to_keep)
        showOptimization(timed_df, exp_ret_type, cov_type,
                         weight_type, invest_amount, start_date, num_days, nifty_csv_file)


if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = 0

    def next_page():
        st.session_state.page += 1

    def prev_page():
        st.session_state.page -= 1

    placeholder = st.empty()

    if st.session_state.page == 0:
        main()

    elif st.session_state.page == 1:
        open_optimization_page()
