import streamlit as st
import datetime
import pandas as pd
import numpy as np

from optimizer import withoutOptimization, withOptimization, backtest_with_nifty
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

    # show a button that will redirect to a new page
    if st.button("Portfolio Optimization"):
        st.session_state.page = 1


def showOptimization(timed_df, exp_ret_type, cov_type, weight_type, invest_amount, start_date, num_days, nifty_csv_file):

    st.title("Portfolio without Optimization")
    port_variance, port_volatility, port_annual_return, percent_var, percent_vols, percent_ret = withoutOptimization(
        timed_df)
    st.write(f"Portfolio Variance: {port_variance}")
    st.write(f"Portfolio Volatility: {port_volatility}")
    st.write(f"Portfolio Annual Return: {port_annual_return}")
    st.write(f"Portfolio Variance in Percentage: {percent_var}")
    st.write(f"Portfolio Volatility in Percentage: {percent_vols}")
    st.write(f"Portfolio Annual Return in Percentage: {percent_ret}")

    st.title("Portfolio with Optimization")
    performance, assest, weights = withOptimization(
        timed_df, exp_ret_type, cov_type, weight_type)
    st.write(f"Expected annual return: {performance[0]}")
    st.write(f"Annual volatility: {performance[1]}")
    st.write(f"Sharpe ratio: {performance[2]}")

    st.write("Portfolio Allocation")
    for key, value in weights.items():
        st.write(f"{key}: {value:.2f}%")

    # st.write("Discrete Allocation")
    # # convert start_date to string
    # remainder, weights = DiscreteAllocation(
    #     timed_df, weight, invest_amount, start_date.strftime("%Y-%m-%d"))
    # st.write(f"Remainder: {remainder}")
    # for key, value in weights.items():
    #     st.write(f"{key}: {value}")

    st.write("Backtesting")
    dats = backtest_with_nifty(
        nifty_csv_file, invest_amount, start_date, num_days, timed_df, weights)
    st.write(dats)
    st.write("Portfolio vs Nifty")
    st.line_chart(dats[["PctChange", "niftyPctChange"]])


def open_optimization_page():
    # show a button that will redirect to a new page
    st.write("## Portfolio Optimization")

    # get the user input
    invest_amount = st.number_input(
        "Enter the amount you want to invest", value=10000)

    start_date = st.date_input(
        "Enter the start date", datetime.date(2010, 1, 5))

    num_days = st.number_input("Enter the number of days", value=3000)

    exp_ret_type = {
        "type": "ema",
        "log_returns": True
    }

    cov_type = {
        "type": "exp_cov"
    }

    weight_type = {
        "type": "efficient_risk",
        "target_volatility": 0.25  # TODO change this value based on user risk score
    }

    nifty_csv_file = "nifty.csv"

    timed_df = load_and_clean("close_dupli.csv")
    timed_df = clean(timed_df)

    # button click to show the optimization
    if st.button("Optimize"):
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
