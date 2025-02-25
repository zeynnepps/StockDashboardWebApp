import yfinance as yf
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# App Configuration
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# Title and Search Bar
st.title("📈 Stock Dashboard")
ticker = st.text_input("Enter Stock or an ETF Ticker (e.g., MSFT, AAPL, SPY):", "") # user input for stock data

if ticker:
    try:
        # Fetch data
        stock = yf.Ticker(ticker)
        info = stock.info

        if 'currentPrice' not in info:  # Check if valid stock data is returned
            raise ValueError("Invalid ticker or no data available.")
        
        # Display Stock Info
        st.header(f"{info.get('shortName', 'N/A')} ({ticker.upper()})")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Price", f"${info.get('currentPrice', 'N/A')}")
            st.metric("Day High", f"${info.get('dayHigh', 'N/A')}")
            st.metric("52-Week High", f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
        with col2:
            st.metric("Open", f"${info.get('open', 'N/A')}")
            st.metric("Day Low", f"${info.get('dayLow', 'N/A')}")
            st.metric("52-Week Low", f"${info.get('fiftyTwoWeekLow', 'N/A')}")

        # Revenue and Market Cap info
        if 'marketCap' in info:
            st.write("**Market Cap:**", f"${info['marketCap']:,}")
        if 'totalRevenue' in info:
            st.write("**Total Revenue:**", f"${info['totalRevenue']:,}")
        if 'revenuePerShare' in info:
            st.write("**Revenue per Share:**", f"${info['revenuePerShare']}")

        # Latest News
        st.subheader("Latest News")
        for news in stock.news[:3]:
            st.write(f"- [{news['title']}]({news['link']})")

        # Balance Sheet (choose quarterly and yearly)
        st.subheader("Balance Sheet")
        balance_type = st.radio("View Balance Sheet:", ["Yearly", "Quarterly"])
        if balance_type == "Yearly":
            st.dataframe(stock.balance_sheet)
        else:
            st.dataframe(stock.quarterly_balance_sheet)

        # Historical Chart
        st.subheader("Historical Chart")
        col1, col2 = st.columns(2)
        with col1:
            period = st.selectbox("Select Period", ['5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'])
        with col2:
            chart_type = st.selectbox("Select Chart Type", ['Line Chart', 'Candlestick Chart', 'Volume-Overlaid Chart'])
        hist = stock.history(period=period)

        # Plot Chart
        fig = go.Figure()

        if chart_type == 'Line Chart':
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close'))
            fig.update_layout(title=f"{ticker.upper()} Stock Price (Line Chart)", xaxis_title="Date", yaxis_title="Price")

        elif chart_type == 'Candlestick Chart':
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name='Candlestick'
            ))
            fig.update_layout(title=f"{ticker.upper()} Stock Price (Candlestick Chart)", xaxis_title="Date", yaxis_title="Price")

        elif chart_type == 'Volume-Overlaid Chart':
            fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'], mode='lines', name='Close'))
            fig.add_trace(go.Bar(x=hist.index, y=hist['Volume'], name='Volume', opacity=0.4, yaxis='y2'))
            fig.update_layout(
                title=f"{ticker.upper()} Stock Price with Volume (Volume-Overlaid Chart)",
                xaxis_title="Date",
                yaxis_title="Price",
                yaxis2=dict(
                    title="Volume",
                    overlaying="y",
                    side="right"
                )
            )

        # Apply the increased size to all chart types
        fig.update_layout(width=1200, height=600)  # Set global width and height
        st.plotly_chart(fig)


        ### ML Part
        # Add prediction section
        st.subheader("Price Prediction")
        st.write("(Based on Polynomial Regression Model)")

        # Prepare data for prediction
        two_years_data = stock.history(period='2y') # use two years of historical data for prediction
        two_years_data = two_years_data[['Close', 'Volume']]
        two_years_data['50_MA'] = two_years_data['Close'].rolling(window=50).mean()
        two_years_data['200_MA'] = two_years_data['Close'].rolling(window=200).mean()
        two_years_data.dropna(inplace=True)  # Remove rows with NaN values due to moving averages

        # Features and target
        X = two_years_data[['Volume', '50_MA', '200_MA']].values
        y = two_years_data['Close'].values

        # Train-test split
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        test_dates = two_years_data.index[train_size:]

        # Polynomial regression with degree 2
        poly_degree = 2
        model = Pipeline([
            ('poly_features', PolynomialFeatures(degree=poly_degree)),
            ('linear_regression', LinearRegression())
        ])

        # Train the model
        model.fit(X_train, y_train)

        # Predict for the test set
        y_pred = model.predict(X_test)

        # Add date input for prediction
        # min_date = two_years_data.index.min().date()  # Earliest available data
        max_date = two_years_data.index.max().date() + timedelta(days=365)  # Allow prediction up to 1 year in the future
        default_date = two_years_data.index.max().date() + timedelta(days=30)  # Default to 30 days in the future
        user_date = st.date_input(
            "Select date for prediction",
            value=default_date,
            min_value=(two_years_data.index.max().date() + timedelta(days=1)),
            max_value=max_date
        )

        # Predict for future date
        future_days = (user_date - two_years_data.index.max().date()).days
        future_prices = []

        # Calculate Mean Squared Error and R^2 Score for the test set
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if future_days > 0:
            # Generate realistic future features for predictions
            for i in range(1, future_days + 1):
                future_date = two_years_data.index.max() + timedelta(days=i)
                last_volume = two_years_data['Volume'].iloc[-1] * (1 + np.random.normal(0, 0.01))  # Add slight randomness
                last_50_ma = two_years_data['50_MA'].iloc[-1] * (1 + np.random.normal(0, 0.005))
                last_200_ma = two_years_data['200_MA'].iloc[-1] * (1 + np.random.normal(0, 0.002))

                future_features = np.array([last_volume, last_50_ma, last_200_ma]).reshape(1, -1)
                future_features_poly = model.named_steps['poly_features'].transform(future_features)
                predicted_price = model.named_steps['linear_regression'].predict(future_features_poly)[0]
                future_prices.append((future_date, predicted_price))

            # Get prediction for the user-selected date
            selected_future_price = future_prices[-1][1]
            st.write(f"Predicted Price on {user_date}: **${selected_future_price:.2f}**")

            # Display the accuracy metrics
            st.write(f"**Test Set Mean Squared Error (MSE):** {mse:.2f}")
            st.write(f"**Test Set R² Score:** {r2:.2f}")
            st.write("**🚨 Disclaimer 🚨: \
            This is not a financial advice. \
            Investing involves risk.**\
            ")
        else:
            st.warning("Selected date is within historical range. Use actual closing prices for this range.")

        # Plot actual and predicted prices
        plt.figure(figsize=(12, 6))

        # Actual test prices
        plt.plot(test_dates, y_test, label="Actual Closing Prices", color='blue')

        # Predicted test prices
        plt.plot(test_dates, y_pred, label="Predicted Closing Prices (Test Data)", color='red', linestyle='--')

        # Future prediction
        if future_days > 0:
            future_dates, future_values = zip(*future_prices)
            plt.plot(future_dates, future_values, label="Predicted Future Prices", color='green', linestyle='--')
            plt.axvline(x=two_years_data.index.max(), color='gray', linestyle='--', label="Prediction Start Date")
            plt.scatter([user_date], [selected_future_price], color='purple', label="User-Selected Date Price", zorder=5)

        plt.xlabel("Date")
        plt.ylabel("Stock Price")
        plt.title("Actual vs Predicted Closing Prices")
        plt.legend()
        st.pyplot(plt)


    except Exception as e:
        st.error(f"Error fetching data for {ticker.upper()}. Please check the ticker.")
        st.stop()
