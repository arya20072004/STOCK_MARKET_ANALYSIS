"""
Stock Market Analysis & Prediction App

- Fetches historical stock data using yfinance
- Visualizes price trends and moving averages
- Predicts future prices using Linear Regression and LSTM
- Built with Streamlit for interactive web UI

Author: Arya Borhade
"""

# Check for required packages and provide helpful error if missing
try:
    import streamlit as st
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.losses import MeanSquaredError
    import os
except ImportError as e:
    missing = str(e).split("'")[1]
    print(f"Missing required package: {missing}. Please install all requirements with 'pip install -r requirements.txt'")
    raise

st.title("📈 Stock Market Analysis & Prediction")

# User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS):", value="AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
model_option = st.selectbox("Select Prediction Model", ["Both", "Linear Regression", "LSTM"])


def fetch_data(ticker, start, end):
    """Fetch historical stock data from Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end)
    data = data.ffill()
    return data


def create_sequences(data, seq_length=60):
    """Create sequences of data for LSTM input."""
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


if st.button("Fetch Data & Predict"):
    if ticker:
        data = fetch_data(ticker, start_date, end_date)
        if not data.empty:
            st.success(f"✅ Data fetched for {ticker}")
            st.write(data.tail())

            # Moving Averages
            ma_windows = [5, 13, 20, 50, 75, 100, 200, 300, 365]
            for ma in ma_windows:
                data[f'MA{ma}'] = data['Close'].rolling(window=ma).mean()

            # Plot Close & MA
            st.subheader("📊 Closing Price & Moving Averages")
            fig, ax = plt.subplots()
            ax.plot(data.index, data['Close'], label='Close')
            for ma in ma_windows:
                ax.plot(data.index, data[f'MA{ma}'], label=f'MA{ma}')
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

            # Volume Plot
            st.subheader("📦 Volume Traded")
            x = mdates.date2num(data.index.to_pydatetime())
            fig2, ax2 = plt.subplots()
            ax2.vlines(x, [0], data['Volume'].values, color='gray', linewidth=1)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Volume")
            ax2.xaxis_date()
            fig2.autofmt_xdate()
            st.pyplot(fig2)

            # Prepare data
            close_prices = data['Close'].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_close = scaler.fit_transform(close_prices)

            # Linear Regression
            if model_option in ["Linear Regression", "Both"]:
                st.subheader("🔮 Linear Regression Prediction")
                df = data.reset_index()
                df['Date_ordinal'] = pd.to_datetime(df['Date']).map(pd.Timestamp.toordinal)
                df = df[['Date_ordinal', 'Close']].dropna()

                X = df[['Date_ordinal']]
                y = df['Close']

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                model_lr = LinearRegression()
                model_lr.fit(X_train, y_train)
                y_pred = model_lr.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.write(f"📉 MSE: {mse:.2f}")
                st.write(f"🔍 R² Score: {r2:.2f}")

                next_day = pd.to_datetime(end_date + pd.Timedelta(days=1)).toordinal()
                next_price = model_lr.predict([[next_day]])
                st.metric("📈 Next Closing Price (LR)", f"₹{float(next_price[0]):.2f}")

                fig3, ax3 = plt.subplots()
                ax3.plot(X_test.index, y_test, label="Actual")
                ax3.plot(X_test.index, y_pred, label="Predicted")
                ax3.set_xlabel("Days")
                ax3.set_ylabel("Price")
                ax3.legend()
                st.pyplot(fig3)

            # LSTM
            if model_option in ["LSTM", "Both"]:
                st.subheader("🔮 LSTM Prediction")
                seq_len = 60

                if len(scaled_close) > seq_len:
                    X_lstm, y_lstm = create_sequences(scaled_close, seq_len)
                    X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

                    split = int(0.8 * len(X_lstm))
                    X_train_lstm, X_test_lstm = X_lstm[:split], X_lstm[split:]
                    y_train_lstm, y_test_lstm = y_lstm[:split], y_lstm[split:]

                    model_path = f"{ticker}_lstm_model.h5"

                    if os.path.exists(model_path):
                        try:
                            model_lstm = load_model(model_path)
                            st.info("📥 LSTM model loaded from file.")
                        except:
                            os.remove(model_path)
                            model_lstm = None
                    else:
                        model_lstm = None

                    if model_lstm is None:
                        model_lstm = Sequential()
                        model_lstm.add(LSTM(50, return_sequences=True, input_shape=(seq_len, 1)))
                        model_lstm.add(LSTM(50))
                        model_lstm.add(Dense(1))
                        model_lstm.compile(optimizer='adam', loss=MeanSquaredError())
                        history = model_lstm.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=16, verbose=0)
                        model_lstm.save(model_path)
                        st.success("✅ LSTM model trained and saved.")

                        # Plot training loss
                        fig_loss, ax_loss = plt.subplots()
                        ax_loss.plot(history.history['loss'])
                        ax_loss.set_title("📉 LSTM Training Loss")
                        ax_loss.set_xlabel("Epoch")
                        ax_loss.set_ylabel("Loss")
                        st.pyplot(fig_loss)

                    y_pred_lstm = model_lstm.predict(X_test_lstm)
                    y_pred_inv = scaler.inverse_transform(y_pred_lstm)
                    y_test_inv = scaler.inverse_transform(y_test_lstm.reshape(-1, 1))

                    mse_lstm = mean_squared_error(y_test_inv, y_pred_inv)
                    r2_lstm = r2_score(y_test_inv, y_pred_inv)
                    st.write(f"📉 MSE: {mse_lstm:.2f}")
                    st.write(f"🔍 R² Score: {r2_lstm:.2f}")

                    # Predict next day
                    last_seq = scaled_close[-seq_len:]
                    last_seq = last_seq.reshape((1, seq_len, 1))
                    next_price_scaled = model_lstm.predict(last_seq)
                    next_price_lstm = scaler.inverse_transform(next_price_scaled)[0, 0]
                    st.metric("📈 Next Closing Price (LSTM)", f"₹{float(next_price_lstm):.2f}")

                    fig_lstm, ax_lstm = plt.subplots()
                    ax_lstm.plot(y_test_inv, label="Actual")
                    ax_lstm.plot(y_pred_inv, label="Predicted")
                    ax_lstm.set_title("📊 LSTM Predicted vs Actual")
                    ax_lstm.legend()
                    st.pyplot(fig_lstm)
                else:
                    st.warning("⚠️ Not enough data for LSTM (need > 60 data points).")
        else:
            st.error("❌ No data found.")
    else:
        st.warning("⚠️ Please enter a valid stock ticker.")
