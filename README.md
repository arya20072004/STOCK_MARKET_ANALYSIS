# STOCK_MARKET_ANALYSIS

A Streamlit web app for visualizing stock data and predicting future prices using Linear Regression and LSTM.

## Features

- Fetch historical stock data (Yahoo Finance)
- Visualize price trends and moving averages
- Predict next-day closing price (Linear Regression & LSTM)
- Interactive charts and metrics

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/stock_market_analysis.git
   cd stock_market_analysis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Usage

- Enter a stock ticker (e.g., `AAPL`, `RELIANCE.NS`)
- Select date range and prediction model
- Click "Fetch Data & Predict" to view analysis and predictions

## Notes

- LSTM model requires at least 60 days of data.
- All models and plots run locally in your browser.

---

**Author:** Arya Borhade
