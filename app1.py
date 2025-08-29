import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from datetime import timedelta
import datetime
from prophet import Prophet
import plotly.graph_objects as go
import requests

FMP_API_KEY = "RyH9jkHljdvsd38NCNwa3b0PwW2RMaBp"

def get_roe_roce_fmp(fmp_ticker, api_key):
    url = f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{fmp_ticker}?apikey={api_key}"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()

        if isinstance(data, list) and data:
            roe = data[0].get("returnOnEquityTTM")
            roce = data[0].get("returnOnCapitalEmployedTTM")
            return roe, roce
        else:
            return None, None
    except Exception as e:
        print(f"FMP API Error for {fmp_ticker}: {e}")
        return None, None




# --- Page Setup ---
st.set_page_config(page_title="Stock Predictor App", layout="wide")
st.title("📈 Stock Predictor App")

# --- Tabs ---

tab1, tab2, tab3 = st.tabs(["📈 Stock Prediction", "🔍 Search for a Company", "⭐ My Watchlist"])


# --- NIFTY 50 Dictionary (Unified) ---
nifty_50 = {
    "Adani Enterprises": "ADANIENT.NS", "Adani Ports": "ADANIPORTS.NS", "Apollo Hospitals": "APOLLOHOSP.NS",
    "Asian Paints": "ASIANPAINT.NS", "Axis Bank": "AXISBANK.NS", "Bajaj Auto": "BAJAJ-AUTO.NS",
    "Bajaj Finance": "BAJFINANCE.NS", "Bajaj Finserv": "BAJAJFINSV.NS", "Bharti Airtel": "BHARTIARTL.NS",
    "BPCL": "BPCL.NS", "Britannia": "BRITANNIA.NS", "Cipla": "CIPLA.NS", "Coal India": "COALINDIA.NS",
    "Divi's Labs": "DIVISLAB.NS", "Dr. Reddy's": "DRREDDY.NS", "Eicher Motors": "EICHERMOT.NS",
    "Grasim Industries": "GRASIM.NS", "HCL Tech": "HCLTECH.NS", "HDFC Bank": "HDFCBANK.NS",
    "Hero MotoCorp": "HEROMOTOCO.NS", "Hindalco": "HINDALCO.NS", "Hindustan Unilever": "HINDUNILVR.NS",
    "ICICI Bank": "ICICIBANK.NS", "ITC": "ITC.NS", "Infosys": "INFY.NS", "JSW Steel": "JSWSTEEL.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS", "LTIMindtree": "LTIM.NS", "Larsen & Toubro": "LT.NS",
    "M&M": "M&M.NS", "Maruti Suzuki": "MARUTI.NS", "NTPC": "NTPC.NS", "Nestle India": "NESTLEIND.NS",
    "ONGC": "ONGC.NS", "Power Grid": "POWERGRID.NS", "Reliance Industries": "RELIANCE.NS",
    "SBI Life Insurance": "SBILIFE.NS", "SBI": "SBIN.NS", "Sun Pharma": "SUNPHARMA.NS",
    "Tata Consumer": "TATACONSUM.NS", "Tata Motors": "TATAMOTORS.NS", "Tata Steel": "TATASTEEL.NS",
    "TCS": "TCS.NS", "Tech Mahindra": "TECHM.NS", "Titan Company": "TITAN.NS",
    "UltraTech Cement": "ULTRACEMCO.NS", "UPL": "UPL.NS", "Wipro": "WIPRO.NS"
}



# --- Tab 1: Stock Prediction ---
with tab1:
    st.header("📈 Stock Price Prediction using Prophet")

    company_name_tab1 = st.selectbox("Select a company for prediction", sorted(nifty_50.keys()), key="selectbox_tab1")
    selected_ticker = nifty_50[company_name_tab1]

    start_date = st.date_input("Start Date", datetime.date(2020, 1, 1), key="prophet_start")
    end_date = st.date_input("End Date", datetime.date.today(), key="prophet_end")

    if start_date >= end_date:
        st.error("⚠️ End Date must be after Start Date")
    else:
        df = yf.download(selected_ticker, start=start_date, end=end_date)

        if df.empty:
            st.warning("No data found for the selected range.")
        else:
            df_prophet = df[["Close"]].copy()
            df_prophet["ds"] = pd.to_datetime(df.index).tz_localize(None)  # Remove timezone
            df_prophet["y"] = df_prophet["Close"]
            df_prophet = df_prophet[["ds", "y"]].dropna()

            forecast_days = st.slider("How many days to predict?", 7, 90, 30)

            model = Prophet(daily_seasonality=True)
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

            # Create dataframe with predictions and confidence intervals
            forecast_display = forecast[forecast["ds"] > df_prophet["ds"].max()][["ds", "yhat", "yhat_lower", "yhat_upper"]]
            forecast_display.columns = ["Date", "Predicted Close", "Lower Bound", "Upper Bound"]
            forecast_display[["Predicted Close", "Lower Bound", "Upper Bound"]] = forecast_display[["Predicted Close", "Lower Bound", "Upper Bound"]].round(2)

            st.subheader("📋 Predicted Stock Prices with Confidence Intervals")
            st.dataframe(forecast_display)

            csv = forecast_display.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions as CSV", data=csv, file_name="prophet_predictions.csv", mime="text/csv")

            st.subheader("📊 Forecast Visualization with Confidence Intervals")
            fig = go.Figure()

            # Historical prices
            fig.add_trace(go.Scatter(x=df_prophet["ds"], y=df_prophet["y"],
                                     mode='lines', name='Historical', line=dict(color='royalblue')))

            # Predicted prices
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"],
                                     mode='lines', name='Predicted', line=dict(color='orange', dash='dash')))

            # Confidence interval (shaded area)
            fig.add_trace(go.Scatter(
                x=forecast["ds"].tolist() + forecast["ds"][::-1].tolist(),
                y=forecast["yhat_upper"].tolist() + forecast["yhat_lower"][::-1].tolist(),
                fill='toself',
                fillcolor='rgba(255, 165, 0, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo="skip",
                name='Confidence Interval'
            ))

            fig.update_layout(
                title=f"{company_name_tab1} Stock Price Forecast ({forecast_days} days ahead)",
                xaxis_title="Date",
                yaxis_title="Price (INR)",
                legend=dict(x=0, y=1.1, orientation="h"),
                template="plotly_white"
            )

            st.plotly_chart(fig, use_container_width=True)

fmp_nifty_50 = {
    "Adani Enterprises": "ADANIENT.BSE", "Adani Ports": "ADANIPORTS.BSE",
    "Apollo Hospitals": "APOLLOHOSP.BSE", "Asian Paints": "ASIANPAINT.BSE",
    "Axis Bank": "AXISBANK.BSE", "Bajaj Auto": "BAJAJ-AUTO.BSE",
    "Bajaj Finance": "BAJFINANCE.BSE", "Bajaj Finserv": "BAJAJFINSV.BSE",
    "Bharti Airtel": "BHARTIARTL.BSE", "BPCL": "BPCL.BSE",
    "Britannia": "BRITANNIA.BSE", "Cipla": "CIPLA.BSE",
    "Coal India": "COALINDIA.BSE", "Divi's Labs": "DIVISLAB.BSE",
    "Dr. Reddy's": "DRREDDY.BSE", "Eicher Motors": "EICHERMOT.BSE",
    "Grasim Industries": "GRASIM.BSE", "HCL Tech": "HCLTECH.BSE",
    "HDFC Bank": "HDFCBANK.BSE", "Hero MotoCorp": "HEROMOTOCO.BSE",
    "Hindalco": "HINDALCO.BSE", "Hindustan Unilever": "HINDUNILVR.BSE",
    "ICICI Bank": "ICICIBANK.BSE", "ITC": "ITC.BSE",
    "Infosys": "INFY.BSE", "JSW Steel": "JSWSTEEL.BSE",
    "Kotak Mahindra Bank": "KOTAKBANK.BSE", "LTIMindtree": "LTIM.BSE",
    "Larsen & Toubro": "LT.BSE", "M&M": "M&M.BSE",
    "Maruti Suzuki": "MARUTI.BSE", "NTPC": "NTPC.BSE",
    "Nestle India": "NESTLEIND.BSE", "ONGC": "ONGC.BSE",
    "Power Grid": "POWERGRID.BSE", "Reliance Industries": "RELIANCE.BSE",
    "SBI Life Insurance": "SBILIFE.BSE", "SBI": "SBIN.BSE",
    "Sun Pharma": "SUNPHARMA.BSE", "Tata Consumer": "TATACONSUM.BSE",
    "Tata Motors": "TATAMOTORS.BSE", "Tata Steel": "TATASTEEL.BSE",
    "TCS": "TCS.BSE", "Tech Mahindra": "TECHM.BSE",
    "Titan Company": "TITAN.BSE", "UltraTech Cement": "ULTRACEMCO.BSE",
    "UPL": "UPL.BSE", "Wipro": "WIPRO.BSE"
}

# --- Tab 2: Company Search ---
with tab2:
    st.header("🔍 Search for a Company")

    search_query = st.text_input("Search by company name", "", key="search_input_tab2")
    filtered_companies = [name for name in nifty_50 if search_query.lower() in name.lower()]

    if filtered_companies:
        company_name_tab2 = st.selectbox("Select Company", filtered_companies, key="selectbox_tab2")
        ticker_symbol = nifty_50[company_name_tab2]
        stock = yf.Ticker(ticker_symbol)

        info = stock.info
        fmp_ticker = fmp_nifty_50.get(company_name_tab2)
        roe, roce = get_roe_roce_fmp(fmp_ticker,FMP_API_KEY) if fmp_ticker else (None, None)


        st.subheader(f"📊 Key Metrics for {company_name_tab2}")
        col1, col2, col3 = st.columns(3)

        if "watchlist" not in st.session_state:
            st.session_state["watchlist"] = []

        if st.button("⭐ Add to Watchlist"):
            if company_name_tab2 not in st.session_state["watchlist"]:
                st.session_state["watchlist"].append(company_name_tab2)
                st.success(f"{company_name_tab2} added to watchlist!")
            else:
                st.info(f"{company_name_tab2} is already in your watchlist.")

        col1.metric("📌 Price", f"${info.get('currentPrice', 'N/A')}")
        col1.metric("🏢 Market Cap", f"{info.get('marketCap', 0)/1e9:.2f} B")
        col1.metric("💵 EPS", info.get('trailingEps', 'N/A'))

        col2.metric("📉 PE Ratio", info.get('trailingPE', 'N/A'))
        col2.metric("📘 P/B Ratio", info.get('priceToBook', 'N/A'))
        col2.metric("📈 PEG Ratio", info.get('pegRatio', 'N/A'))

        col3.metric("🏦 ROE", f"{roe*100:.2f}%" if roe else "N/A")
        col3.metric("🏭 ROCE", f"{roce*100:.2f}%" if roce else "N/A")
        div_yield = info.get('dividendYield', None)
        div_yield_display = f"{div_yield:.2f}%" if div_yield is not None else "N/A"
        col3.metric("💰 Dividend Yield", div_yield_display)

        # Smart Insights
        st.subheader("🧠 Smart Insights")
        insights = []

        pe = info.get("trailingPE", None)
        if pe:
            if pe < 15:
                insights.append("📉 PE ratio is low – may be undervalued.")
            elif pe > 30:
                insights.append("📈 PE ratio is high – may be overvalued.")
            else:
                insights.append("✅ PE ratio is in a healthy range.")

        if roe:
            insights.append("🚀 High ROE – strong returns on equity." if roe > 0.15 else "⚠️ ROE is below ideal levels.")
        if roce:
            insights.append("💼 Healthy ROCE – efficient capital usage." if roce > 0.15 else "📉 Low ROCE – may be inefficient.")

        de_ratio = info.get("debtToEquity", None)
        if de_ratio is not None:
            if de_ratio < 0.5:
                insights.append("✅ Low debt – financially stable.")
            elif de_ratio > 2:
                insights.append("⚠️ High debt levels – potential risk.")

        profit_margin = info.get("profitMargins", None)
        if profit_margin:
            if profit_margin > 0.15:
                insights.append("💰 Strong profit margins.")
            elif profit_margin < 0.05:
                insights.append("⚠️ Thin margins – low profitability.")

        st.markdown("### 📝 Summary")
        for insight in insights:
            st.write(insight)

        st.divider()

        # Valuation Table
        st.subheader("📋 Valuation Summary")
        val_data = {
            "EV/EBITDA": info.get("enterpriseToEbitda", "N/A"),
            "Book Value / Share": info.get("bookValue", "N/A"),
            "Debt/Equity": info.get("debtToEquity", "N/A"),
            "Profit Margin": f"{info.get('profitMargins', 0) * 100:.2f}%" if info.get("profitMargins") else "N/A",
        }
        val_df = pd.DataFrame(val_data.items(), columns=["Metric", "Value"])
        st.table(val_df)

        # Price Chart
        st.subheader("📈 Historical Price Chart (6 Months)")
        end = datetime.date.today()
        start = end - datetime.timedelta(days=180)
        hist = stock.history(start=start, end=end)

        if not hist.empty:
            hist["MA20"] = hist["Close"].rolling(window=20).mean()
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(hist.index, hist["Close"], label="Close Price", color="blue")
            ax.plot(hist.index, hist["MA20"], label="20-Day MA", color="orange", linestyle="--")
            ax.set_title(f"{company_name_tab2} - Close Price with Moving Average")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (INR)")
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("No historical data found.")
    else:
        if search_query:
            st.warning("No matching companies found.")



# --- Tab 3: Watchlist ---
with tab3:
    st.header("⭐ My Watchlist")

    if "watchlist" not in st.session_state or len(st.session_state["watchlist"]) == 0:
        st.info("Your watchlist is empty. Add companies from the 'Search for a Company' tab.")
    else:
        for company in st.session_state["watchlist"]:
            ticker = nifty_50[company]
            data = yf.Ticker(ticker)
            info = data.info

            current_price = info.get("currentPrice", "N/A")
            pe = info.get("trailingPE", "N/A")
            market_cap = f"{info.get('marketCap', 0) / 1e9:.2f} B"

            with st.container():
                col1, col2, col3, col4 = st.columns(4)
                col1.markdown(f"### {company}")
                col2.metric("Price", f"{current_price}")
                col3.metric("PE Ratio", pe)
                col4.metric("Market Cap", market_cap)

        # Optional: Clear watchlist button
        if st.button("🗑️ Clear Watchlist"):
            st.session_state["watchlist"] = []
            st.success("Watchlist cleared.")


