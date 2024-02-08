import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import yfinance as yf

st.title('S&P 500 App')

st.markdown("""
This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
* **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, yfinance
* **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
""")

st.sidebar.header('User Input Features')

# Web scraping of S&P 500 data
#
@st.cache
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    # As we get the information using pandas from wikipedia, it gets only the tables
    html = pd.read_html(url, header = 0)
    # html[0] correspond to the first table in the wikipedia page
    df = html[0]
    return df

# Unpack the data into the variable
df = load_data()
# Group by the column "GICS Sector" from the data frame
sector = df.groupby('GICS Sector')

# Sidebar - Sector selection
sorted_sector_unique = sorted( df['GICS Sector'].unique() )
# Allow to select the sectors to display in the st.dataframe
selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

# Filtering data that belongs to the selected_sector variable
df_selected_sector = df[ (df['GICS Sector'].isin(selected_sector)) ]

st.header('Display Companies in Selected Sector')
st.write('Data Dimension: ' + str(df_selected_sector.shape[0]) + ' rows and ' + str(df_selected_sector.shape[1]) + ' columns.')
# Display the df_selected sector from line 41
st.dataframe(df_selected_sector)

# Download S&P500 data
# https://discuss.streamlit.io/t/how-to-download-file-in-streamlit/1806
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_sector), unsafe_allow_html=True)

# With this function we retreive the stock prices. Obtained from
# https://pypi.org/project/yfinance/

data = yf.download(
        # tickers list of strings as well. Select only 10 to make it faster
        tickers = list(df_selected_sector[:10].Symbol),
        # Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        # (optional, default is "1mo")
        period = "ytd",
        # Fetch data by intrval (including intraday if period < 60 days)
        # valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        # (optionsl, default is "1d")
        interval = "1d",
        # Group by ticker (to access via data["SPY"])
        # (Optional, default is "column")
        group_by = 'ticker',
        # Adjust all OHLC automatically
        # (optional, default is False)
        auto_adjust = True,
        # download pre/post regula market hours data
        # (optional, default is True)
        prepost = True,
        # proxy URL scheme use when downloading?
        # (optional, default is None)
        threads = True,
        proxy = None
    )

# Plot Closing Price of Query Symbol
def price_plot(symbol):
  df = pd.DataFrame(data[symbol].Close)
  df['Date'] = df.index
  plt.fill_between(df.Date, df.Close, color='skyblue', alpha=0.3)
  plt.plot(df.Date, df.Close, color='skyblue', alpha=0.8)
  plt.xticks(rotation=90)
  plt.title(symbol, fontweight='bold')
  plt.xlabel('Date', fontweight='bold')
  plt.ylabel('Closing Price', fontweight='bold')
  return st.pyplot()

# Slider to select the number of companies to display
num_company = st.sidebar.slider('Number of Companies', 1, 10)

if st.button('Show Plots'):
    st.header('Stock Closing Price')
    for i in list(df_selected_sector.Symbol)[:num_company]:
        price_plot(i)
