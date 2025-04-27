# Sentiment-Fused-GAN-Stock-Forecasting

This project develops a Sentiment-Enhanced GAN for multi-horizon stock market forecasting. By combining financial sentiment analysis with technical market data, the model aims to predict future stock prices with improved accuracy and interpretability.

## Project Overview

The core idea behind this project is to fuse sentiment data derived from financial news and earnings calls with technical indicators (like stock price, volume, RSI, and MACD) in order to make more accurate multi-horizon predictions for stock prices.

## Key Components:

1. **Sentiment Analysis with FinBERT**:

   - Extract nuanced sentiment from financial news articles using [FinBERT](https://arxiv.org/abs/2006.07538).
   - This helps us to understand the market mood, which can influence stock price movements.

2. **WGAN-GP for Time Series**:

   - The model uses a **Wasserstein GAN with Gradient Penalty** (WGAN-GP) for stable training of GANs on time-series data.
   - The GAN will generate synthetic data to augment the training dataset for the model.

3. **Cross-Attention Fusion**:

   - Align sentiment data (from news articles) and technical indicators using a **Cross-Attention Module**.
   - This approach allows for more effective fusion compared to traditional concatenation.

4. **Temporal Fusion Transformer (TFT)**:
   - The TFT model is employed for **multi-horizon forecasting** (1-day, 7-day, 30-day stock price predictions).

## Data Sources:

1. **Stock Market Data**:

   - Daily stock data (including open, high, low, close, volume, etc.) for the top 5 Indian companies is fetched using the [Alpha Vantage API](https://www.alphavantage.co/).
   - The data is filtered for the period from January 2020 to December 2024.

2. **News Data**:
   - News headlines related to the top 5 companies are fetched from [Google News RSS feeds](https://news.google.com/news/rss).
   - Data includes headlines for each month between 2020 and 2024, which are used for sentiment analysis.

## Data Collection:

- **Stock Data**: The stock data for the top 5 companies (Reliance Industries, TCS, HDFC Bank, Infosys, and ICICI Bank) is fetched using Alpha Vantage API.
- **News Data**: News headlines are fetched month by month using Google News RSS for the same set of companies.

## References:

Lin, H., Chen, C., Huang, G. & Jafari, A. (2021). Stock price prediction using Generative Adversarial Networks. Journal of Computer Science, 17(3), 188-196. https://doi.org/10.3844/jcssp.2021.188.196
