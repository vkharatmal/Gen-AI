import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import ast
import glob
from google.colab import drive

GROK_API_KEY = 'API_KEY'
GROK_API_ENDPOINT = "END_POINT"  # Hypothetical endpoint

class FinancialAnalystAgent:
    def __init__(self, ticker, horizon_months=12):
        """
        Initialize the financial analyst agent.

        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            horizon_months (int): Investment time horizon in months
        """
        self.ticker = ticker.upper()
        self.horizon_months = horizon_months
        self.stock = yf.Ticker(ticker)
        self.data = None
        self.fundamentals = None
        self.technical_indicators = {}
        self.market_insights = {}
        self.industry_insights = {}
        self.company_analysis = {}
        self.recommendation = {}

    def fetch_market_data(self):
        """
        Perform market and economic research using yfinance and Grok API.
        """
        sp500 = yf.Ticker("^GSPC")
        sp500_hist = sp500.history(period="1y")

        # Use .iloc for positional indexing
        market_return = (sp500_hist['Close'].iloc[-1] / sp500_hist['Close'].iloc[0] - 1) * 100
        volatility = sp500_hist['Close'].pct_change().std() * np.sqrt(252) * 100

        # Grok API call for economic insights
        prompt = "Provide a brief economic outlook, including interest rates, inflation trends, and geopolitical risks for the next 12 months."
        grok_payload = {
            "query": prompt,
            "api_key": GROK_API_KEY
        }
        try:
            response = requests.post(GROK_API_ENDPOINT, json=grok_payload)
            response.raise_for_status()
            economic_outlook = response.json().get("analysis", "No economic data from Grok API")
        except Exception as e:
            economic_outlook = f"Stable economy, moderate growth expected. (API error: {str(e)})"

        self.market_insights = {
            "market_return": round(market_return, 2),
            "volatility": round(volatility, 2),
            "economic_outlook": economic_outlook
        }

    def industry_sector_analysis(self):
        """
        Analyze the industry and sector of the company using yfinance and Grok API.
        """
        info = self.stock.info
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")

        sector_tickers = {
            "Technology": "XLK",
            "Healthcare": "XLV",
            "Financials": "XLF",
            "Consumer Discretionary": "XLY",
            "Industrials": "XLI",
        }
        sector_ticker = sector_tickers.get(sector, "SPY")
        sector_etf = yf.Ticker(sector_ticker)
        sector_hist = sector_etf.history(period="1y")

        # Use .iloc for positional indexing
        sector_return = (sector_hist['Close'].iloc[-1] / sector_hist['Close'].iloc[0] - 1) * 100
        sector_pe = sector_etf.info.get("trailingPE", np.nan)

        # Grok API call for industry trends
        prompt = f"Provide a brief analysis of trends, growth drivers, and risks in the {industry} industry for the next {self.horizon_months} months."
        grok_payload = {
            "query": prompt,
            "api_key": GROK_API_KEY
        }
        try:
            response = requests.post(GROK_API_ENDPOINT, json=grok_payload)
            response.raise_for_status()
            industry_trends = response.json().get("analysis", "No industry data from Grok API")
        except Exception as e:
            industry_trends = f"{industry} shows steady growth. (API error: {str(e)})"

        self.industry_insights = {
            "sector": sector,
            "industry": industry,
            "sector_return": round(sector_return, 2),
            "sector_pe": round(sector_pe, 2) if not np.isnan(sector_pe) else "N/A",
            "industry_trends": industry_trends
        }

    def company_fundamental_analysis(self):
        """
        Perform fundamental analysis of the company using yfinance and Grok API.
        """
        info = self.stock.info

        pe_ratio = info.get("trailingPE", np.nan)
        pb_ratio = info.get("priceToBook", np.nan)
        eps = info.get("trailingEps", np.nan)
        debt_to_equity = info.get("debtToEquity", np.nan)
        roe = info.get("returnOnEquity", np.nan)

        # Grok API call for qualitative analysis
        prompt = f"Provide a brief evaluation of {self.ticker}'s competitive position, management quality, and growth prospects."
        grok_payload = {
            "query": prompt,
            "api_key": GROK_API_KEY
        }
        try:
            response = requests.post(GROK_API_ENDPOINT, json=grok_payload)
            response.raise_for_status()
            qualitative_analysis = response.json().get("analysis", "No qualitative data from Grok API")
        except Exception as e:
            qualitative_analysis = f"{self.ticker} has strong market position. (API error: {str(e)})"

        self.company_analysis = {
            "pe_ratio": round(pe_ratio, 2) if not np.isnan(pe_ratio) else "N/A",
            "pb_ratio": round(pb_ratio, 2) if not np.isnan(pb_ratio) else "N/A",
            "eps": round(eps, 2) if not np.isnan(eps) else "N/A",
            "debt_to_equity": round(debt_to_equity, 2) if not np.isnan(debt_to_equity) else "N/A",
            "roe": round(roe, 2) if not np.isnan(roe) else "N/A",
            "qualitative_analysis": qualitative_analysis
        }

    def technical_analysis(self):
        """
        Perform technical analysis using price data from yfinance.
        """
        self.data = self.stock.history(period="1y")

        self.data['SMA50'] = self.data['Close'].rolling(window=50).mean()
        self.data['SMA200'] = self.data['Close'].rolling(window=200).mean()

        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

        # Use .iloc for positional indexing
        latest_price = self.data['Close'].iloc[-1]
        sma50 = self.data['SMA50'].iloc[-1]
        sma200 = self.data['SMA200'].iloc[-1]
        rsi = self.data['RSI'].iloc[-1]

        trend = "Bullish" if latest_price > sma50 > sma200 else "Bearish" if latest_price < sma50 < sma200 else "Neutral"
        overbought = rsi > 70
        oversold = rsi < 30

        self.technical_indicators = {
            "latest_price": round(latest_price, 2),
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2),
            "rsi": round(rsi, 2),
            "trend": trend,
            "overbought": overbought,
            "oversold": oversold
        }

    def estimate_target_price(self):
        """
        Estimate target price based on fundamentals and technicals.
        """
        current_price = self.technical_indicators["latest_price"]
        pe_ratio = self.company_analysis["pe_ratio"]
        eps = self.company_analysis["eps"]

        growth_rate = 0.05
        discount_rate = 0.1
        if isinstance(pe_ratio, (int, float)) and isinstance(eps, (int, float)):
            forward_eps = eps * (1 + growth_rate)
            target_price = forward_eps * pe_ratio * (1 / (1 + discount_rate))
        else:
            target_price = current_price * 1.1

        if self.technical_indicators["trend"] == "Bullish":
            target_price *= 1.05
        elif self.technical_indicators["trend"] == "Bearish":
            target_price *= 0.95

        return round(target_price, 2)

    # def identify_risks_catalysts(self):
    #     """
    #     Identify risks and catalysts using Grok API.
    #     """
    #     prompt = f"Identify potential risks (e.g., regulatory, competitive) and catalysts (e.g., new product launches, earnings) for {self.ticker} over the next {self.horizon_months} months."
    #     grok_payload = {
    #         "query": prompt,
    #         "api_key": GROK_API_KEY
    #     }
    #     try:
    #         response = requests.post(GROK_API_ENDPOINT, json=grok_payload)
    #         response.raise_for_status()
    #         data = response.json()
    #         risks = data.get("risks", ["No risks identified by Grok API"])
    #         catalysts = data.get("catalysts", ["No catalysts identified by Grok API"])
    #     except Exception as e:
    #         risks = [f"Market competition (API error: {str(e)})"]
    #         catalysts = [f"Product launches (API error: {str(e)})"]

    #     return risks, catalysts

    def synthesize_recommendation(self):
        """
        Combine all analyses into an actionable recommendation.
        """
        self.fetch_market_data()
        self.industry_sector_analysis()
        self.company_fundamental_analysis()
        self.technical_analysis()

        target_price = self.estimate_target_price()
        # risks, catalysts = self.identify_risks_catalysts()

        score = 0
        sector_pe = self.industry_insights["sector_pe"]
        if isinstance(self.company_analysis["pe_ratio"], (int, float)) and isinstance(sector_pe, (int, float)):
            if self.company_analysis["pe_ratio"] < sector_pe:
                score += 1
        if self.technical_indicators["trend"] == "Bullish":
            score += 1
        if self.industry_insights["sector_return"] > self.market_insights["market_return"]:
            score += 1

        recommendation = "Buy" if score >= 2 else "Hold" if score == 1 else "Sell"

        self.recommendation = {
            "recommendation": recommendation,
            "target_price": target_price,
            "current_price": self.technical_indicators["latest_price"],
            "time_horizon": f"{self.horizon_months} months",
            #"risks": risks,
            #"catalysts": catalysts
        }

    def display_report(self):
        """
        Display a concise analysis report.
        """
        print(f"\n=== Financial Analysis Summary for {self.ticker} ===")
        print(f"Recommendation: {self.recommendation['recommendation']}")
        print(f"Current Price: {self.recommendation['current_price']}")
        print(f"Target Price: {self.recommendation['target_price']}")
        print(f"Time Horizon: {self.recommendation['time_horizon']}")
        print(f"Trend: {self.technical_indicators['trend']}")
        #print(f"Risks: {', '.join(self.recommendation['risks'])}")
        #print(f"Catalysts: {', '.join(self.recommendation['catalysts'])}")



# Example usage
if __name__ == "__main__":

    drive.mount('/content/drive')
    directory_path=glob.glob('drive/MyDrive/outputfiles/*.txt')

    # Function to read the file and return its content as a list
    def read_file(file_path):
        with open(file_path, 'r') as file:
            return ast.literal_eval(file.read().strip())

    # Load the contents of both files into separate lists
    list1 = read_file('drive/MyDrive/outputfiles/20%file.txt')
    list2 = read_file('drive/MyDrive/outputfiles/IVfile.txt')

    for tick in list1:
        ticker = tick
        agent = FinancialAnalystAgent(ticker, horizon_months=3)
        agent.synthesize_recommendation()
        agent.display_report()
    for tick in list2:
        ticker = tick
        agent = FinancialAnalystAgent(ticker, horizon_months=3)
        agent.synthesize_recommendation()
        agent.display_report()
