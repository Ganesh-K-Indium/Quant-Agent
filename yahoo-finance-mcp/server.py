import json
import pandas as pd
from enum import Enum
import yfinance as yf
from textblob import TextBlob
from datetime import datetime
from fastmcp import FastMCP



# Define an enum for the type of financial statement
class FinancialType(str, Enum):
    income_stmt = "income_stmt"
    quarterly_income_stmt = "quarterly_income_stmt"
    balance_sheet = "balance_sheet"
    quarterly_balance_sheet = "quarterly_balance_sheet"
    cashflow = "cashflow"
    quarterly_cashflow = "quarterly_cashflow"


class HolderType(str, Enum):
    major_holders = "major_holders"
    institutional_holders = "institutional_holders"
    mutualfund_holders = "mutualfund_holders"
    insider_transactions = "insider_transactions"
    insider_purchases = "insider_purchases"
    insider_roster_holders = "insider_roster_holders"


class RecommendationType(str, Enum):
    recommendations = "recommendations"
    upgrades_downgrades = "upgrades_downgrades"

yfinance_server = FastMCP(
    "yfinance",
    instructions="""
# Yahoo Finance MCP Server

This server is used to get information about a given ticker symbol from yahoo finance.

Available tools:
- get_historical_stock_prices: Get historical stock prices for a given ticker symbol from yahoo finance. Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
- get_stock_info: Get stock information for a given ticker symbol from yahoo finance. Include the following information: Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics, Other.
- get_yahoo_finance_news: Get news for a given ticker symbol from yahoo finance.
- get_stock_actions: Get stock dividends and stock splits for a given ticker symbol from yahoo finance.
- get_financial_statement: Get financial statement for a given ticker symbol from yahoo finance. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
- get_holder_info: Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
- get_option_expiration_dates: Fetch the available options expiration dates for a given ticker symbol.
- get_option_chain: Fetch the option chain for a given ticker symbol, expiration date, and option type.
- get_recommendations: Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance. You can also specify the number of months back to get upgrades/downgrades for, default is 12.
- get_target_price: Fetch the 1-year analyst target price for the given ticker symbol.
- get_news_sentiment_and_price_prediction: Get company-specific news headlines from Yahoo Finance, perform sentiment analysis on headlines, and predict stock price movement (UP/DOWN/STABLE) based on average sentiment.
- get_stock_5_year_projection: Analyze stock growth and revenue projection over the last 5 years using Yahoo Finance data.
- get_financial_ratios: Calculate key financial ratios (Liquidity, Solvency, Profitability, Valuation, Efficiency) using data from Yahoo Finance.""",
)

@yfinance_server.tool(
    name="get_historical_stock_prices",
    description="""Get historical stock prices for a given ticker symbol from yahoo finance. Include the following information: Date, Open, High, Low, Close, Volume, Adj Close.
Args:
    ticker: str
        The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
    period : str
        Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        Either Use period parameter or use start and end
        Default is "1mo"
    interval : str
        Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
        Intraday data cannot extend last 60 days
        Default is "1d"
""",
)
async def get_historical_stock_prices(
    ticker: str, period: str = "1mo", interval: str = "1d"
) -> str:
    """Get historical stock prices for a given ticker symbol

    Args:
        ticker: str
            The ticker symbol of the stock to get historical prices for, e.g. "AAPL"
        period : str
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            Either Use period parameter or use start and end
            Default is "1mo"
        interval : str
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
            Default is "1d"
    """
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting historical stock prices for {ticker}: {e}")
        return f"Error: getting historical stock prices for {ticker}: {e}"

    hist_data = company.history(period=period, interval=interval)
    hist_data = hist_data.reset_index(names="Date")
    hist_data = hist_data.to_json(orient="records", date_format="iso")
    return hist_data


@yfinance_server.tool(
    name="get_stock_info",
    description="""Get stock information for a given ticker symbol from yahoo finance. Include the following information:
Stock Price & Trading Info, Company Information, Financial Metrics, Earnings & Revenue, Margins & Returns, Dividends, Balance Sheet, Ownership, Analyst Coverage, Risk Metrics, Other.

Args:
    ticker: str
        The ticker symbol of the stock to get information for, e.g. "AAPL"
""",
)
async def get_stock_info(ticker: str) -> str:
    """Get stock information for a given ticker symbol"""
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting stock information for {ticker}: {e}")
        return f"Error: getting stock information for {ticker}: {e}"
    info = company.info
    return json.dumps(info)


@yfinance_server.tool(
    name="get_yahoo_finance_news",
    description="""Get news for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get news for, e.g. "AAPL"
""",
)
async def get_yahoo_finance_news(ticker: str) -> str:
    """Get news for a given ticker symbol

    Args:
        ticker: str
            The ticker symbol of the stock to get news for, e.g. "AAPL"
    """
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting news for {ticker}: {e}")
        return f"Error: getting news for {ticker}: {e}"
    try:
        news = company.news
    except Exception as e:
        print(f"Error: getting news for {ticker}: {e}")
        return f"Error: getting news for {ticker}: {e}"

    news_list = []
    for news in company.news:
        if news.get("content", {}).get("contentType", "") == "STORY":
            title = news.get("content", {}).get("title", "")
            summary = news.get("content", {}).get("summary", "")
            description = news.get("content", {}).get("description", "")
            url = news.get("content", {}).get("canonicalUrl", {}).get("url", "")
            news_list.append(
                f"Title: {title}\nSummary: {summary}\nDescription: {description}\nURL: {url}"
            )
    if not news_list:
        print(f"No news found for company that searched with {ticker} ticker.")
        return f"No news found for company that searched with {ticker} ticker."
    return "\n\n".join(news_list)


@yfinance_server.tool(
    name="get_stock_actions",
    description="""Get stock dividends and stock splits for a given ticker symbol from yahoo finance.

Args:
    ticker: str
        The ticker symbol of the stock to get stock actions for, e.g. "AAPL"
""",
)
async def get_stock_actions(ticker: str) -> str:
    """Get stock dividends and stock splits for a given ticker symbol"""
    try:
        company = yf.Ticker(ticker)
    except Exception as e:
        print(f"Error: getting stock actions for {ticker}: {e}")
        return f"Error: getting stock actions for {ticker}: {e}"
    actions_df = company.actions
    actions_df = actions_df.reset_index(names="Date")
    return actions_df.to_json(orient="records", date_format="iso")


@yfinance_server.tool(
    name="get_financial_statement",
    description="""Get financial statement for a given ticker symbol from yahoo finance. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.

Args:
    ticker: str
        The ticker symbol of the stock to get financial statement for, e.g. "AAPL"
    financial_type: str
        The type of financial statement to get. You can choose from the following financial statement types: income_stmt, quarterly_income_stmt, balance_sheet, quarterly_balance_sheet, cashflow, quarterly_cashflow.
""",
)
async def get_financial_statement(ticker: str, financial_type: str) -> str:
    """Get financial statement for a given ticker symbol"""

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting financial statement for {ticker}: {e}")
        return f"Error: getting financial statement for {ticker}: {e}"

    if financial_type == FinancialType.income_stmt:
        financial_statement = company.income_stmt
    elif financial_type == FinancialType.quarterly_income_stmt:
        financial_statement = company.quarterly_income_stmt
    elif financial_type == FinancialType.balance_sheet:
        financial_statement = company.balance_sheet
    elif financial_type == FinancialType.quarterly_balance_sheet:
        financial_statement = company.quarterly_balance_sheet
    elif financial_type == FinancialType.cashflow:
        financial_statement = company.cashflow
    elif financial_type == FinancialType.quarterly_cashflow:
        financial_statement = company.quarterly_cashflow
    else:
        return f"Error: invalid financial type {financial_type}. Please use one of the following: {FinancialType.income_stmt}, {FinancialType.quarterly_income_stmt}, {FinancialType.balance_sheet}, {FinancialType.quarterly_balance_sheet}, {FinancialType.cashflow}, {FinancialType.quarterly_cashflow}."
    result = []

    for column in financial_statement.columns:
        if isinstance(column, pd.Timestamp):
            date_str = column.strftime("%Y-%m-%d")
        else:
            date_str = str(column)
        date_obj = {"date": date_str}
        for index, value in financial_statement[column].items():
            date_obj[index] = None if pd.isna(value) else value

        result.append(date_obj)

    return json.dumps(result)


@yfinance_server.tool(
    name="get_holder_info",
    description="""Get holder information for a given ticker symbol from yahoo finance. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.

Args:
    ticker: str
        The ticker symbol of the stock to get holder information for, e.g. "AAPL"
    holder_type: str
        The type of holder information to get. You can choose from the following holder types: major_holders, institutional_holders, mutualfund_holders, insider_transactions, insider_purchases, insider_roster_holders.
""",
)
async def get_holder_info(ticker: str, holder_type: str) -> str:
    """Get holder information for a given ticker symbol"""

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting holder info for {ticker}: {e}")
        return f"Error: getting holder info for {ticker}: {e}"

    if holder_type == HolderType.major_holders:
        return company.major_holders.reset_index(names="metric").to_json(orient="records")
    elif holder_type == HolderType.institutional_holders:
        return company.institutional_holders.to_json(orient="records")
    elif holder_type == HolderType.mutualfund_holders:
        return company.mutualfund_holders.to_json(orient="records", date_format="iso")
    elif holder_type == HolderType.insider_transactions:
        return company.insider_transactions.to_json(orient="records", date_format="iso")
    elif holder_type == HolderType.insider_purchases:
        return company.insider_purchases.to_json(orient="records", date_format="iso")
    elif holder_type == HolderType.insider_roster_holders:
        return company.insider_roster_holders.to_json(orient="records", date_format="iso")
    else:
        return f"Error: invalid holder type {holder_type}. Please use one of the following: {HolderType.major_holders}, {HolderType.institutional_holders}, {HolderType.mutualfund_holders}, {HolderType.insider_transactions}, {HolderType.insider_purchases}, {HolderType.insider_roster_holders}."


@yfinance_server.tool(
    name="get_option_expiration_dates",
    description="""Fetch the available options expiration dates for a given ticker symbol.

Args:
    ticker: str
        The ticker symbol of the stock to get option expiration dates for, e.g. "AAPL"
""",
)
async def get_option_expiration_dates(ticker: str) -> str:
    """Fetch the available options expiration dates for a given ticker symbol."""

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting option expiration dates for {ticker}: {e}")
        return f"Error: getting option expiration dates for {ticker}: {e}"
    return json.dumps(company.options)


@yfinance_server.tool(
    name="get_option_chain",
    description="""Fetch the option chain for a given ticker symbol, expiration date, and option type.

Args:
    ticker: str
        The ticker symbol of the stock to get option chain for, e.g. "AAPL"
    expiration_date: str
        The expiration date for the options chain (format: 'YYYY-MM-DD')
    option_type: str
        The type of option to fetch ('calls' or 'puts')
""",
)
async def get_option_chain(ticker: str, expiration_date: str, option_type: str) -> str:
    """Fetch the option chain for a given ticker symbol, expiration date, and option type.

    Args:
        ticker: The ticker symbol of the stock
        expiration_date: The expiration date for the options chain (format: 'YYYY-MM-DD')
        option_type: The type of option to fetch ('calls' or 'puts')

    Returns:
        str: JSON string containing the option chain data
    """

    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting option chain for {ticker}: {e}")
        return f"Error: getting option chain for {ticker}: {e}"

    if expiration_date not in company.options:
        return f"Error: No options available for the date {expiration_date}. You can use `get_option_expiration_dates` to get the available expiration dates."

    if option_type not in ["calls", "puts"]:
        return "Error: Invalid option type. Please use 'calls' or 'puts'."

    option_chain = company.option_chain(expiration_date)
    if option_type == "calls":
        return option_chain.calls.to_json(orient="records", date_format="iso")
    elif option_type == "puts":
        return option_chain.puts.to_json(orient="records", date_format="iso")
    else:
        return f"Error: invalid option type {option_type}. Please use one of the following: calls, puts."


@yfinance_server.tool(
    name="get_recommendations",
    description="""Get recommendations or upgrades/downgrades for a given ticker symbol from yahoo finance. You can also specify the number of months back to get upgrades/downgrades for, default is 12.

Args:
    ticker: str
        The ticker symbol of the stock to get recommendations for, e.g. "AAPL"
    recommendation_type: str
        The type of recommendation to get. You can choose from the following recommendation types: recommendations, upgrades_downgrades.
    months_back: int
        The number of months back to get upgrades/downgrades for, default is 12.
""",
)
async def get_recommendations(ticker: str, recommendation_type: str, months_back: int = 12) -> str:
    """Get recommendations or upgrades/downgrades for a given ticker symbol"""
    company = yf.Ticker(ticker)
    try:
        if company.isin is None:
            print(f"Company ticker {ticker} not found.")
            return f"Company ticker {ticker} not found."
    except Exception as e:
        print(f"Error: getting recommendations for {ticker}: {e}")
        return f"Error: getting recommendations for {ticker}: {e}"
    try:
        if recommendation_type == RecommendationType.recommendations:
            return company.recommendations.to_json(orient="records")
        elif recommendation_type == RecommendationType.upgrades_downgrades:
            upgrades_downgrades = company.upgrades_downgrades.reset_index()
            cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_back)
            upgrades_downgrades = upgrades_downgrades[
                upgrades_downgrades["GradeDate"] >= cutoff_date
            ]
            upgrades_downgrades = upgrades_downgrades.sort_values("GradeDate", ascending=False)
            latest_by_firm = upgrades_downgrades.drop_duplicates(subset=["Firm"])
            return latest_by_firm.to_json(orient="records", date_format="iso")
    except Exception as e:
        print(f"Error: getting recommendations for {ticker}: {e}")
        return f"Error: getting recommendations for {ticker}: {e}"


@yfinance_server.tool(
    name="get_target_price",
    description="""Get the 1-year target price estimate for a given ticker symbol from Yahoo Finance.
Args:
    ticker: str
        The ticker symbol of the stock to get the target price for, e.g. "AAPL"
""",
)
def get_target_price(ticker: str) -> str:
    """
    Fetch the 1-year analyst target price for the given ticker symbol.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        float: Target price estimate, or None if unavailable
    """
    try:
        ticker = yf.Ticker(ticker)
        price_target = ticker.info.get('targetMeanPrice')

        if price_target is None:
            return None
        return f"Price Target for {ticker} is : {price_target:.2f}\nDate & Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"Error fetching target price for {ticker}: {str(e)}"

@yfinance_server.tool(
    name="get_news_sentiment_and_price_prediction",
    description="""Fetch company-specific news headlines from Yahoo Finance,
    perform sentiment analysis on headlines,
    and predict stock price movement (UP/DOWN/STABLE) based on average sentiment.
    Args:
        ticker: str
            Stock ticker symbol, e.g. "AAPL"
    """,
)
def get_news_sentiment_and_price_prediction(ticker: str) -> dict:
    """
    Get news headlines for the ticker, analyze sentiment, and predict price movement.
    
    Returns a dict with:
        - headlines: list of headlines strings
        - sentiment_scores: list of float sentiment polarity scores
        - average_sentiment: float average sentiment
        - prediction: str (UP, DOWN, STABLE)
    """
    company = yf.Ticker(ticker)
    try:
        news = company.news
    except Exception as e:
        return {"error": f"Failed to fetch news: {str(e)}"}

    company_name = company.info.get("shortName", "").lower()
    company_first = company_name.split(" ")[0] if company_name else ""
    ticker_upper = ticker.upper()

    headlines = []
    count = 1
    for item in news:
        title = item.get("content", {}).get("title", "")
        title_lower = title.lower()
        if company_name and (company_name in title_lower or ticker_upper in title or company_first in title_lower):
            headlines.append(f"Headline {count}: {title}")
            count += 1

    if not headlines:
        return {"error": f"No relevant news found for ticker {ticker}."}

    sentiment_scores = []
    for headline in headlines:
        blob = TextBlob(headline)
        polarity = blob.sentiment.polarity
        sentiment_scores.append(polarity)

    average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
    if average_sentiment > 0.1:
        prediction = "Stock price likely to go UP"
    elif average_sentiment < -0.1:
        prediction = "Stock price likely to go DOWN"
    else:
        prediction = "Stock price likely to remain STABLE"

    return {
        "headlines": headlines,
        "sentiment_scores": sentiment_scores,
        "average_sentiment": average_sentiment,
        "prediction": prediction,
        "datetime": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@yfinance_server.tool(
    name="get_stock_5_year_projection",
    description="""
    Analyze stock growth and revenue projection over the last 5 years using Yahoo Finance data.

    Args:
        ticker: str
            The stock ticker symbol, e.g., "AAPL"
    Returns:
        JSON with 5Y price CAGR, 5Y revenue CAGR, and 5-year projected revenue.
    """,
)
async def get_stock_5_year_projection(ticker: str) -> str:
    """Analyze 5-year price CAGR, revenue CAGR, and project future revenue."""
    stock = yf.Ticker(ticker)

    try:
        hist = stock.history(period="5y")
        if hist.empty:
            return json.dumps({"error": f"No historical data found for {ticker}"})
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch historical data: {str(e)}"})

    start_price = hist["Close"].iloc[0]
    end_price = hist["Close"].iloc[-1]
    years = 5
    cagr_price = ((end_price / start_price) ** (1 / years) - 1) * 100

    try:
        financials = stock.financials.loc["Total Revenue"]
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch financials: {str(e)}"})

    revenue_df = financials.reset_index()
    revenue_df.columns = ["Date", "Revenue"]
    revenue_df["Date"] = pd.to_datetime(revenue_df["Date"]).dt.year
    revenue_df = revenue_df.dropna(subset=["Revenue"])

    if revenue_df.empty or len(revenue_df) < 2:
        return json.dumps({"error": "Not enough revenue data to compute CAGR."})

    rev_start = revenue_df["Revenue"].iloc[-1]
    rev_end = revenue_df["Revenue"].iloc[0]
    years_rev = len(revenue_df) - 1
    cagr_revenue = ((rev_end / rev_start) ** (1 / years_rev) - 1) * 100

    future_years = []
    future_revenue = []
    last_year = revenue_df["Date"].max()
    last_revenue = revenue_df["Revenue"].iloc[0]

    for i in range(1, 6):
        year = last_year + i
        last_revenue *= (1 + cagr_revenue / 100)
        future_years.append(year)
        future_revenue.append(round(last_revenue, 2))

    projection = {str(year): f"{rev:,.2f}" for year, rev in zip(future_years, future_revenue)}

    result = {
        "Stock": ticker,
        "5Y Price CAGR (%)": round(cagr_price, 2),
        "5Y Revenue CAGR (%)": round(cagr_revenue, 2),
        "Revenue Projection (Next 5 Years)": projection
    }

    return json.dumps(result, indent=2)


def _validate_period(period: str) -> dict:
    """
    Validate the requested period and return guidance if invalid.
    
    Returns:
        dict: {"valid": bool, "error": str, "suggestions": list} if invalid
              {"valid": True} if valid
    """
    period_lower = period.lower().strip()
    
    # Check for monthly periods (invalid)
    monthly_indicators = ['january', 'february', 'march', 'april', 'may', 'june',
                          'july', 'august', 'september', 'october', 'november', 'december',
                          'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    if any(month in period_lower for month in monthly_indicators):
        return {
            "valid": False,
            "error": "Monthly financial ratios are not available",
            "reason": "Companies only publish financial statements quarterly and annually, not monthly.",
            "minimum_required": "quarterly or annual",
            "suggested_periods": [
                "TTM (Trailing Twelve Months - most current)",
                "Q1 2025", "Q2 2025", "Q3 2025", "Q4 2024",
                "FY2024 (Fiscal Year 2024)",
                "2024 (Annual)",
                "2023 (Annual)"
            ]
        }
    
    # Check for daily periods (invalid)
    if 'day' in period_lower or 'daily' in period_lower:
        return {
            "valid": False,
            "error": "Daily financial ratios are not available",
            "reason": "Financial ratios are calculated from balance sheets and income statements, which are published quarterly or annually.",
            "minimum_required": "quarterly or annual",
            "suggested_periods": [
                "TTM (Trailing Twelve Months)",
                "Q3 2024", "Q2 2024",
                "2024 (Annual)"
            ]
        }
    
    return {"valid": True}


def _parse_period(period: str, balance_sheet_df, financials_df):
    """
    Parse period string and return appropriate column index from financial statements.
    
    Args:
        period: Period string (e.g., "2024", "Q3 2024", "TTM")
        balance_sheet_df: Balance sheet DataFrame
        financials_df: Financials DataFrame
    
    Returns:
        tuple: (column_index or None, use_ttm: bool, error_message or None)
    """
    period_str = period.strip().upper()
    
    # Handle TTM/Latest
    if period_str in ['TTM', 'LATEST', 'CURRENT']:
        return (None, True, None)  # Use TTM data from stock.info
    
    # Handle quarterly (e.g., "Q3 2024", "Q1 2025")
    if period_str.startswith('Q'):
        # For quarterly data, we need to use quarterly statements
        # This is a simplified implementation - you'd need quarterly_balance_sheet and quarterly_financials
        return (None, True, "Quarterly historical data requires using quarterly financial statements. Using TTM data instead.")
    
    # Handle annual (e.g., "2024", "FY2024")
    year_str = period_str.replace('FY', '').replace('FISCAL', '').strip()
    
    try:
        year = int(year_str)
    except ValueError:
        return (None, True, f"Could not parse year from '{period}'. Using TTM data.")
    
    # Try to find matching column in balance sheet
    if not balance_sheet_df.empty:
        for idx, col in enumerate(balance_sheet_df.columns):
            if hasattr(col, 'year') and col.year == year:
                return (idx, False, None)
    
    # Year not found in available data
    available_years = []
    if not balance_sheet_df.empty:
        available_years = [col.year for col in balance_sheet_df.columns if hasattr(col, 'year')]
    
    if available_years:
        return (None, True, f"Data for year {year} not available. Available years: {available_years}. Using TTM data.")
    else:
        return (None, True, f"No historical data available. Using TTM data.")


async def _calculate_financial_ratios_logic(ticker: str, period: str = "TTM") -> str:
    """Calculate key financial ratios independently."""
    
    # Step 1: Validate the period
    validation_result = _validate_period(period)
    if not validation_result["valid"]:
        return json.dumps({
            "error": validation_result["error"],
            "reason": validation_result["reason"],
            "minimum_required": validation_result["minimum_required"],
            "suggested_periods": validation_result["suggested_periods"],
            "ticker": ticker,
            "requested_period": period
        }, indent=2)
    
    stock = yf.Ticker(ticker)

    try:
        info = stock.info
        # Try to get financial statements, but don't fail if some are missing
        try:
            bs = stock.balance_sheet
            ist = stock.financials
        except Exception:
            bs = pd.DataFrame()
            ist = pd.DataFrame()
            
    except Exception as e:
        return json.dumps({"error": f"Failed to fetch data for {ticker}: {str(e)}"})
    
    # Step 2: Parse the period to determine which column to use
    col_idx, use_ttm, parse_warning = _parse_period(period, bs, ist)
    
    ratios = {}

    # Helper function to safe get from info
    def get_info(key):
        return info.get(key)

    # Helper function to safe get from dataframe (with period-aware column selection)
    def get_fs_item(df, key):
        if df.empty: return None
        try:
            # Check if key exists in index
            matches = [i for i in df.index if key.lower() in str(i).lower()]
            if not matches: return None
            # Use specified column index or default to most recent (column 0)
            idx_to_use = col_idx if col_idx is not None else 0
            if idx_to_use >= len(df.loc[matches[0]]):
                return None
            val = df.loc[matches[0]].iloc[idx_to_use]
            return val if not pd.isna(val) else None
        except:
            return None

    # --- 1. Liquidity Ratios ---
    current_ratio = get_info('currentRatio') if use_ttm else None
    quick_ratio = get_info('quickRatio') if use_ttm else None
    
    # Calculate if missing
    if current_ratio is None:
        ca = get_fs_item(bs, "Current Assets")
        cl = get_fs_item(bs, "Current Liabilities")
        if ca and cl and cl != 0:
            current_ratio = ca / cl
            
    ratios['Liquidity'] = {
        "Current Ratio": round(current_ratio, 2) if current_ratio else None,
        "Quick Ratio": round(quick_ratio, 2) if quick_ratio else None
    }

    # --- 2. Solvency Ratios ---
    debt_to_equity = get_info('debtToEquity') # Usually returned as percentage in info (e.g., 150 for 1.5)
    
    total_debt = get_info('totalDebt')
    if total_debt is None:
        total_debt = get_fs_item(bs, "Total Debt")
        
    total_equity = get_fs_item(bs, "Total Equity") or get_fs_item(bs, "Stockholders Equity")
    total_assets = get_fs_item(bs, "Total Assets")
    
    if debt_to_equity is None and total_debt and total_equity and total_equity != 0:
        debt_to_equity = (total_debt / total_equity) * 100 # Keep consistent with yfinance % format

    debt_to_assets = None
    if total_debt and total_assets and total_assets != 0:
        debt_to_assets = total_debt / total_assets

    ratios['Solvency'] = {
        "Debt-to-Equity": round(debt_to_equity, 2) if debt_to_equity else None,
        "Debt-to-Assets": round(debt_to_assets, 2) if debt_to_assets else None,
        "Total Debt": total_debt,
        "Total Equity": total_equity
    }

    # --- 3. Profitability Ratios ---
    profit_margin = get_info('profitMargins') # decimal
    roe = get_info('returnOnEquity') # decimal
    roa = get_info('returnOnAssets') # decimal
    
    # Calculate Gross Margin
    gross_margin = None
    revenue = get_info('totalRevenue') or get_fs_item(ist, "Total Revenue")
    gross_profit = get_info('grossProfits') or get_fs_item(ist, "Gross Profit")
    
    if revenue and gross_profit and revenue != 0:
        gross_margin = gross_profit / revenue

    ratios['Profitability'] = {
        "Profit Margin": round(profit_margin * 100, 2) if profit_margin else None,
        "Return on Equity (ROE)": round(roe * 100, 2) if roe else None,
        "Return on Assets (ROA)": round(roa * 100, 2) if roa else None,
        "Gross Margin": round(gross_margin * 100, 2) if gross_margin else None
    }

    # --- 4. Valuation Ratios ---
    pe_ratio = get_info('trailingPE')
    forward_pe = get_info('forwardPE')
    peg_ratio = get_info('pegRatio')
    pb_ratio = get_info('priceToBook')
    ps_ratio = get_info('priceToSalesTrailing12Months')
    
    ratios['Valuation'] = {
        "P/E Ratio": round(pe_ratio, 2) if pe_ratio else None,
        "Forward P/E": round(forward_pe, 2) if forward_pe else None,
        "PEG Ratio": round(peg_ratio, 2) if peg_ratio else None,
        "P/B Ratio": round(pb_ratio, 2) if pb_ratio else None,
        "P/S Ratio": round(ps_ratio, 2) if ps_ratio else None
    }

    # --- 5. Efficiency Ratios ---
    asset_turnover = None
    if revenue and total_assets and total_assets != 0:
        asset_turnover = revenue / total_assets
        
    ratios['Efficiency'] = {
        "Asset Turnover": round(asset_turnover, 2) if asset_turnover else None,
        "Revenue per Share": get_info('revenuePerShare')
    }

    result = {
        "ticker": ticker,
        "period_requested": period,
        "period_type": "TTM" if use_ttm else f"Annual {bs.columns[col_idx].year if col_idx is not None and not bs.empty else 'Unknown'}",
        "ratios": ratios,
        "currency": get_info('currency'),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add warning if period parsing had issues
    if parse_warning:
        result["warning"] = parse_warning
    
    return json.dumps(result, indent=2)


@yfinance_server.tool(
    name="get_financial_ratios",
    description="""
    Calculate comprehensive financial ratios for a given ticker symbol using Yahoo Finance data.
    Includes Liquidity, Solvency, Profitability, Valuation, and Efficiency ratios.

    Args:
        ticker: str
            The stock ticker symbol, e.g., "AAPL"
        period: str (optional, default="TTM")
            The time period for financial data:
            - "TTM" or "latest": Trailing Twelve Months (most recent, real-time)
            - "2024" or "FY2024": Annual data from fiscal year 2024
            - "2023": Annual data from fiscal year 2023
            - "Q3 2024": Quarterly data (note: uses TTM if quarterly statements not available)
            
            Note: Monthly or daily periods are NOT supported as companies only publish 
            financial statements quarterly or annually.
    Returns:
        JSON with calculated ratios and metrics, along with period information.
    """,
)
async def get_financial_ratios(ticker: str, period: str = "TTM") -> str:
    """Calculate key financial ratios independently."""
    return await _calculate_financial_ratios_logic(ticker, period)


if __name__ == "__main__":
    print("Starting Yahoo Finance MCP server...")
    yfinance_server.run(transport="streamable-http", port=8565)
