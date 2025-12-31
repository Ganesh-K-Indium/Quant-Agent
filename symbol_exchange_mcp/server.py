"""
Symbol & Exchange Management MCP Server
Handles ticker lookup, ISIN resolution, exchange mapping, and symbol validation
"""
import json
import sqlite3
import yfinance as yf
from fastmcp import FastMCP
from datetime import datetime
from typing import Optional, Dict, List
import asyncio
from pathlib import Path
import re
from langchain_community.tools import TavilySearchResults
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize FastMCP server
symbol_exchange_server = FastMCP(
    "symbol_exchange",
    instructions="""
# Symbol & Exchange Management MCP Server

This server provides comprehensive symbol and exchange management capabilities for stock trading.

Available tools:
- find_ticker: Find ticker symbol from company name (supports multiple exchanges)
- get_symbol_info: Get detailed information about a ticker (ISIN, exchange, company info)
- resolve_isin: Find all tickers for a given ISIN across exchanges
- validate_ticker: Check if a ticker symbol is valid
- get_exchange_info: Get information about supported exchanges
- map_cross_listings: Find cross-listings of the same company on different exchanges
- standardize_symbol: Convert symbol to preferred exchange format
- search_similar_tickers: Find similar ticker symbols
""",
)

# Database setup
DB_PATH = Path(__file__).parent / "symbol_mapping.db"


def init_database():
    """Initialize SQLite database for symbol mappings"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Symbol mapping table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS symbol_mappings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            isin TEXT,
            ticker TEXT NOT NULL,
            exchange TEXT,
            exchange_suffix TEXT,
            company_name TEXT,
            country TEXT,
            currency TEXT,
            instrument_type TEXT,
            primary_listing BOOLEAN DEFAULT 0,
            source TEXT DEFAULT 'yahoo',
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(ticker, exchange)
        )
    """)
    
    # Exchange information table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exchange_info (
            exchange_code TEXT PRIMARY KEY,
            exchange_name TEXT,
            country TEXT,
            timezone TEXT,
            currency TEXT,
            suffix TEXT,
            supported BOOLEAN DEFAULT 1
        )
    """)
    
    # Populate common exchanges
    exchanges = [
        ('NASDAQ', 'NASDAQ Stock Market', 'US', 'America/New_York', 'USD', '', 1),
        ('NYSE', 'New York Stock Exchange', 'US', 'America/New_York', 'USD', '', 1),
        ('NSE', 'National Stock Exchange of India', 'IN', 'Asia/Kolkata', 'INR', '.NS', 1),
        ('BSE', 'Bombay Stock Exchange', 'IN', 'Asia/Kolkata', 'INR', '.BO', 1),
        ('LSE', 'London Stock Exchange', 'GB', 'Europe/London', 'GBP', '.L', 1),
        ('HKEX', 'Hong Kong Stock Exchange', 'HK', 'Asia/Hong_Kong', 'HKD', '.HK', 1),
        ('TSE', 'Tokyo Stock Exchange', 'JP', 'Asia/Tokyo', 'JPY', '.T', 1),
        ('SSE', 'Shanghai Stock Exchange', 'CN', 'Asia/Shanghai', 'CNY', '.SS', 1),
        ('SZSE', 'Shenzhen Stock Exchange', 'CN', 'Asia/Shanghai', 'CNY', '.SZ', 1),
        ('TSX', 'Toronto Stock Exchange', 'CA', 'America/Toronto', 'CAD', '.TO', 1),
        ('ASX', 'Australian Securities Exchange', 'AU', 'Australia/Sydney', 'AUD', '.AX', 1),
    ]
    
    cursor.executemany(
        "INSERT OR IGNORE INTO exchange_info VALUES (?, ?, ?, ?, ?, ?, ?)",
        exchanges
    )
    
    conn.commit()
    conn.close()


# Initialize database on server start
init_database()

# Initialize Tavily search tool
tavily_search = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=False,
    include_images=False,
    include_domains=["https://finance.yahoo.com/"],
)


async def search_ticker_with_tavily(query: str, search_type: str = "company") -> List[Dict]:
    """Use Tavily to search for ticker symbol on Yahoo Finance
    
    Args:
        query: Company name or ISIN to search for
        search_type: "company" for company name search, "isin" for ISIN search
    
    Returns:
        List of dicts with ticker, company_name, exchange info
    """
    try:
        # Build search query based on type
        if search_type == "isin":
            search_query = f"site:finance.yahoo.com ISIN {query} stock ticker symbol"
        else:
            search_query = f"site:finance.yahoo.com {query} stock ticker symbol quote"
        
        print(f"Searching Tavily for: {search_query}")
        
        # Execute search
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: tavily_search.invoke({"query": search_query}))
        
        if not results:
            return []
        
        # Extract ticker from results
        ticker_patterns = [
            r'/quote/([A-Z0-9\.\-]+)',  # From Yahoo Finance URLs
            r'symbol[=/]([A-Z0-9\.\-]+)',  # symbol= or symbol/
            r'\(([A-Z]{2,5}(?:\.[A-Z]{1,3})?)\)',  # (TICKER) or (TICKER.NS) in text
            r'ticker[:\s]+([A-Z0-9\.\-]+)',  # ticker: TICKER
            r'symbol[:\s]+([A-Z0-9\.\-]+)',  # symbol: TICKER
        ]
        
        found_items = []
        seen_tickers = set()
        
        for result in results:
            # Check URL
            url = result.get('url', '')
            content = result.get('content', '') + ' ' + result.get('snippet', '')
            
            for pattern in ticker_patterns:
                matches = re.findall(pattern, url, re.IGNORECASE)
                for match in matches:
                    ticker = match.upper()
                    if len(ticker) >= 2 and len(ticker) <= 15 and ticker not in seen_tickers:
                        seen_tickers.add(ticker)
                        found_items.append({
                            'ticker': ticker,
                            'url': url,
                            'content_snippet': content[:200]
                        })
        
        print(f"Found {len(found_items)} potential tickers: {[item['ticker'] for item in found_items]}")
        return found_items
        
    except Exception as e:
        print(f"Error in Tavily search: {e}")
        return []


async def search_cross_listings(company_name: str, base_ticker: Optional[str] = None) -> List[Dict]:
    """Search for cross-listings of a company across multiple exchanges"""
    try:
        # Build comprehensive search query
        search_terms = [company_name]
        if base_ticker:
            search_terms.append(base_ticker)
        
        query = f"site:finance.yahoo.com {' '.join(search_terms)} stock exchanges listed NSE BSE NASDAQ NYSE LSE"
        print(f"Searching for cross-listings: {query}")
        
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, lambda: tavily_search.invoke({"query": query}))
        
        # Extract all unique tickers from results
        ticker_pattern = r'/quote/([A-Z0-9\.\-]+)'
        found_tickers = set()
        
        for result in results:
            url = result.get('url', '')
            matches = re.findall(ticker_pattern, url)
            found_tickers.update(m.upper() for m in matches)
        
        # Validate each ticker
        validated_listings = []
        for ticker in found_tickers:
            info = await fetch_yahoo_info(ticker)
            if info:
                # Check if it's the same company (basic name matching)
                fetched_name = info.get('company_name', '').lower()
                search_words = [w.lower() for w in company_name.split() if len(w) > 3]
                if any(word in fetched_name for word in search_words):
                    validated_listings.append(info)
        
        return validated_listings
        
    except Exception as e:
        print(f"Error searching cross-listings: {e}")
        return []


async def search_ticker_by_isin(isin: str) -> List[Dict]:
    """Search for all tickers associated with an ISIN across exchanges"""
    try:
        print(f"Searching web for ISIN: {isin}")
        
        # Search on Yahoo Finance and other financial sites
        search_results = await search_ticker_with_tavily(isin, search_type="isin")
        
        # Validate each ticker found
        validated_tickers = []
        for item in search_results:
            ticker = item['ticker']
            info = await fetch_yahoo_info(ticker)
            if info and info.get('isin') == isin:
                validated_tickers.append(info)
        
        # Also try common exchange suffixes if we have partial info
        if not validated_tickers and len(isin) == 12:
            country_code = isin[:2]
            # Map country codes to exchanges
            exchange_map = {
                'US': ['', '.US'],  # NASDAQ/NYSE
                'IN': ['.NS', '.BO'],  # NSE, BSE
                'GB': ['.L'],  # London
                'HK': ['.HK'],  # Hong Kong
                'JP': ['.T'],  # Tokyo
                'CN': ['.SS', '.SZ'],  # Shanghai, Shenzhen
            }
            
            suffixes = exchange_map.get(country_code, [])
            # Try searching with country-specific terms
            country_search = f"site:finance.yahoo.com ISIN {isin} stock ticker"
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, lambda: tavily_search.invoke({"query": country_search}))
            
            for result in results:
                url = result.get('url', '')
                ticker_match = re.search(r'/quote/([A-Z0-9\.\-]+)', url)
                if ticker_match:
                    ticker = ticker_match.group(1).upper()
                    info = await fetch_yahoo_info(ticker)
                    if info and info.get('isin') == isin:
                        validated_tickers.append(info)
        
        return validated_tickers
        
    except Exception as e:
        print(f"Error searching ticker by ISIN: {e}")
        return []


async def fetch_yahoo_info(ticker: str) -> Optional[Dict]:
    """Fetch ticker information from Yahoo Finance"""
    try:
        loop = asyncio.get_event_loop()
        company = await loop.run_in_executor(None, yf.Ticker, ticker)
        info = await loop.run_in_executor(None, lambda: company.info)
        
        if not info or info.get('regularMarketPrice') is None:
            return None
        
        return {
            'ticker': ticker,
            'isin': getattr(company, 'isin', None),
            'company_name': info.get('longName') or info.get('shortName'),
            'exchange': info.get('exchange'),
            'country': info.get('country'),
            'currency': info.get('currency'),
            'instrument_type': info.get('quoteType', 'EQUITY'),
            'market_cap': info.get('marketCap'),
            'sector': info.get('sector'),
            'industry': info.get('industry')
        }
    except Exception as e:
        print(f"Error fetching Yahoo info for {ticker}: {e}")
        return None


def save_symbol_mapping(mapping: Dict):
    """Save symbol mapping to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO symbol_mappings 
        (isin, ticker, exchange, company_name, country, currency, instrument_type, source, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        mapping.get('isin'),
        mapping['ticker'],
        mapping.get('exchange'),
        mapping.get('company_name'),
        mapping.get('country'),
        mapping.get('currency'),
        mapping.get('instrument_type', 'EQUITY'),
        mapping.get('source', 'yahoo'),
        datetime.now()
    ))
    
    conn.commit()
    conn.close()


def get_symbol_from_db(ticker: str, exchange: Optional[str] = None) -> Optional[Dict]:
    """Retrieve symbol mapping from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if exchange:
        cursor.execute("""
            SELECT * FROM symbol_mappings 
            WHERE ticker = ? AND exchange = ?
            ORDER BY last_updated DESC LIMIT 1
        """, (ticker, exchange))
    else:
        cursor.execute("""
            SELECT * FROM symbol_mappings 
            WHERE ticker = ?
            ORDER BY primary_listing DESC, last_updated DESC LIMIT 1
        """, (ticker,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        columns = ['id', 'isin', 'ticker', 'exchange', 'exchange_suffix', 'company_name', 
                   'country', 'currency', 'instrument_type', 'primary_listing', 'source', 'last_updated']
        return dict(zip(columns, row))
    return None


@symbol_exchange_server.tool(
    name="find_ticker",
    description="""**EXPERT TICKER DISCOVERY TOOL** - Primary tool for resolving company names to ticker symbols with disambiguation support.
    
**USE THIS TOOL FIRST** when user provides a company name that might be ambiguous.

This tool uses web search (Tavily + Yahoo Finance) to find the most accurate ticker matches, then validates each result.

Args:
    company_name: str
        The company name to search for (e.g., "Apple", "Reliance Industries", "Tesla")
        Can be partial or full name - the tool will search intelligently
        
    preferred_exchange: str (optional)
        Filter results by exchange code to reduce ambiguity
        Examples: "NASDAQ", "NYSE", "NSE", "BSE", "LSE", "HKEX"
        Use this when user specifies exchange preference
        
    country: str (optional)
        Filter by country code to narrow results
        Examples: "US", "IN" (India), "GB" (UK), "HK" (Hong Kong)
        
Returns:
    JSON with primary match AND alternative_listings array:
    {
        "success": true,
        "ticker": "AAPL",           // Best match ticker
        "isin": "US0378331005",     // ISIN for cross-reference
        "exchange": "NASDAQ",        // Primary exchange
        "company_name": "Apple Inc.",
        "country": "US",
        "currency": "USD",
        "instrument_type": "EQUITY",
        "source": "yahoo_finance",
        "search_method": "web_search",
        "alternative_listings": [    // Other exchanges if available
            {
                "ticker": "AAPL.L",
                "exchange": "LSE",
                "currency": "GBP"
            }
        ]
    }

**AGENT INSTRUCTIONS:**
- When this returns multiple alternative_listings, ALWAYS present them to the user
- Ask user to confirm which exchange they want before proceeding
- Use ISIN from result for cross-listing queries
- If ambiguous company name (e.g., "Apple"), present all options and ask for clarification

**EXAMPLES:**

✓ Ambiguous query:
  find_ticker(company_name="Apple")
  → Returns AAPL with potential alternatives
  → Agent should present: "Found Apple Inc. (AAPL on NASDAQ). Is this correct?"

✓ Specific exchange:
  find_ticker(company_name="Reliance Industries", preferred_exchange="NSE")
  → Returns RELIANCE.NS specifically

✓ Country filter:
  find_ticker(company_name="Tata Consultancy", country="IN")
  → Returns TCS.NS (India)
""")
async def find_ticker(company_name: str, preferred_exchange: Optional[str] = None, country: Optional[str] = None) -> str:
    """Find ticker symbol from company name"""
    try:
        # Get exchange suffix if specified
        exchange_suffix = ""
        if preferred_exchange:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT suffix FROM exchange_info WHERE exchange_code = ?", (preferred_exchange,))
            result = cursor.fetchone()
            conn.close()
            if result:
                exchange_suffix = result[0] or ""
        
        results = []
        
        # Strategy 1: Check database cache first for recent lookups
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM symbol_mappings 
            WHERE LOWER(company_name) LIKE ? 
            AND (? IS NULL OR exchange = ?)
            ORDER BY last_updated DESC LIMIT 5
        """, (f"%{company_name.lower()}%", preferred_exchange, preferred_exchange))
        cached_rows = cursor.fetchall()
        conn.close()
        
        if cached_rows:
            print(f"Found {len(cached_rows)} cached results for {company_name}")
            for row in cached_rows:
                results.append({
                    'ticker': row[2],
                    'isin': row[1],
                    'exchange': row[3],
                    'company_name': row[5],
                    'country': row[6],
                    'currency': row[7],
                    'instrument_type': row[8],
                    'source': 'database_cache'
                })
        
        # Strategy 2: Use Tavily web search to find ALL matching tickers
        if not results:
            print(f"Strategy 2: Searching with Tavily for {company_name}")
            
            # Build search query - search broadly to find all matches
            search_query = company_name
            if preferred_exchange:
                search_query += f" {preferred_exchange}"
            elif country:
                search_query += f" {country}"
            
            search_results = await search_ticker_with_tavily(search_query, search_type="company")
            
            # Validate EACH ticker found - don't break after first match
            validated_count = 0
            for item in search_results:
                ticker = item['ticker']
                
                # Apply exchange filter if specified
                if preferred_exchange:
                    if preferred_exchange == "NSE" and not ticker.endswith('.NS'):
                        continue
                    elif preferred_exchange == "BSE" and not ticker.endswith('.BO'):
                        continue
                    elif preferred_exchange in ["NASDAQ", "NYSE"] and '.' in ticker:
                        continue
                
                # Fetch and validate
                print(f"Validating ticker from web search: {ticker}")
                info = await fetch_yahoo_info(ticker)
                if info:
                    # Verify it matches the company name
                    fetched_name = info.get('company_name', '').lower()
                    search_words = [w.lower() for w in company_name.split() if len(w) > 3]
                    # Match if any search word appears in company name
                    if any(word in fetched_name for word in search_words) or company_name.lower() in fetched_name:
                        results.append(info)
                        validated_count += 1
                        print(f"✓ Validated: {ticker} -> {info.get('company_name')}")
                        # Continue searching to find ALL matches, not just first one
                        if validated_count >= 5:  # Limit to 5 matches max
                            break
        
        # Strategy 3: Search for similar companies and cross-listings
        if results:
            if len(results) == 1:
                # Search for cross-listings of the same company
                print(f"Strategy 3a: Searching for cross-listings of {results[0]['company_name']}")
                cross_listings = await search_cross_listings(
                    company_name=results[0]['company_name'],
                    base_ticker=results[0]['ticker']
                )
                for listing in cross_listings:
                    if preferred_exchange and listing.get('exchange') == preferred_exchange:
                        results.insert(0, listing)  # Prioritize preferred exchange
                    elif listing['ticker'] not in [r['ticker'] for r in results]:
                        results.append(listing)
            
            # Search for OTHER companies with similar names (e.g., Apple vs Apple Hospitality)
            if len(results) < 3:
                print(f"Strategy 3b: Searching for other companies with similar names")
                # Broader search without company name exact match
                broader_search = f"site:finance.yahoo.com {company_name} stock company ticker"
                loop = asyncio.get_event_loop()
                broader_results = await loop.run_in_executor(None, lambda: tavily_search.invoke({"query": broader_search}))
                
                ticker_pattern = r'/quote/([A-Z0-9\.\-]+)'
                found_tickers = set([r['ticker'] for r in results])
                
                for result in broader_results[:10]:  # Check more results
                    url = result.get('url', '')
                    matches = re.findall(ticker_pattern, url)
                    for ticker in matches:
                        if ticker.upper() not in found_tickers and len(results) < 5:
                            info = await fetch_yahoo_info(ticker.upper())
                            if info:
                                # Check if company name contains ANY word from search
                                fetched_name = info.get('company_name', '').lower()
                                search_words = [w.lower() for w in company_name.split() if len(w) > 3]
                                if any(word in fetched_name for word in search_words):
                                    results.append(info)
                                    found_tickers.add(ticker.upper())
                                    print(f"✓ Found similar company: {ticker.upper()} -> {info.get('company_name')}")
        
        if not results:
            return json.dumps({
                "success": False,
                "error": f"No ticker found for company: {company_name}",
                "suggestion": "Please try:\n1. Provide the exact ticker symbol if known\n2. Use the full official company name\n3. Specify the country or preferred exchange\n4. For Indian companies, try NSE or BSE as preferred_exchange"
            })
        
        # Get the best match (prioritize preferred exchange)
        best_match = results[0]
        
        # Save all results to database
        for result in results:
            if result.get('source') != 'database_cache':
                save_symbol_mapping(result)
        
        # Build comprehensive response with ALL matches
        response = {
            "success": True,
            "match_count": len(results),
            "primary_match": {
                "ticker": best_match['ticker'],
                "isin": best_match.get('isin'),
                "exchange": best_match.get('exchange'),
                "company_name": best_match.get('company_name'),
                "country": best_match.get('country'),
                "currency": best_match.get('currency'),
                "instrument_type": best_match.get('instrument_type'),
                "market_cap": best_match.get('market_cap'),
                "sector": best_match.get('sector')
            },
            "search_method": "web_search",
            "source": best_match.get('source', 'yahoo_finance')
        }
        
        # Include ALL other matches with full details for disambiguation
        if len(results) > 1:
            response["alternative_matches"] = [
                {
                    "ticker": r['ticker'],
                    "isin": r.get('isin'),
                    "exchange": r.get('exchange'),
                    "company_name": r.get('company_name'),
                    "country": r.get('country'),
                    "currency": r.get('currency'),
                    "market_cap": r.get('market_cap'),
                    "sector": r.get('sector'),
                    "instrument_type": r.get('instrument_type')
                } for r in results[1:]
            ]
            response["disambiguation_required"] = True
            response["note"] = f"Found {len(results)} securities matching '{company_name}'. Review all matches and confirm your selection."
        else:
            response["disambiguation_required"] = False
        
        return json.dumps(response, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@symbol_exchange_server.tool(
    name="get_symbol_info",
    description="""Get detailed information about a ticker symbol including ISIN, exchange, and company metadata.
    
Args:
    ticker: str
        The ticker symbol (e.g., "AAPL", "RELIANCE.NS", "TCS.BO")
    fetch_live: bool (optional)
        Whether to fetch fresh data from Yahoo Finance (default: True)
        
Returns:
    JSON with comprehensive symbol information
""")
async def get_symbol_info(ticker: str, fetch_live: bool = True) -> str:
    """Get detailed symbol information"""
    try:
        # Check database first
        cached = get_symbol_from_db(ticker)
        
        if cached and not fetch_live:
            return json.dumps({
                "success": True,
                "source": "database",
                **cached
            }, indent=2)
        
        # Fetch live data
        info = await fetch_yahoo_info(ticker)
        
        if not info:
            if cached:
                return json.dumps({
                    "success": True,
                    "source": "database_fallback",
                    "note": "Live fetch failed, returning cached data",
                    **cached
                }, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"Ticker {ticker} not found"
                })
        
        # ALWAYS save to database after successful fetch
        try:
            save_symbol_mapping(info)
        except Exception as save_error:
            print(f"Warning: Failed to save symbol mapping: {save_error}")
        
        return json.dumps({
            "success": True,
            "source": "yahoo_finance",
            **info
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@symbol_exchange_server.tool(
    name="resolve_isin",
    description="""Find all ticker symbols for a given ISIN across different exchanges.
    
Args:
    isin: str
        The ISIN code (e.g., "US0378331005" for Apple)
        
Returns:
    JSON with list of tickers for the ISIN across all exchanges
""")
async def resolve_isin(isin: str) -> str:
    """Find all tickers for a given ISIN"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ticker, exchange, company_name, country, currency, primary_listing
            FROM symbol_mappings
            WHERE isin = ?
            ORDER BY primary_listing DESC, last_updated DESC
        """, (isin,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            # Database doesn't have this ISIN - search the web
            country_code = isin[:2]
            print(f"No cached data for ISIN {isin}, searching web for tickers...")
            
            # Use web search to find tickers for this ISIN
            web_tickers = await search_ticker_by_isin(isin)
            
            if web_tickers:
                # Save to database
                for ticker_info in web_tickers:
                    save_symbol_mapping(ticker_info)
                
                # Format and return results
                tickers = []
                for info in web_tickers:
                    tickers.append({
                        "ticker": info['ticker'],
                        "exchange": info.get('exchange'),
                        "company_name": info.get('company_name'),
                        "country": info.get('country'),
                        "currency": info.get('currency'),
                        "primary_listing": False  # Can't determine from web search
                    })
                
                return json.dumps({
                    "success": True,
                    "isin": isin,
                    "ticker_count": len(tickers),
                    "tickers": tickers,
                    "source": "web_search"
                }, indent=2)
            else:
                return json.dumps({
                    "success": False,
                    "error": f"No tickers found for ISIN: {isin}",
                    "note": "Unable to find this ISIN in web search. The security may be delisted or the ISIN may be incorrect.",
                    "country_code": country_code
                })
        
        tickers = []
        for row in rows:
            tickers.append({
                "ticker": row[0],
                "exchange": row[1],
                "company_name": row[2],
                "country": row[3],
                "currency": row[4],
                "primary_listing": bool(row[5])
            })
        
        return json.dumps({
            "success": True,
            "isin": isin,
            "ticker_count": len(tickers),
            "tickers": tickers
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@symbol_exchange_server.tool(
    name="validate_ticker",
    description="""Validate if a ticker symbol is valid and tradeable.
    
Args:
    ticker: str
        The ticker symbol to validate
        
Returns:
    JSON with validation status and basic ticker info
""")
async def validate_ticker(ticker: str) -> str:
    """Validate ticker symbol"""
    try:
        info = await fetch_yahoo_info(ticker)
        
        if info:
            return json.dumps({
                "success": True,
                "valid": True,
                "ticker": ticker,
                "company_name": info.get('company_name'),
                "exchange": info.get('exchange'),
                "isin": info.get('isin')
            }, indent=2)
        else:
            return json.dumps({
                "success": True,
                "valid": False,
                "ticker": ticker,
                "message": "Ticker not found or not tradeable"
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@symbol_exchange_server.tool(
    name="get_exchange_info",
    description="""Get information about supported exchanges.
    
Args:
    exchange_code: str (optional)
        Specific exchange code (e.g., "NASDAQ", "NSE", "BSE")
        If not provided, returns all supported exchanges
        
Returns:
    JSON with exchange information
""")
async def get_exchange_info(exchange_code: Optional[str] = None) -> str:
    """Get exchange information"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        if exchange_code:
            cursor.execute("SELECT * FROM exchange_info WHERE exchange_code = ?", (exchange_code,))
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return json.dumps({
                    "success": False,
                    "error": f"Exchange {exchange_code} not found"
                })
            
            return json.dumps({
                "success": True,
                "exchange_code": row[0],
                "exchange_name": row[1],
                "country": row[2],
                "timezone": row[3],
                "currency": row[4],
                "suffix": row[5],
                "supported": bool(row[6])
            }, indent=2)
        else:
            cursor.execute("SELECT * FROM exchange_info WHERE supported = 1")
            rows = cursor.fetchall()
            conn.close()
            
            exchanges = []
            for row in rows:
                exchanges.append({
                    "exchange_code": row[0],
                    "exchange_name": row[1],
                    "country": row[2],
                    "timezone": row[3],
                    "currency": row[4],
                    "suffix": row[5]
                })
            
            return json.dumps({
                "success": True,
                "exchange_count": len(exchanges),
                "exchanges": exchanges
            }, indent=2)
            
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@symbol_exchange_server.tool(
    name="map_cross_listings",
    description="""Find all cross-listings of a company across different exchanges.
    
Args:
    ticker: str (optional)
        A known ticker symbol for the company
    company_name: str (optional)
        The company name to search for
    isin: str (optional)
        The ISIN to find all listings for
        
Returns:
    JSON with list of all tickers for the same company across exchanges
""")
async def map_cross_listings(ticker: Optional[str] = None, company_name: Optional[str] = None, isin: Optional[str] = None) -> str:
    """Map cross-listings of the same company across exchanges"""
    try:
        listings = []
        
        # If ISIN provided, use that directly
        if isin:
            print(f"Searching cross-listings by ISIN: {isin}")
            # Check database first
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT ticker, exchange, company_name, country, currency
                FROM symbol_mappings WHERE isin = ?
            """, (isin,))
            rows = cursor.fetchall()
            conn.close()
            
            for row in rows:
                listings.append({
                    'ticker': row[0],
                    'exchange': row[1],
                    'company_name': row[2],
                    'country': row[3],
                    'currency': row[4],
                    'source': 'database'
                })
            
            # Also search web for additional listings
            web_results = await search_ticker_by_isin(isin)
            for info in web_results:
                if info['ticker'] not in [l['ticker'] for l in listings]:
                    listings.append({
                        'ticker': info['ticker'],
                        'exchange': info.get('exchange'),
                        'company_name': info.get('company_name'),
                        'country': info.get('country'),
                        'currency': info.get('currency'),
                        'source': 'web_search'
                    })
                    save_symbol_mapping(info)
        
        # If ticker provided, get its info first
        elif ticker:
            print(f"Searching cross-listings for ticker: {ticker}")
            info = await fetch_yahoo_info(ticker)
            if not info:
                return json.dumps({
                    "success": False,
                    "error": f"Ticker {ticker} not found"
                })
            
            isin = info.get('isin')
            company_name = info.get('company_name')
            
            # Add this ticker to results
            listings.append({
                'ticker': info['ticker'],
                'exchange': info.get('exchange'),
                'company_name': company_name,
                'country': info.get('country'),
                'currency': info.get('currency'),
                'source': 'primary'
            })
            
            # Search for cross-listings
            if isin:
                cross_results = await search_ticker_by_isin(isin)
            else:
                cross_results = await search_cross_listings(company_name, ticker)
            
            for cross_info in cross_results:
                if cross_info['ticker'] != ticker:
                    listings.append({
                        'ticker': cross_info['ticker'],
                        'exchange': cross_info.get('exchange'),
                        'company_name': cross_info.get('company_name'),
                        'country': cross_info.get('country'),
                        'currency': cross_info.get('currency'),
                        'source': 'web_search'
                    })
                    save_symbol_mapping(cross_info)
        
        # If company name provided, search for it
        elif company_name:
            print(f"Searching cross-listings for company: {company_name}")
            cross_results = await search_cross_listings(company_name)
            
            for info in cross_results:
                listings.append({
                    'ticker': info['ticker'],
                    'exchange': info.get('exchange'),
                    'company_name': info.get('company_name'),
                    'country': info.get('country'),
                    'currency': info.get('currency'),
                    'source': 'web_search'
                })
                save_symbol_mapping(info)
        else:
            return json.dumps({
                "success": False,
                "error": "Must provide at least one of: ticker, company_name, or isin"
            })
        
        if not listings:
            return json.dumps({
                "success": False,
                "error": "No cross-listings found"
            })
        
        return json.dumps({
            "success": True,
            "listing_count": len(listings),
            "listings": listings
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@symbol_exchange_server.tool(
    name="standardize_symbol",
    description="""Convert a symbol to preferred exchange format based on configuration.
    
Args:
    ticker: str
        The ticker symbol to standardize
    target_exchange: str (optional)
        Target exchange to convert to (e.g., "NSE", "BSE")
        If not provided, uses primary listing
        
Returns:
    JSON with standardized ticker symbol
""")
async def standardize_symbol(ticker: str, target_exchange: Optional[str] = None) -> str:
    """Standardize symbol to preferred exchange"""
    try:
        # Get symbol info
        info = await fetch_yahoo_info(ticker)
        
        if not info:
            return json.dumps({
                "success": False,
                "error": f"Ticker {ticker} not found"
            })
        
        isin = info.get('isin')
        
        if not isin:
            return json.dumps({
                "success": True,
                "original_ticker": ticker,
                "standardized_ticker": ticker,
                "note": "No ISIN available, using original ticker"
            }, indent=2)
        
        # If target exchange specified, find ticker for that exchange
        if target_exchange:
            # Check database first
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT ticker FROM symbol_mappings
                WHERE isin = ? AND exchange = ?
                LIMIT 1
            """, (isin, target_exchange))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return json.dumps({
                    "success": True,
                    "original_ticker": ticker,
                    "standardized_ticker": row[0],
                    "exchange": target_exchange,
                    "isin": isin,
                    "source": "database"
                }, indent=2)
            
            # Not in database - search web for ticker on target exchange
            if isin:
                print(f"Searching web for {isin} on {target_exchange}")
                web_tickers = await search_ticker_by_isin(isin)
                
                for ticker_info in web_tickers:
                    if ticker_info.get('exchange') == target_exchange:
                        save_symbol_mapping(ticker_info)
                        return json.dumps({
                            "success": True,
                            "original_ticker": ticker,
                            "standardized_ticker": ticker_info['ticker'],
                            "exchange": target_exchange,
                            "isin": isin,
                            "source": "web_search"
                        }, indent=2)
            
            # Try searching by company name and exchange
            company_name = info.get('company_name')
            if company_name:
                print(f"Searching for {company_name} on {target_exchange}")
                search_query = f"{company_name} {target_exchange}"
                search_results = await search_ticker_with_tavily(search_query, search_type="company")
                
                for item in search_results:
                    found_ticker = item['ticker']
                    # Verify it's on the target exchange
                    found_info = await fetch_yahoo_info(found_ticker)
                    if found_info and found_info.get('exchange') == target_exchange:
                        if found_info.get('isin') == isin:  # Confirm same company
                            save_symbol_mapping(found_info)
                            return json.dumps({
                                "success": True,
                                "original_ticker": ticker,
                                "standardized_ticker": found_ticker,
                                "exchange": target_exchange,
                                "isin": isin,
                                "source": "web_search"
                            }, indent=2)
            
            return json.dumps({
                "success": False,
                "error": f"Could not find ticker for {ticker} on {target_exchange}",
                "original_ticker": ticker,
                "isin": isin
            })
        
        # Use primary listing
        return json.dumps({
            "success": True,
            "original_ticker": ticker,
            "standardized_ticker": ticker,
            "exchange": info.get('exchange'),
            "isin": isin,
            "note": "Using primary listing"
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    print("Starting Symbol & Exchange Management MCP server on port 8568...")
    symbol_exchange_server.run(transport="streamable-http", port=8568)
