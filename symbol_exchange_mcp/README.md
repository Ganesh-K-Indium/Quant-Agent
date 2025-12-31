# Symbol & Exchange Management MCP Server

Comprehensive symbol and exchange management service for the Quant Agent system.

## Features

### 🔍 **Ticker Lookup**
- Convert company names to ticker symbols
- Support for multiple exchanges (NASDAQ, NYSE, NSE, BSE, LSE, HKEX, etc.)
- Smart search with country and exchange filtering

### 🌐 **ISIN Resolution**
- Resolve ISIN to ticker symbols across all exchanges
- Identify cross-listings (same company on different exchanges)
- Track primary listing information

### ✅ **Symbol Validation**
- Validate ticker symbols before use
- Quick validation checks
- Return basic info for valid tickers

### 🏢 **Exchange Information**
- Comprehensive exchange database
- Exchange codes, suffixes, timezones, currencies
- Support for 11+ major global exchanges

### 🔄 **Symbol Standardization**
- Convert tickers to preferred exchange format
- Handle NSE/BSE conversions for Indian stocks
- ISIN-based mapping for consistency

### 💾 **Persistent Mapping**
- SQLite database for symbol mappings
- Cache ticker info for performance
- Track ISIN, exchange, and metadata

## Supported Exchanges

| Exchange | Code | Country | Suffix | Currency |
|----------|------|---------|--------|----------|
| NASDAQ | NASDAQ | US | (none) | USD |
| NYSE | NYSE | US | (none) | USD |
| NSE | NSE | India | .NS | INR |
| BSE | BSE | India | .BO | INR |
| LSE | LSE | UK | .L | GBP |
| HKEX | HKEX | Hong Kong | .HK | HKD |
| TSE | TSE | Japan | .T | JPY |
| SSE | SSE | China | .SS | CNY |
| SZSE | SZSE | China | .SZ | CNY |
| TSX | TSX | Canada | .TO | CAD |
| ASX | ASX | Australia | .AX | AUD |

## API Tools

### 1. `find_ticker`
Find ticker symbol from company name.

**Parameters:**
- `company_name` (str): Company name to search
- `preferred_exchange` (str, optional): Exchange code (e.g., "NSE", "NASDAQ")
- `country` (str, optional): Country code filter (e.g., "US", "IN")

**Returns:** JSON with ticker, ISIN, exchange, company info

**Example:**
```json
{
  "success": true,
  "ticker": "RELIANCE.NS",
  "isin": "INE002A01018",
  "exchange": "NSI",
  "company_name": "Reliance Industries Limited",
  "country": "IN",
  "currency": "INR"
}
```

### 2. `get_symbol_info`
Get detailed information about a ticker.

**Parameters:**
- `ticker` (str): Ticker symbol
- `fetch_live` (bool, optional): Fetch fresh data (default: true)

**Returns:** Comprehensive symbol information

### 3. `resolve_isin`
Find all tickers for a given ISIN.

**Parameters:**
- `isin` (str): ISIN code

**Returns:** List of tickers across exchanges

**Example:**
```json
{
  "success": true,
  "isin": "US0378331005",
  "ticker_count": 1,
  "tickers": [
    {
      "ticker": "AAPL",
      "exchange": "NASDAQ",
      "company_name": "Apple Inc.",
      "country": "US",
      "primary_listing": true
    }
  ]
}
```

### 4. `validate_ticker`
Check if a ticker is valid.

**Parameters:**
- `ticker` (str): Ticker to validate

**Returns:** Validation status with basic info

### 5. `get_exchange_info`
Get exchange information.

**Parameters:**
- `exchange_code` (str, optional): Specific exchange or all if omitted

**Returns:** Exchange metadata

### 6. `standardize_symbol`
Convert symbol to preferred exchange format.

**Parameters:**
- `ticker` (str): Ticker to standardize
- `target_exchange` (str, optional): Target exchange code

**Returns:** Standardized ticker

## Setup

### Install Dependencies
```bash
pip install fastmcp yfinance
```

### Run Server
```bash
python server.py
```

Server will start on `http://localhost:8569/mcp`

## Usage in LangGraph Agent

The symbol_exchange_agent automatically connects to this MCP server and exposes all tools.

```python
from stock_exchange_agent.subagents.symbol_exchange_agent import create_symbol_exchange_agent

agent = await create_symbol_exchange_agent()
```

## Database Schema

### `symbol_mappings` Table
Stores ticker to ISIN mappings.

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| isin | TEXT | ISIN code |
| ticker | TEXT | Ticker symbol |
| exchange | TEXT | Exchange code |
| company_name | TEXT | Company name |
| country | TEXT | Country code |
| currency | TEXT | Currency code |
| instrument_type | TEXT | Equity, ETF, etc. |
| primary_listing | BOOLEAN | Primary listing flag |
| source | TEXT | Data source |
| last_updated | TIMESTAMP | Last update time |

### `exchange_info` Table
Exchange metadata.

| Column | Type | Description |
|--------|------|-------------|
| exchange_code | TEXT | Exchange code (PK) |
| exchange_name | TEXT | Full name |
| country | TEXT | Country |
| timezone | TEXT | Timezone |
| currency | TEXT | Currency |
| suffix | TEXT | Yahoo Finance suffix |
| supported | BOOLEAN | Support flag |

## Best Practices

### For Indian Stocks
- **Default to NSE** (.NS suffix) for higher liquidity
- BSE alternative (.BO suffix) available via `standardize_symbol`
- Always specify exchange explicitly

### For US Stocks
- Use plain ticker (e.g., "AAPL", not "AAPL.US")
- No suffix needed

### For International Stocks
- Include exchange suffix (e.g., "0700.HK" for Tencent)
- Use `get_exchange_info` to find correct suffix

### ISIN Usage
- Always capture ISIN when available
- Use `resolve_isin` for cross-listing queries
- ISIN is the canonical identifier for cross-exchange mapping

## Integration with Main Agent

The symbol_exchange_agent is automatically integrated into the main supervisor and will be invoked for:
- Company name lookups
- Ticker validation
- Exchange queries
- Cross-listing detection
- Symbol standardization

## Error Handling

All tools return JSON with `success` field:
- `success: true` - Operation completed
- `success: false` - Error occurred, see `error` field

## Future Enhancements

- [ ] Add FIGI, CUSIP, SEDOL support
- [ ] Expand exchange coverage (50+ exchanges)
- [ ] Real-time ticker validation
- [ ] Company name fuzzy matching
- [ ] Multi-language company name support
- [ ] Historical ticker symbol changes
- [ ] Delisting detection
