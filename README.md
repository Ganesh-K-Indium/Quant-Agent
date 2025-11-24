# Quant Agent

A collection of tools and servers for quantitative stock analysis, trading indicators, and financial data integration using the Model Context Protocol (MCP).

## Features

- Stock analysis with technical indicators
- Yahoo Finance data integration via MCP server
- LangGraph-based agent for stock exchange operations
- FastAPI-based API server for interactions

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ganesh-K-Indium/Quant-Agent.git
   cd Quant-Agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `Stock_Analysis/`: Core stock analysis tools and MCP server
- `yahoo-finance-mcp/`: MCP server for Yahoo Finance integration
- `stock_agent/`: LangGraph-based agent system for stock operations
  - `stock_exchange_agent/`: Sub-agents for information, technical analysis, and ticker finding

## Usage

Run the individual servers or agents as needed. Refer to each subfolder's README for specific instructions.

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

The `yahoo-finance-mcp` component is licensed under the MIT License - see the [LICENSE](yahoo-finance-mcp/LICENSE) file for details. Other components may have different licenses.