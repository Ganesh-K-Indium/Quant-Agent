"""
Research Agent - LangGraph Implementation
Handles stock research, analyst ratings, sentiment analysis, and scenario generation
using MCP tools via LangGraph React Agent
"""
import asyncio
import aiohttp
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from dotenv import load_dotenv
import os

load_dotenv()


async def wait_for_server(url: str, timeout: int = 10):
    """Wait until the MCP server is ready to accept connections."""
    import time
    import socket
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    host = parsed.hostname or 'localhost'
    port = parsed.port
    
    start = time.time()
    while time.time() - start < timeout:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            if result == 0:
                print(f"✅ Research MCP server is up at {url}")
                return True
        except:
            pass
        await asyncio.sleep(1)
    raise TimeoutError(f"Research MCP server at {url} did not respond within {timeout} seconds")


async def create_research_agent(checkpointer=None):
    """Create the Research sub-agent with all MCP tools."""
    system_prompt = """You are a research agent. Gather analyst ratings, sentiment, and generate investment scenarios.

**AVAILABLE TOOLS:**
- web_search: General web search (args: query, max_results, search_depth)
- search_analyst_ratings: Find analyst ratings (args: symbol, company_name, days_back)
- aggregate_ratings: Normalize and aggregate ratings (args: symbol, search_results)
- analyze_sentiment: Analyze text sentiment (args: text, symbol)
- summarize_content: Summarize articles (args: content, symbol, max_length, focus)
- generate_scenarios: Create bull/bear scenarios (args: symbol, company_name, ratings_data, news_summary)
- get_cached_research: Get cached data (args: symbol, data_type)
- comprehensive_research: Full research pipeline (args: symbol, company_name, current_price, include_scenarios)

**CRITICAL RULES:**
1. Only provide information from tool responses. Do NOT invent analyst ratings, price targets, or scenarios.
2. If tools return errors or no data, report that clearly - do NOT make up data.
3. When using generate_scenarios, provide actual data from previous tool calls.
4. For comprehensive analysis, use comprehensive_research tool.
5. Always specify the stock symbol when calling tools.

**PARAMETER REQUIREMENTS:**
- web_search: query is required
- search_analyst_ratings: symbol is required
- summarize_content: content and symbol are required, focus options: "ratings", "news", "analysis", "general"
- generate_scenarios: symbol is required

**RESPONSE FORMAT:**
After gathering data, present:
1. **Summary**: Key findings
2. **Analyst Ratings**: Consensus and notable opinions (only from tool data)
3. **Sentiment**: Based on analyze_sentiment results
4. **Scenarios**: Bull/bear cases (only from generate_scenarios tool)
5. **Key Risks**: What to watch

**EXAMPLES:**
User: "Research AAPL"
You: [Call comprehensive_research with symbol="AAPL", then present the results]

User: "What do analysts think about Tesla?"
You: [Call search_analyst_ratings with symbol="TSLA", then present the ratings found]

Do NOT provide specific price targets, ratings, or percentages unless they come directly from tool responses."""
    
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    MCP_HTTP_STREAM_URL = "http://localhost:8567/mcp"  # Research MCP server
    
    # Keep the client and session open for the lifetime of the agent
    client = streamablehttp_client(MCP_HTTP_STREAM_URL)
    read_stream, write_stream, _ = await client.__aenter__()
    session = ClientSession(read_stream, write_stream)
    await session.__aenter__()
    await session.initialize()
    tools = await load_mcp_tools(session)
    
    agent = create_react_agent(
        model=model,
        tools=tools,
        name="research_agent",
        prompt=system_prompt,
        checkpointer=checkpointer
    )
    
    # Attach the session and client to the agent to keep them alive
    agent._mcp_session = session
    agent._mcp_client = client
    
    return agent
