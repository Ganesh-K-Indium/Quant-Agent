"""
Main LangGraph Supervisor Agent for Stock Analysis
---------------------------------------------------
Manages Stock Information, Technical Analysis, and Ticker Finder agents as specialized sub-agents.
Uses langgraph-supervisor to coordinate work between agents.
"""

import asyncio
import aiohttp
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.tools import load_mcp_tools

from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from dotenv import load_dotenv
from stock_exchange_agent.subagents.stock_information.langgraph_agent import create_stock_information_agent
from stock_exchange_agent.subagents.technical_analysis_agent.langgraph_agent import create_technical_analysis_agent
from stock_exchange_agent.subagents.symbol_exchange_agent.langgraph_agent import create_symbol_exchange_agent
from stock_exchange_agent.subagents.research_agent.langgraph_agent import create_research_agent
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import os
from datetime import datetime
import json

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
                print(f"✅ MCP server is up at {url}")
                return True
        except:
            pass
        await asyncio.sleep(1)
    raise TimeoutError(f"MCP server at {url} did not respond within {timeout} seconds")


async def main():
    """Main supervisor agent that coordinates stock analysis sub-agents."""
    
    print("🚀 Initializing Stock Analysis Supervisor Agent...")
    print("=" * 80)
    
    # Initialize memory saver
    print("💾 Initializing SQLite memory...")
    db_path = os.getenv("SQLITE_DB_PATH", "sqlite:///checkpoints.db")
    
    async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
        await saver.setup()  # Creates tables if needed
        print("✅ Memory initialized successfully")
        
        # Wait for MCP servers to be ready
        print("⏳ Waiting for MCP servers...")
        await wait_for_server("http://localhost:8565/mcp")  # Stock Information
        await wait_for_server("http://localhost:8566/mcp")  # Technical Analysis
        await wait_for_server("http://localhost:8567/mcp")  # Research
        await wait_for_server("http://localhost:8569/mcp")  # Symbol & Exchange
        
        # Create sub-agents
        print("🔧 Creating sub-agents...")
        stock_info_agent = await create_stock_information_agent(checkpointer=saver)
        technical_agent = await create_technical_analysis_agent(checkpointer=saver)
        symbol_exchange_agent = await create_symbol_exchange_agent(checkpointer=saver)
        research_agent = await create_research_agent(checkpointer=saver)
        
        print("✅ Sub-agents created successfully")
    
        supervisor_graph = create_supervisor(
            model=ChatOpenAI(temperature=0, model_name="gpt-4o"),
            agents=[stock_info_agent, technical_agent, symbol_exchange_agent, research_agent],
            prompt=(
                "You are HEAD OF EQUITY RESEARCH coordinating specialized financial analysts with institutional-grade precision.\n\n"
                
                "**AGENT CAPABILITIES:**\n"
                "1. symbol_exchange_agent: Ticker discovery, ISIN resolution, exchange mapping, cross-listing analysis, disambiguation\n"
                "2. stock_information_agent: Prices, financials, dividends, ownership, analyst targets, options\n"
                "3. technical_analysis_agent: Charts, indicators (SMA/RSI/MACD/Bollinger), volume, support/resistance\n"
                "4. research_agent: Analyst ratings, sentiment, bull/bear scenarios, industry analysis\n\n"
                
                "**ROUTING PRIORITIES:**\n"
                "• Ambiguous company name (e.g., 'Apple', 'Reliance') → symbol_exchange_agent FIRST to disambiguate\n"
                "• Clear ticker (AAPL, TSLA, RELIANCE.NS) → Skip to specialist agent directly\n"
                "• Cross-listing queries ('which exchanges?', 'NSE vs BSE') → symbol_exchange_agent\n"
                "• Price/financials/news → stock_information_agent\n"
                "• Charts/technical indicators → technical_analysis_agent\n"
                "• Research/ratings/sentiment → research_agent\n\n"
                
                "**KEY WORKFLOWS:**\n\n"
                "Ambiguous name: 'Apple price worldwide'\n"
                "  → symbol_exchange_agent (disambiguate + list exchanges) → WAIT user confirmation → stock_information_agent\n\n"
                
                "Clear ticker: 'AAPL price and RSI'\n"
                "  → stock_information_agent (price) → technical_analysis_agent (RSI) → Synthesize\n\n"
                
                "Exchange comparison: 'RELIANCE.NS vs .BO prices'\n"
                "  → symbol_exchange_agent (confirm same company) → stock_information_agent (both prices) → Compare\n\n"
                
                "**PROFESSIONAL STANDARDS:**\n"
                "1. NEVER fabricate data - only present agent responses\n"
                "2. ASK for clarification when ambiguous\n"
                "3. Confirm security identity before price/analysis\n"
                "4. Preserve ISIN context throughout conversation\n"
                "5. Delegate ONE task at a time, wait for completion\n"
                "6. Remember confirmed tickers, don't re-lookup\n"
                "7. Provide context: exchange, currency, timestamp\n"
                "8. Explain exchange differences when relevant\n\n"
                
                "**EXAMPLES:**\n"
                "❌ 'Apple price?' → stock_information_agent immediately\n"
                "✓ 'Apple price?' → symbol_exchange_agent first (AAPL vs APLE disambiguation)\n\n"
                
                "❌ Make up data: 'AAPL trading at $180'\n"
                "✓ Route to stock_information_agent, present exact response\n\n"
                
                "Ensure precision and professionalism in all coordination."
            ),
            add_handoff_back_messages=True,
            output_mode="full_history",
        )
        supervisor = supervisor_graph.compile(
            checkpointer=saver,
        )
        
        # Set recursion limit for the supervisor to prevent infinite loops
        supervisor.recursion_limit = 50
        
        print("\n" + "="*80)
        print("🤖 STOCK ANALYSIS SUPERVISOR AGENT - Ready for Commands")
        print("="*80)
        print("\n📋 What I can help you with:")
        print("\n📊 FUNDAMENTAL ANALYSIS:")
        print("  • Current stock prices and market data")
        print("  • Historical price charts and trends")
        print("  • Financial news and sentiment analysis")
        print("  • Dividends, stock splits, and corporate actions")
        print("  • Financial statements and company financials")
        print("  • Analyst recommendations and price targets")
        print("  • Holder information and institutional ownership")
        print("  • 5-year projections and growth estimates")
        
        print("\n📈 TECHNICAL ANALYSIS:")
        print("  • Moving averages (SMA, EMA)")
        print("  • RSI and momentum indicators")
        print("  • Bollinger Bands and volatility")
        print("  • MACD and trend analysis")
        print("  • Volume analysis")
        print("  • Support and resistance levels")
        print("  • Comprehensive technical charting")
        
        print("\n🔬 RESEARCH & SCENARIOS:")
        print("  • Web search for analyst ratings and news")
        print("  • Aggregated analyst consensus and price targets")
        print("  • Sentiment analysis of market commentary")
        print("  • Bull case scenarios with catalysts")
        print("  • Bear case scenarios with risks")
        print("  • Comprehensive investment research")
        print("  • Upgrades, downgrades, and rating changes")
        
        print("\n🔍 SYMBOL & EXCHANGE MANAGEMENT:")
        print("  • Find ticker symbols from company names")
        print("  • ISIN resolution and cross-listing detection")
        print("  • Exchange information (NSE, BSE, NASDAQ, NYSE, etc.)")
        print("  • Symbol validation and standardization")
        print("  • Support for US, Indian, and international stocks")
        
        print("\n🤖 INTELLIGENT FEATURES:")
        print("  • Automatic ticker resolution from company names")
        print("  • Context-aware conversation (remembers previous tickers)")
        print("  • Multi-part query handling (fundamentals + technicals + research)")
        print("  • Smart routing to specialized agents")
        
        print("\nEnter your command (or 'quit' to exit): ")
        
        while True:
            try:
                user_input = input("\n>>> ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print(f"\n🧠 Processing: {user_input}")
                print("-" * 50)
                
                # Get the current state to know how many messages exist
                current_state = await supervisor.aget_state(config={"configurable": {"thread_id": "main_thread"}})
                messages_before = len(current_state.values.get('messages', [])) if current_state.values else 0
                
                # Invoke supervisor with thread_id for memory persistence
                response = await supervisor.ainvoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config={"configurable": {"thread_id": "main_thread"}}
                )
                
                # Extract only NEW messages from this turn
                all_messages = response['messages']
                new_messages = all_messages[messages_before:] if messages_before > 0 else all_messages
                
                # Find the last AI message from the new messages that is not a transfer/handoff
                final_message = None
                for msg in reversed(new_messages):
                    if msg.type == 'ai' and msg.name != 'supervisor' and not msg.content.startswith('Transferring back') and not msg.content.startswith('Successfully transferred'):
                        final_message = msg
                        break
                
                # Fallback to last new message if no suitable AI message found
                if final_message is None and new_messages:
                    final_message = new_messages[-1]
                elif final_message is None:
                    final_message = all_messages[-1]
                
                print("\n🤖 Response:")
                print(final_message.content)

                def serialize_response(obj):
                    try:
                        if isinstance(obj, dict):
                            return {k: serialize_response(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [serialize_response(item) for item in obj]
                        elif isinstance(obj, (str, int, float, bool, type(None))):
                            return obj
                        elif hasattr(obj, 'dict') and callable(getattr(obj, 'dict', None)):
                            return obj.model_dump()
                        elif hasattr(obj, '__dict__'):
                            return serialize_response(obj.__dict__)
                        else:
                            return str(obj)
                    except Exception:
                        return str(obj)
                
                responses_dir = os.path.join(os.path.dirname(__file__), "responses")
                os.makedirs(responses_dir, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"response_{timestamp}.json"
                filepath = os.path.join(responses_dir, filename)
                with open(filepath, "w") as f:
                    json.dump(serialize_response(response), f, indent=4)
                print(f"📁 Response saved to {filepath}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        print("💾 Memory saved successfully")


if __name__ == "__main__":
    asyncio.run(main())
