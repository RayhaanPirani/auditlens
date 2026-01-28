
import asyncio
import os
import sys

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from services.tax_agent import (
    tool_calculate_totals,
    tool_estimate_tax,
    tool_get_summary,
    tool_tax_tips,
    run_agent
)

async def test_tools():
    print("=== Testing Direct Tools ===")
    
    # 1. Test Tax Tips
    print("\n1. Testing Tax Tips (RRSP):")
    tips = tool_tax_tips("rrsp")
    print(tips[:100] + "...")
    
    # 2. Test Estimate Tax (Direct Input)
    print("\n2. Testing Estimate Tax (Manual Input $100k):")
    estimate = tool_estimate_tax("100000")
    print(estimate)

    # 3. Test Totals (might be empty if no docs)
    print("\n3. Testing Totals:")
    totals = tool_calculate_totals()
    print(totals)

async def test_agent_interaction():
    print("\n=== Testing Agent Interaction (LLM) ===")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping agent test - no API key")
        return

    query = "How much tax would I pay on $85,000 income in Ontario?"
    print(f"\nQuery: {query}")
    
    try:
        response = await run_agent(query)
        print(f"Response:\n{response}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_tools())
    asyncio.run(test_agent_interaction())
