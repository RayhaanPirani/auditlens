
import asyncio
import os
import sys

sys.path.append(os.getcwd())

from services.tax_agent import tool_research_tax_query, run_agent

async def test_unified_search():
    print("=== Testing Unified RAG Search ===")
    
    # 1. Test Direct Tool Usage
    print("\n\n1. Testing 'research_tax_query' tool directly:")
    query = "What rules apply to my tuition and is there a limit?"
    print(f"Query: {query}")
    print("-" * 40)
    result = tool_research_tax_query(query)
    print(result)
    print("-" * 40)
    
    # 2. Test Agent Decision Making
    print("\n\n2. Testing Agent Tool Selection:")
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping agent test (no API key)")
        return

    queries = [
        "Can I deduct the moving expenses I uploaded?",
        "What are the 2025 tax brackets?"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        try:
            res = await run_agent(q)
            print(f"Response: {res[:300]}...")  # Show beginning of response
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_unified_search())
