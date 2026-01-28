
import asyncio
import os
import sys

sys.path.append(os.getcwd())

from services.tax_agent import tool_query_documents, tool_query_tax_law, run_agent

async def test_multi_index():
    print("=== Testing Multi-Index RAG ===")
    
    # 1. Test Tax Law Query (Should hit 'tax_laws' collection)
    print("\n1. Testing Tax Law Query (General):")
    law_result = tool_query_tax_law("What are the 2025 federal tax brackets?")
    print(law_result[:300] + "...")
    
    # 2. Test Parsed Document Query (Should hit 'tax_documents' collection)
    # Note: Might be empty if no user docs are uploaded
    print("\n2. Testing User Document Query (Personal):")
    doc_result = tool_query_documents("tuition") 
    print(doc_result[:300] + "...")
    
    # 3. Test Agent Routing
    print("\n3. Testing Agent Routing:")
    if not os.getenv("OPENAI_API_KEY"):
        print("Skipping agent test (no API key)")
        return

    queries = [
        "What is the RRSP contribution deadline?",
        "Do I have any T4 forms uploaded?"
    ]
    
    for q in queries:
        print(f"\nQuery: {q}")
        try:
            res = await run_agent(q)
            print(f"Response: {res[:200]}...")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_multi_index())
