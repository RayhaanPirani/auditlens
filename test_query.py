#!/usr/bin/env python3
"""
Test script for the vector store query function.
Demonstrates the modular query_tax_documents function that will be converted to a LangChain tool.
"""

import sys
sys.path.insert(0, '/Users/rayhaan/auditlens')

from services.vector_store import query_tax_documents, get_document_summary
import json

def print_separator():
    print("\n" + "="*80 + "\n")

def main():
    print("🔍 VECTOR STORE QUERY TEST")
    print_separator()
    
    # 1. First, get a summary of what's in the store
    print("📊 DOCUMENT SUMMARY")
    print("-" * 40)
    summary = get_document_summary()
    
    if summary["success"]:
        print(f"Total documents in store: {summary['total_documents']}")
        for doc in summary["documents"]:
            print(f"  • {doc['document_type']} ({doc['filename']}) - {doc['confidence']:.0%} confidence")
    else:
        print(f"Error: {summary.get('error', 'Unknown error')}")
    
    print_separator()
    
    # 2. Test queries
    test_queries = [
        "employment income from T4",
        "tuition fees paid for education",
        "CPP contributions and deductions",
        "University of Windsor",
    ]
    
    for query in test_queries:
        print(f"🔎 QUERY: \"{query}\"")
        print("-" * 40)
        
        result = query_tax_documents(query, top_k=3, include_fields=True)
        
        if result["success"]:
            print(f"Found {result['num_results']} results:\n")
            
            for doc in result["results"]:
                print(f"  #{doc['rank']} {doc['document_type']}")
                print(f"     Similarity Score: {doc['similarity_score']:.2%}")
                print(f"     Distance: {doc['distance']}")
                print(f"     Confidence: {doc['confidence']:.0%}")
                print(f"     File: {doc['filename']}")
                
                # Show extracted fields
                if doc.get("extracted_fields"):
                    print("     Extracted Fields:")
                    for field, value in doc["extracted_fields"].items():
                        print(f"       • {field}: {value}")
                
                print()
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")
        
        print_separator()
    
    # 3. Show the raw return format for LangChain integration
    print("📦 RAW RETURN FORMAT (for LangChain tool)")
    print("-" * 40)
    result = query_tax_documents("income tax deducted", top_k=2)
    print(json.dumps(result, indent=2, default=str))
    
    print_separator()
    print("✅ Test complete!")
    print("\nThis query_tax_documents() function can be easily converted to a LangChain tool:")
    print("""
    from langchain.tools import tool
    
    @tool
    def query_tax_documents(query: str, top_k: int = 5) -> dict:
        '''Query the tax document knowledge base for relevant documents.'''
        from services.vector_store import query_tax_documents as query_fn
        return query_fn(query, top_k=top_k)
    """)

if __name__ == "__main__":
    main()
