
import os
import sys

sys.path.append(os.getcwd())

from services.vector_store import get_vector_store

def test_view():
    print("Testing get_all_documents...")
    vs = get_vector_store()
    docs = vs.get_all_documents(collection_name="tax_documents")
    print(f"Count: {len(docs)}")
    for d in docs:
        print(f"- {d['original_filename']} ({d['file_hash']})")

if __name__ == "__main__":
    test_view()
