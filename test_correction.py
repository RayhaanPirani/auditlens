
import logging
from services.vector_store import get_vector_store
from services.tax_agent import tool_update_field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_correction_persistence():
    print("=== Testing Correction Persistence ===")
    
    vs = get_vector_store()
    
    # 1. Store a mock document
    file_hash = "test_hash_123"
    filename = "correction_test_doc.pdf"
    
    mock_fields = {
        "income": {
            "value": "1000",
            "confidence": 0.5,
            "field_name": "income",
            "manually_corrected": False
        }
    }
    
    print(f"1. Storing initial document: {filename} (Income: $1000)")
    vs.store_document(
        file_hash=file_hash,
        original_filename=filename,
        document_type="Test Doc",
        markdown_content="Test content",
        extracted_fields=mock_fields,
        raw_chunks=[],
        overall_confidence=0.5,
        needs_verification=True
    )
    
    # Verify initial state
    doc = vs.get_document(file_hash)
    print(f"   Initial fetch: Income = {doc['extracted_fields']['income']['value']}")
    assert doc['extracted_fields']['income']['value'] == "1000"
    
    # 2. Perform Correction using the Tool
    print("\n2. Calling tool_update_field('correction_test:income:50000')")
    # Note: testing fuzzy matching "correction_test" should match "correction_test_doc.pdf"
    result = tool_update_field("correction_test:income:50000")
    print(f"   Tool Output: {result}")
    
    # 3. Verify Persistence
    print("\n3. Verifying persistence in Vector Store...")
    updated_doc = vs.get_document(file_hash)
    new_data = updated_doc['extracted_fields']['income']
    
    print(f"   New Value: {new_data['value']}")
    print(f"   Manually Corrected: {new_data.get('manually_corrected')}")
    print(f"   Confidence: {new_data.get('confidence')}")
    print(f"   Needs Verification: {updated_doc.get('needs_verification')}")
    
    if (new_data['value'] == "50000" and 
        new_data.get('manually_corrected') is True and 
        updated_doc.get('needs_verification') is False):
        print("\n✅ TEST PASSED: Update persisted correctly!")
    else:
        print("\n❌ TEST FAILED: Data mismatch.")
        
    # Cleanup
    vs.delete_document(file_hash)

if __name__ == "__main__":
    test_correction_persistence()
