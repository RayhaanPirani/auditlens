#!/usr/bin/env python3
"""
Test script to diagnose LandingAI document parsing issues.
"""

import sys
import logging

# Enable verbose logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

# Add current directory to path
sys.path.insert(0, '/Users/rayhaan/auditlens')

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def test_parsing():
    """Test the document parsing pipeline."""
    
    # Find a test file
    uploads_dir = Path("./uploads")
    test_files = list(uploads_dir.glob("*"))
    
    if not test_files:
        print("No files found in uploads directory!")
        return
    
    test_file = test_files[0]
    print(f"\n=== Testing with file: {test_file} ===\n")
    
    # Try direct LandingAI parsing first
    print("--- Step 1: Direct LandingAI API call ---")
    try:
        from agentic_doc.parse import parse
        from pydantic import BaseModel, Field
        from typing import Optional
        
        class TestSchema(BaseModel):
            document_title: Optional[str] = Field(None, description="Document title")
            tax_year: Optional[str] = Field(None, description="Tax year")
            total_income: Optional[str] = Field(None, description="Total income")
        
        print(f"Calling parse({test_file}, extraction_model=TestSchema)")
        results = parse(str(test_file), extraction_model=TestSchema)
        
        print(f"\nResults type: {type(results)}")
        print(f"Results length: {len(results) if results else 'None'}")
        
        if results and len(results) > 0:
            result = results[0]
            print(f"\nFirst result type: {type(result)}")
            print(f"Result attributes: {dir(result)}")
            
            # Print markdown
            if hasattr(result, 'markdown'):
                print(f"\nMarkdown length: {len(result.markdown)} chars")
                print(f"Markdown preview: {result.markdown[:500] if result.markdown else 'None'}...")
            
            # Print chunks
            if hasattr(result, 'chunks'):
                print(f"\nChunks count: {len(result.chunks) if result.chunks else 0}")
            
            # Print extraction
            if hasattr(result, 'extraction'):
                print(f"\nExtraction type: {type(result.extraction)}")
                print(f"Extraction: {result.extraction}")
                
                if result.extraction:
                    if hasattr(result.extraction, 'model_dump'):
                        print(f"Extraction fields: {result.extraction.model_dump()}")
                    elif hasattr(result.extraction, '__dict__'):
                        print(f"Extraction __dict__: {result.extraction.__dict__}")
            
            # Print extraction metadata
            if hasattr(result, 'extraction_metadata'):
                print(f"\nExtraction metadata: {result.extraction_metadata}")
        
    except Exception as e:
        print(f"\nERROR in direct API call: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    # Try our wrapper
    print("\n\n--- Step 2: Testing parse_document_sync wrapper ---")
    try:
        from services.document_parser import parse_document_sync
        
        result = parse_document_sync(str(test_file), test_file.name)
        
        print(f"\nParsedDocument result:")
        print(f"  file_hash: {result.file_hash}")
        print(f"  document_type: {result.document_type}")
        print(f"  overall_confidence: {result.overall_confidence}")
        print(f"  needs_verification: {result.needs_verification}")
        print(f"  extracted_fields count: {len(result.extracted_fields)}")
        print(f"  markdown length: {len(result.markdown_content)} chars")
        
        if result.extracted_fields:
            print("\nExtracted fields:")
            for name, field in result.extracted_fields.items():
                print(f"  {name}: {field.value} (confidence: {field.confidence})")
        
    except Exception as e:
        print(f"\nERROR in wrapper: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_parsing()
