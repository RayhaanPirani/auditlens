
import os
import sys
import json
import logging

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from services.vector_store import get_vector_store

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TAX_LAWS_FILE = "data/tax_laws_initial.json"
COLLECTION_NAME = "tax_laws"

def ingest_tax_laws():
    """Read tax laws JSON and ingest into ChromaDB."""
    
    if not os.path.exists(TAX_LAWS_FILE):
        logger.error(f"Tax laws file not found: {TAX_LAWS_FILE}")
        return
    
    try:
        with open(TAX_LAWS_FILE, "r") as f:
            tax_laws = json.load(f)
            
        logger.info(f"Loaded {len(tax_laws)} tax law entries")
        
        vs = get_vector_store()
        
        for law in tax_laws:
            law_id = law.get("id")
            title = law.get("title")
            content = law.get("content")
            
            # Format content for embedding
            markdown_content = f"# {title}\n\n**Jurisdiction:** {law.get('jurisdiction')}\n**Type:** {law.get('type')}\n\n{content}"
            
            # Store
            success = vs.store_document(
                file_hash=law_id,
                original_filename=f"tax_law_{law_id}",  # Pseudo-filename
                document_type="Tax Law",
                markdown_content=markdown_content,
                extracted_fields={
                    "title": title,
                    "jurisdiction": law.get("jurisdiction"),
                    "type": law.get("type")
                },
                raw_chunks=[],
                overall_confidence=1.0,
                needs_verification=False,
                collection_name=COLLECTION_NAME
            )
            
            if success:
                logger.info(f"✅ Ingested: {title}")
            else:
                logger.error(f"❌ Failed to ingest: {title}")
                
        logger.info("Ingestion complete!")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")

if __name__ == "__main__":
    ingest_tax_laws()
