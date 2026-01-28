"""
Tax Agent Service
LangChain-based agent for tax calculations, document queries, and optimization.
Uses GPT-4o-mini for cost efficiency.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate

from services.vector_store import get_vector_store

logger = logging.getLogger(__name__)


# Canadian Federal Tax Brackets 2025
FEDERAL_BRACKETS = [
    (55867, 0.15),      # 15% on first $55,867
    (55866, 0.205),     # 20.5% on next $55,866
    (61942, 0.26),      # 26% on next $61,942
    (62258, 0.29),      # 29% on next $62,258
    (float('inf'), 0.33) # 33% on remainder
]

# Provincial Tax Brackets 2025 (simplified)
PROVINCIAL_BRACKETS = {
    "ON": [  # Ontario
        (51446, 0.0505),
        (51446, 0.0915),
        (57144, 0.1116),
        (70000, 0.1216),
        (float('inf'), 0.1316)
    ],
    "BC": [  # British Columbia
        (47937, 0.0506),
        (47937, 0.077),
        (12045, 0.105),
        (23930, 0.1229),
        (47860, 0.147),
        (95720, 0.168),
        (float('inf'), 0.205)
    ],
    "AB": [  # Alberta (flat rate)
        (float('inf'), 0.10)
    ],
    "QC": [  # Quebec
        (51780, 0.14),
        (51780, 0.19),
        (19305, 0.24),
        (float('inf'), 0.2575)
    ],
    "DEFAULT": [  # Default estimate
        (float('inf'), 0.10)
    ]
}

BASIC_PERSONAL_AMOUNT = 15705  # Federal 2025


def calculate_bracket_tax(income: float, brackets: List[tuple]) -> float:
    """Calculate tax using progressive brackets."""
    tax = 0.0
    remaining = income
    
    for bracket_amount, rate in brackets:
        if remaining <= 0:
            break
        taxable = min(remaining, bracket_amount)
        tax += taxable * rate
        remaining -= taxable
    
    return tax


def get_province_from_documents() -> str:
    """Try to determine province from stored documents."""
    try:
        vs = get_vector_store()
        docs = vs.get_all_documents()
        
        for doc in docs:
            # Check employer address or institution
            fields = doc.get("extracted_fields", {})
            for field_name, field_data in fields.items():
                value = field_data.get("value", "") if isinstance(field_data, dict) else str(field_data)
                value_upper = value.upper()
                
                # Look for province codes
                if " ON " in value_upper or "ONTARIO" in value_upper:
                    return "ON"
                elif " BC " in value_upper or "BRITISH COLUMBIA" in value_upper:
                    return "BC"
                elif " AB " in value_upper or "ALBERTA" in value_upper:
                    return "AB"
                elif " QC " in value_upper or "QUEBEC" in value_upper:
                    return "QC"
        
        return "DEFAULT"
    except Exception as e:
        logger.warning(f"Could not determine province: {e}")
        return "DEFAULT"


# ============== AGENT TOOLS ==============

def tool_calculate_totals(query: str = "") -> str:
    """Calculate total income, deductions, and credits from all processed documents."""
    try:
        vs = get_vector_store()
        docs = vs.get_all_documents()
        
        if not docs:
            return "No documents have been processed yet. Please upload and process tax documents first."
        
        totals = {
            "employment_income": 0.0,
            "other_income": 0.0,
            "income_tax_deducted": 0.0,
            "cpp_contributions": 0.0,
            "ei_premiums": 0.0,
            "rrsp_contribution": 0.0,
            "tuition_amount": 0.0,
            "rent_amount": 0.0,
            "documents_processed": len(docs)
        }
        
        doc_types = []
        
        for doc in docs:
            doc_type = doc.get("document_type", "Unknown")
            doc_types.append(doc_type)
            
            fields = doc.get("extracted_fields", {})
            for field_name, field_data in fields.items():
                if isinstance(field_data, dict):
                    value_str = field_data.get("value", "0")
                else:
                    value_str = str(field_data)
                
                # Clean and parse numeric value
                try:
                    value = float(value_str.replace(",", "").replace("$", "").strip())
                except (ValueError, AttributeError):
                    continue
                
                # Map to totals
                if field_name in totals:
                    totals[field_name] += value
        
        total_income = totals["employment_income"] + totals["other_income"]
        
        result = f"""
📊 **Tax Document Totals**

**Documents Processed:** {totals['documents_processed']}
- Types: {', '.join(set(doc_types))}

**Income:**
- Employment Income: ${totals['employment_income']:,.2f}
- Other Income: ${totals['other_income']:,.2f}
- **Total Income: ${total_income:,.2f}**

**Amounts Withheld:**
- Income Tax Deducted: ${totals['income_tax_deducted']:,.2f}
- CPP Contributions: ${totals['cpp_contributions']:,.2f}
- EI Premiums: ${totals['ei_premiums']:,.2f}

**Credits & Deductions:**
- RRSP Contributions: ${totals['rrsp_contribution']:,.2f}
- Tuition Amount: ${totals['tuition_amount']:,.2f}
- Rent Paid: ${totals['rent_amount']:,.2f}
"""
        return result
        
    except Exception as e:
        logger.error(f"Error calculating totals: {e}")
        return f"Error calculating totals: {str(e)}"


def tool_estimate_tax(income_str: str = "") -> str:
    """Estimate federal and provincial tax based on income and detected province."""
    try:
        # Get income from parameter or calculate from documents
        if income_str and income_str.strip():
            try:
                income = float(income_str.replace(",", "").replace("$", "").strip())
            except ValueError:
                # Fall back to calculating from documents
                income = None
        else:
            income = None
        
        if income is None:
            # Calculate from documents
            vs = get_vector_store()
            docs = vs.get_all_documents()
            
            income = 0.0
            for doc in docs:
                fields = doc.get("extracted_fields", {})
                for field_name in ["employment_income", "other_income"]:
                    if field_name in fields:
                        field_data = fields[field_name]
                        value_str = field_data.get("value", "0") if isinstance(field_data, dict) else str(field_data)
                        try:
                            income += float(value_str.replace(",", "").replace("$", "").strip())
                        except (ValueError, AttributeError):
                            pass
        
        if income <= 0:
            return "No income found. Please provide an income amount or process income documents first."
        
        # Detect province
        province = get_province_from_documents()
        province_brackets = PROVINCIAL_BRACKETS.get(province, PROVINCIAL_BRACKETS["DEFAULT"])
        
        # Calculate taxes
        taxable_income = max(0, income - BASIC_PERSONAL_AMOUNT)
        federal_tax = calculate_bracket_tax(taxable_income, FEDERAL_BRACKETS)
        provincial_tax = calculate_bracket_tax(taxable_income, province_brackets)
        total_tax = federal_tax + provincial_tax
        
        effective_rate = (total_tax / income * 100) if income > 0 else 0
        marginal_rate = 0
        
        # Determine marginal rate
        remaining = taxable_income
        for bracket_amount, rate in FEDERAL_BRACKETS:
            if remaining <= bracket_amount:
                marginal_rate = rate
                break
            remaining -= bracket_amount
        
        for bracket_amount, rate in province_brackets:
            if taxable_income <= bracket_amount:
                marginal_rate += rate
                break
        
        province_names = {"ON": "Ontario", "BC": "British Columbia", "AB": "Alberta", "QC": "Quebec", "DEFAULT": "Estimated"}
        
        result = f"""
💰 **Tax Estimate for {province_names.get(province, province)}**

**Income:** ${income:,.2f}
**Basic Personal Amount:** ${BASIC_PERSONAL_AMOUNT:,.2f}
**Taxable Income:** ${taxable_income:,.2f}

**Estimated Tax:**
- Federal Tax: ${federal_tax:,.2f}
- Provincial Tax ({province}): ${provincial_tax:,.2f}
- **Total Tax: ${total_tax:,.2f}**

**Rates:**
- Effective Rate: {effective_rate:.1f}%
- Marginal Rate: {marginal_rate*100:.1f}%

⚠️ This is an estimate. Actual tax may vary based on credits and deductions.
"""
        return result
        
    except Exception as e:
        logger.error(f"Error estimating tax: {e}")
        return f"Error estimating tax: {str(e)}"


def tool_query_documents(query: str) -> str:
    """Search processed tax documents for specific information."""
    try:
        vs = get_vector_store()
        results = vs.search_documents(query, n_results=5)
        
        if not results:
            return "No matching documents found. Try a different search term."
        
        response_parts = [f"📄 **Found {len(results)} relevant document(s):**\n"]
        
        for i, result in enumerate(results, 1):
            doc_type = result.get("document_type", "Unknown")
            filename = result.get("original_filename", "Unknown")
            confidence = result.get("confidence", 0) * 100
            
            response_parts.append(f"\n**{i}. {doc_type}** ({filename})")
            response_parts.append(f"   Confidence: {confidence:.0f}%")
            
            # Add relevant fields
            fields = result.get("extracted_fields", {})
            if fields:
                response_parts.append("   Key fields:")
                for field_name, field_data in list(fields.items())[:5]:
                    if isinstance(field_data, dict):
                        value = field_data.get("value", "N/A")
                    else:
                        value = str(field_data)
                    label = field_name.replace("_", " ").title()
                    response_parts.append(f"   - {label}: {value}")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error querying documents: {e}")
        return f"Error querying documents: {str(e)}"


def tool_get_summary(query: str = "") -> str:
    """Get a summary of all processed tax documents."""
    try:
        vs = get_vector_store()
        docs = vs.get_all_documents()
        
        if not docs:
            return "No documents have been processed yet."
        
        response_parts = [f"📋 **Document Summary** ({len(docs)} documents)\n"]
        
        for i, doc in enumerate(docs, 1):
            doc_type = doc.get("document_type", "Unknown")
            filename = doc.get("original_filename", "Unknown")
            confidence = doc.get("overall_confidence", 0) * 100
            needs_verify = doc.get("needs_verification", False)
            
            status = "⚠️ Needs Review" if needs_verify else "✅ Verified"
            response_parts.append(f"{i}. **{doc_type}** - {filename}")
            response_parts.append(f"   {status} | Confidence: {confidence:.0f}%")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error getting summary: {e}")
        return f"Error getting summary: {str(e)}"


def tool_update_field(update_str: str) -> str:
    """
    Update a field value in a stored document.
    Format: "document_identifier:field_name:new_value" 
    Example: "T4:employment_income:50000" or "receipt_1.pdf:total:100.50"
    """
    try:
        import difflib
        
        # 1. Parse Input
        # Fallback for "Update X in Y to Z" patterns could be added here, 
        # but we rely on strict agent instructions for now.
        parts = update_str.split(":", 2)
        if len(parts) != 3:
            # Try to handle "doc field value" space separated as fallback
            parts = update_str.split(" ", 2)
            if len(parts) != 3:
                return "Invalid format. Please use: document_identifier:field_name:new_value"
        
        doc_id, field_name, new_value = parts
        doc_id = doc_id.strip()
        field_name = field_name.strip().lower().replace(" ", "_")
        new_value = new_value.strip()
        
        vs = get_vector_store()
        docs = vs.get_all_documents()
        
        if not docs:
            return "No documents found in database to update."
            
        # 2. Find Document (Fuzzy Matching)
        target_doc = None
        
        # Exact Match on Hash or Filename
        for doc in docs:
            if doc.get("file_hash", "").startswith(doc_id):
                target_doc = doc
                break
            if doc.get("original_filename", "").lower() == doc_id.lower():
                target_doc = doc
                break
        
        # Fuzzy Match on Filename or Document Type if no exact match
        if not target_doc:
            doc_names = [d.get("original_filename", "") for d in docs]
            doc_types = [d.get("document_type", "") for d in docs]
            
            # Check filenames
            matches = difflib.get_close_matches(doc_id, doc_names, n=1, cutoff=0.4)
            if matches:
                 expected_name = matches[0]
                 target_doc = next((d for d in docs if d.get("original_filename") == expected_name), None)
            
            # Check doc types (e.g. user says "T4" -> finds "T4 - Statement...")
            if not target_doc:
                 # Check if doc_id is a substring of any type
                 for doc in docs:
                     dtype = doc.get("document_type", "").lower()
                     if doc_id.lower() in dtype:
                         target_doc = doc
                         break

        if not target_doc:
            available = ", ".join([f"{d.get('original_filename')} ({d.get('document_type')})" for d in docs])
            return f"Document '{doc_id}' not found. Available: {available}"
        
        # 3. Apply Update
        file_hash = target_doc.get("file_hash")
        fields = target_doc.get("extracted_fields", {})
        
        # Normalize new value (naive try to convert to float if looks like number, else string)
        # Actually keep as string to preserve formatting, but strip currency symbols for consistency
        clean_value = new_value.replace("$", "").replace(",", "")
        
        old_value = "N/A"
        
        if field_name not in fields:
            # New Field
            fields[field_name] = {
                "value": new_value,
                "confidence": 1.0,
                "field_name": field_name,
                "bounding_boxes": [],
                "manually_corrected": True
            }
        else:
            # Existing Field
            field_data = fields[field_name]
            if isinstance(field_data, dict):
                old_value = field_data.get("value", "N/A")
                fields[field_name]["value"] = new_value
                fields[field_name]["confidence"] = 1.0
                fields[field_name]["manually_corrected"] = True
            else:
                old_value = str(field_data)
                fields[field_name] = {
                    "value": new_value,
                    "confidence": 1.0,
                    "field_name": field_name,
                    "manually_corrected": True
                }
        
        # 4. Persist
        success = vs.store_document(
            file_hash=file_hash,
            markdown_content=target_doc.get("markdown_content", ""),
            document_type=target_doc.get("document_type", "Unknown"),
            overall_confidence=1.0, # Bump overall confidence since human verified
            needs_verification=False, # Mark as verified
            extracted_fields=fields,
            raw_chunks=target_doc.get("raw_chunks", []),
            original_filename=target_doc.get("original_filename", "Unknown")
        )
        
        if success:
            return f"✅ **SUCCESS**: Updated '{field_name}' in {target_doc.get('original_filename')}.\nValue changed from '{old_value}' -> '{new_value}'."
        else:
            return f"⚠️ **FAILURE**: Could not save changes to database for {field_name}."
            
    except Exception as e:
        logger.error(f"Error updating field: {e}")
        return f"Error updating field: {str(e)}"


def tool_tax_tips(query: str) -> str:
    """Provide tax optimization tips based on the query."""
    
    tips_db = {
        "rrsp": """
💡 **RRSP Tips:**
- Contributions reduce taxable income dollar-for-dollar
- Contribution limit: 18% of previous year's income (max ~$31,560 for 2025)
- Unused room carries forward indefinitely
- Best strategy: Contribute when in high tax bracket, withdraw when in lower bracket
- Deadline: Within 60 days of year end (usually March 1)
""",
        "tuition": """
💡 **Tuition Credit Tips:**
- T2202 tuition amount can be claimed as non-refundable credit
- Federal credit: 15% of tuition amount
- Unused amounts can carry forward indefinitely
- Can transfer up to $5,000 to spouse/parent (federal)
- Keep all tuition receipts and T2202 forms
""",
        "rent": """
💡 **Rent Deductions (Ontario Trillium):**
- Ontario residents can claim rent through Trillium Benefit
- Not a deduction, but a refundable credit
- Must be 18+ and Ontario resident
- Keep all rent receipts and landlord info
- Claimed when filing taxes, paid monthly after
""",
        "fhsa": """
💡 **FHSA (First Home Savings Account):**
- Combines RRSP deduction with TFSA tax-free growth
- Contribution limit: $8,000/year (max $40,000 lifetime)
- Must be first-time home buyer
- Contributions reduce taxable income
- Withdrawals for home purchase are tax-free
""",
        "general": """
💡 **General Tax Tips for Students/Workers:**
1. **Maximize RRSP/FHSA** - Reduce taxable income
2. **Claim all tuition** - Even if you owe no tax, carry forward
3. **Track medical expenses** - May qualify if over 3% of income
4. **Home office** - $2/day flat rate or detailed method
5. **Moving expenses** - If moved 40km+ for work/school
6. **Union/professional dues** - Fully deductible
7. **Interest on student loans** - 15% federal credit
"""
    }
    
    query_lower = query.lower()
    
    # Match query to tips
    for keyword, tip in tips_db.items():
        if keyword in query_lower:
            return tip
    
    return tips_db["general"]


def tool_research_tax_query(query: str) -> str:
    """
    Search BOTH the user's processed documents AND general tax laws/rules.
    Always uses this for research.
    """
    try:
        vs = get_vector_store()
        response_parts = [f"🔎 **Research Results for:** '{query}'\n"]
        sources = []  # Track sources for citations
        
        # 1. Search Personal Documents
        user_docs = vs.search_documents(query, n_results=5, collection_name="tax_documents")
        if user_docs:
            response_parts.append("\n📄 **FROM USER DOCUMENTS (Personal):**")
            for i, result in enumerate(user_docs, 1):
                fname = result.get("original_filename", "Unknown")
                doc_type = result.get("document_type", "Document")
                content = result.get("markdown_content", "")
                
                # Extract key fields for context
                fields = result.get("extracted_fields", {})
                field_summary = ", ".join([f"{k}={v.get('value')}" for k, v in fields.items() if v.get('value')])
                
                response_parts.append(f"{i}. **{fname}** ({doc_type})")
                if field_summary:
                    response_parts.append(f"   Key Data: {field_summary}")
                response_parts.append(f"   Excerpt: {content[:300]}...")
                
                # Add to sources
                sources.append(f"[{fname}] - {doc_type}")
        else:
            response_parts.append("\n📄 **FROM USER DOCUMENTS:** No relevant personal documents found.")

        # 2. Search General Tax Laws
        tax_laws = vs.search_documents(query, n_results=5, collection_name="tax_laws")
        if tax_laws:
            response_parts.append("\n⚖️ **FROM TAX LAWS (General Rules):**")
            for i, result in enumerate(tax_laws, 1):
                title = result.get("extracted_fields", {}).get("title", "Unknown Rule")
                jurisdiction = result.get("extracted_fields", {}).get("jurisdiction", "General")
                content = result.get("markdown_content", "")
                content = content.replace(f"# {title}", "").strip()
                
                response_parts.append(f"{i}. **{title}** ({jurisdiction})")
                response_parts.append(f"   {content[:400]}...")
                
                # Add to sources
                sources.append(f"[{title}] - {jurisdiction} Tax Law")
        else:
            response_parts.append("\n⚖️ **FROM TAX LAWS:** No specific tax laws found.")

        # Add sources section for agent to use (separated by type)
        doc_sources = [s for s in sources if "Tax Law" not in s]
        law_sources = [s for s in sources if "Tax Law" in s]
        
        if doc_sources or law_sources:
            response_parts.append("\n---")
            response_parts.append("**AVAILABLE SOURCES FOR CITATION:**")
            
            if doc_sources:
                response_parts.append("\n*Your Documents:*")
                for src in doc_sources:
                    response_parts.append(f"- {src}")
            
            if law_sources:
                response_parts.append("\n*Tax Laws & Regulations:*")
                for src in law_sources:
                    response_parts.append(f"- {src}")
        
        response_parts.append("\n_Note: Use the personal data for specific calculations, and tax laws for rules/limits._")
        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error in unified search: {e}")
        return f"Error searching data: {str(e)}"


# Global variable to store the last generated proof image path
_last_proof_image_path = None

def get_last_proof_image_path() -> str:
    """Get and clear the last proof image path."""
    global _last_proof_image_path
    path = _last_proof_image_path
    _last_proof_image_path = None  # Clear after reading
    return path


def tool_show_proof(query: str) -> str:
    """
    Show proof/evidence for a specific field value by cropping the document image
    around its bounding box.
    
    Input format: 'field_name' or 'document_name:field_name'
    Examples: 'employment_income', 'T4:employment_income', 'tuition_amount'
    """
    try:
        from PIL import Image
        import os
        from pathlib import Path
        
        vs = get_vector_store()
        all_docs = vs.get_all_documents()
        
        if not all_docs:
            return "❌ No documents found. Please upload some tax documents first."
        
        # Parse input
        parts = query.split(":")
        if len(parts) == 2:
            doc_filter = parts[0].strip().lower()
            field_name = parts[1].strip().lower()
        else:
            doc_filter = None
            field_name = query.strip().lower()
        
        # Search for the field across documents
        found_field = None
        found_doc = None
        
        for doc in all_docs:
            # Check if doc matches filter
            filename = doc.get("original_filename", "").lower()
            doc_type = doc.get("document_type", "").lower()
            
            if doc_filter and doc_filter not in filename and doc_filter not in doc_type:
                continue
            
            # Search fields
            fields = doc.get("extracted_fields", {})
            for fname, fdata in fields.items():
                if field_name in fname.lower() or fname.lower() in field_name:
                    if isinstance(fdata, dict) and fdata.get("bounding_boxes"):
                        found_field = {
                            "name": fname,
                            "value": fdata.get("value"),
                            "confidence": fdata.get("confidence", 0),
                            "bounding_boxes": fdata.get("bounding_boxes", [])
                        }
                        found_doc = doc
                        break
            
            if found_field:
                break
        
        if not found_field:
            return f"❌ Could not find field '{field_name}' with bounding box data. Try: employment_income, tuition_amount, tax_year, etc."
        
        if not found_field["bounding_boxes"]:
            return f"⚠️ Found field '{found_field['name']}' = {found_field['value']} but no bounding box data is available for cropping."
        
        # Get the saved image path (actual file in uploads/)
        original_filename = found_doc.get("original_filename", "")
        saved_path = found_doc.get("saved_path", "")
        
        # Use saved_path if available, otherwise try to find the file
        if saved_path and Path(saved_path).exists():
            image_path = Path(saved_path)
        else:
            # Fallback: search uploads directory
            uploads_dir = Path("./uploads")
            image_path = None
            
            for img_file in uploads_dir.glob("*"):
                if original_filename in img_file.name or img_file.stem in original_filename:
                    image_path = img_file
                    break
            
            # Try matching by recent upload pattern as last resort
            if not image_path:
                for img_file in sorted(uploads_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
                    if img_file.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                        image_path = img_file
                        break
        
        if not image_path or not image_path.exists():
            return f"⚠️ Found field '{found_field['name']}' = {found_field['value']} but cannot locate the original image file for cropping."
        
        # Load the image
        try:
            img = Image.open(image_path)
            img_width, img_height = img.size
        except Exception as e:
            return f"⚠️ Could not open image: {str(e)}"
        
        # Get the first bounding box and crop
        bbox = found_field["bounding_boxes"][0]
        
        # Original bounding box (normalized coordinates 0-1)
        orig_left = bbox.get("l", 0)
        orig_top = bbox.get("t", 0)
        orig_right = bbox.get("r", 1)
        orig_bottom = bbox.get("b", 1)
        
        # Add padding around the bounding box for context
        padding_x = (orig_right - orig_left) * 0.3
        padding_y = (orig_bottom - orig_top) * 0.5
        
        crop_left_norm = max(0, orig_left - padding_x)
        crop_top_norm = max(0, orig_top - padding_y)
        crop_right_norm = min(1, orig_right + padding_x)
        crop_bottom_norm = min(1, orig_bottom + padding_y)
        
        # Convert to pixel coordinates for cropping
        crop_left = int(crop_left_norm * img_width)
        crop_top = int(crop_top_norm * img_height)
        crop_right = int(crop_right_norm * img_width)
        crop_bottom = int(crop_bottom_norm * img_height)
        
        # Crop the image
        cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
        
        # Calculate the bounding box position within the cropped image
        cropped_width, cropped_height = cropped.size
        
        # Convert original bbox to cropped image coordinates
        bbox_left = int((orig_left - crop_left_norm) / (crop_right_norm - crop_left_norm) * cropped_width)
        bbox_top = int((orig_top - crop_top_norm) / (crop_bottom_norm - crop_top_norm) * cropped_height)
        bbox_right = int((orig_right - crop_left_norm) / (crop_right_norm - crop_left_norm) * cropped_width)
        bbox_bottom = int((orig_bottom - crop_top_norm) / (crop_bottom_norm - crop_top_norm) * cropped_height)
        
        # Draw bounding box on the cropped image
        from PIL import ImageDraw
        
        # Create a copy to draw on
        annotated = cropped.copy()
        draw = ImageDraw.Draw(annotated)
        
        # Draw a thick colored rectangle around the field
        box_color = (0, 200, 0)  # Green
        line_width = 4
        
        # Draw rectangle (multiple passes for thickness)
        for i in range(line_width):
            draw.rectangle(
                [bbox_left - i, bbox_top - i, bbox_right + i, bbox_bottom + i],
                outline=box_color
            )
        
        # Save the cropped proof image
        proof_dir = Path("./uploads/proofs")
        proof_dir.mkdir(exist_ok=True)
        
        import uuid
        proof_filename = f"proof_{found_field['name']}_{uuid.uuid4().hex[:8]}.png"
        proof_path = proof_dir / proof_filename
        annotated.save(proof_path)
        
        # Save proof path to a global variable that app.py can access
        # This keeps the path hidden from the LLM (which would try to render it as markdown)
        global _last_proof_image_path
        _last_proof_image_path = str(proof_path.absolute())
        
        # Return success message WITHOUT the image path (LLM shouldn't see it)
        confidence_pct = found_field['confidence'] * 100
        confidence_emoji = "🟢" if confidence_pct >= 85 else "🟠" if confidence_pct >= 70 else "🔴"
        
        result = f"""✅ **Found proof for '{found_field['name']}'**

**Value:** {found_field['value']}
**Confidence:** {confidence_emoji} {confidence_pct:.1f}%
**Document:** {original_filename}

The cropped evidence image has been generated and will be displayed below."""
        return result
        
    except Exception as e:
        logger.error(f"Error showing proof: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error generating proof: {str(e)}"


# ============== AGENT SETUP ==============

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ... [Existing imports remain]

AGENT_PROMPT = """You are a helpful Canadian tax assistant for AuditLens. You help users:
- Calculate their total income and deductions from processed tax documents
- Estimate their federal and provincial taxes
- FIND INFORMATION by researching both user documents and tax laws -> Use `research_tax_query`
- Provide tax optimization tips -> Use `research_tax_query` to check rules first
- Correct any errors in extracted data
- SHOW PROOF/EVIDENCE for any extracted value -> Use `show_proof` when user asks to verify or see proof

**CRITICAL - Query Formulation:**
When calling `research_tax_query`, use SHORT KEYWORD-BASED queries (5-10 words max).
The search uses embeddings - concise queries work best.

✅ GOOD: "noa_2024 tuition carryforward", "T4 employment income", "RRSP room"
❌ BAD: "Find the unused tuition amount carried forward from prior years for the taxpayer"

Include document filename if searching for specific data from a known document.

CRITICAL INSTRUCTION FOR CORRECTIONS:
If the user says a value is wrong or provides a correction (e.g., "The income is actually $50k", "Change the rent to $2000", "Update T4 amount"), you MUST:
1.  **Trust the user.** Do not argue or ask for verification.
2.  **IMMEDIATELY** call the `update_field` tool with the new value.
3.  Do NOT call `research_tax_query` to "check" if they are right. Just update it.

CITATIONS:
At the end of EVERY response, you MUST include a "Sources" section that lists the documents or tax laws you referenced. Separate them into two categories:

**Sources:**

*Your Documents:*
- [Document Name] - Brief description

*Tax Laws & Regulations:*
- [Tax Law/Rule Name] - Brief description

If a category has no sources, omit that category. If no sources were used at all (e.g., for simple greetings), you may omit the entire Sources section.

NOTE: Ensure that the responses are brief and concise. Do not provide unnecessary or extra information unless otherwise asked.
"""



def create_tax_agent() -> Optional[AgentExecutor]:
    """Create and return the tax agent executor."""
    
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not set - agent will not function")
        return None
    
    try:
        # Use GPT-4o with streaming enabled for token-by-token output
        # Get model from env or use default
        model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        
        llm = ChatOpenAI(
            model=model_name,
            temperature=1,
            streaming=True,
            api_key=openai_api_key
        )
        
        # Define tools
        tools = [
            Tool(
                name="calculate_totals",
                func=tool_calculate_totals,
                description="Calculate total income, deductions, and credits from all processed tax documents. No input required."
            ),
            Tool(
                name="estimate_tax",
                func=tool_estimate_tax,
                description="Estimate federal and provincial taxes. Input: income amount (optional)"
            ),
            Tool(
                name="research_tax_query",
                func=tool_research_tax_query,
                description="Search BOTH personal documents and general tax laws. Use this for ANY question requiring information lookup (e.g., 'my income', 'RRSP deadline', 'can I deduct X'). Input: search query"
            ),
            Tool(
                name="get_summary",
                func=tool_get_summary,
                description="Get a summary of all processed tax documents. No input required."
            ),
            Tool(
                name="update_field",
                func=tool_update_field,
                description="Correct or update a field value in a stored document. Input format: 'filename:field_name:new_value'"
            ),
            Tool(
                name="tax_tips",
                func=tool_tax_tips,
                description="Get generic tax optimization tips. Input: topic"
            ),
            Tool(
                name="show_proof",
                func=tool_show_proof,
                description="Show proof/evidence for an extracted field by cropping the document image around it. Use when user asks for proof, evidence, or wants to verify a number. Input: field_name (e.g., 'employment_income', 'tuition_amount', 'tax_year')"
            )
        ]
        
        # Create prompt with chat history support
        prompt = ChatPromptTemplate.from_messages([
            ("system", AGENT_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        
        # Create executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )
        
        logger.info("Tax agent created successfully (Unified Search)")
        return agent_executor
        
    except Exception as e:
        logger.error(f"Failed to create tax agent: {e}")
        return None


# Singleton agent instance
_tax_agent: Optional[AgentExecutor] = None


def reset_tax_agent():
    """Reset the cached agent instance. Call this when model config changes."""
    global _tax_agent
    _tax_agent = None
    logger.info("Tax agent cache cleared")


def get_tax_agent() -> Optional[AgentExecutor]:
    """Get the singleton tax agent instance."""
    global _tax_agent
    if _tax_agent is None:
        _tax_agent = create_tax_agent()
    return _tax_agent


async def run_agent(query: str, chat_history: list = None, callbacks: Optional[List[Any]] = None) -> str:
    """Run the tax agent with a query and return the response.
    
    Args:
        query: The user's question or command
        chat_history: List of previous messages (HumanMessage/AIMessage objects)
        callbacks: Optional LangChain callbacks
    """
    agent = get_tax_agent()
    
    if agent is None:
        return "❌ Tax agent is not available. Please check your OpenAI API key."
    
    try:
        result = await agent.ainvoke(
            {
                "input": query,
                "chat_history": chat_history or []
            },
            config={"callbacks": callbacks}
        )
        return result.get("output", "No response from agent.")
    except Exception as e:
        logger.error(f"Agent error: {e}")
        return f"❌ Agent error: {str(e)}"
